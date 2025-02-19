import os
import time
import wandb

import numpy as np
from time import gmtime, strftime
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.cuda.amp import GradScaler, autocast
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

import utils
from data_utils import Dataset_Main, Dataset_Speaker, Collate, Collate_Speaker
from models.timbre import ECAPA, ECAPA2, ECAPA3
import modules.commons as commons
from modules.losses import discriminator_loss, feature_loss, generator_loss, kl_loss
from modules.mel_processing import mel_spectrogram_torch, spec_to_mel_torch

class Solver(object):
    def __init__(self, args):
        self.args = args
        self.wandb = wandb.init(entity='jjlee0721', project='libri', group=args.base_args.exp_name, config=args)
        self.global_step = 0
    
    def build_dataset(self, args):
        mp_context = torch.multiprocessing.get_context('fork')
        args.train.batch_size = int(args.train.batch_size / args.base_args.ngpus_per_node)
        if 'voveve' in args.base_args.exp_name:
            self.trainset = Dataset_Speaker(args.data.data_root, args.data.filelist_root, args.data.label_root, typ="train")
            # self.trainset = Dataset_Speaker(args.data.data_root, args.data.filelist_root, args.data.label_root, typ="full")
            collate_fn = Collate_Speaker()
        else:
            self.trainset = Dataset_Main(args.data.data_root, args.data.filelist_root, args.data.label_root, typ="train")
            # self.trainset = Dataset_Main(args.data.data_root, args.data.filelist_root, args.data.label_root, typ="full")
            collate_fn = Collate()
        self.validset = Dataset_Main(args.data.data_root, args.data.filelist_root, args.data.label_root, typ="valid")
        # self.validset = Dataset_Main(args.data.data_root, args.data.filelist_root, args.data.label_root, typ="full_test")
        
        self.train_sampler = DistributedSampler(self.trainset, shuffle=True, rank=self.args.base_args.gpu)
        self.train_loader = DataLoader(self.trainset, num_workers=args.base_args.workers, shuffle=False, pin_memory=True,
                                batch_size=args.train.batch_size, collate_fn=collate_fn,
                                multiprocessing_context=mp_context, sampler=self.train_sampler, drop_last=True,
                                worker_init_fn=lambda _: np.random.seed(int(torch.initial_seed())%(2**32-1)))
        self.valid_sampler = DistributedSampler(self.validset, shuffle=True, rank=self.args.base_args.gpu)
        self.valid_loader = DataLoader(self.validset, num_workers=args.base_args.workers, shuffle=False, pin_memory=True,
                                batch_size=args.train.batch_size, collate_fn=Collate(),
                                multiprocessing_context=mp_context, sampler=self.valid_sampler, drop_last=True,
                                worker_init_fn=lambda _: np.random.seed(int(torch.initial_seed())%(2**32-1)))        
        self.max_iter = len(self.train_loader)
        
    def build_models(self, args):
        torch.cuda.set_device(self.args.base_args.gpu)
        if 'voveve' in args.base_args.exp_name:
            net_g = ECAPA2(**args.model)
        else:
            net_g = ECAPA(**args.model)
        net_g = net_g.to('cuda:{}'.format(self.args.base_args.gpu))
        net_g = DDP(net_g, device_ids=[self.args.base_args.gpu], output_device=self.args.base_args.gpu, find_unused_parameters=True)
        self.net = {'g':net_g}
    
    def build_optimizers(self, args):
        optim_g = torch.optim.AdamW(self.net['g'].parameters(), args.train.learning_rate,
                                    betas=args.train.betas, eps=args.train.eps)
        self.optim = {'g':optim_g}
        
        self.warmup_epoch = args.train.warmup_epochs
        self.ce_loss = nn.CrossEntropyLoss()
        self.ce_loss2 = nn.CrossEntropyLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.nll_loss = nn.NLLLoss()
        
    def build_scheduler(self, args, epoch_str):
        scheduler_g = torch.optim.lr_scheduler.ExponentialLR(self.optim['g'], gamma=args.train.lr_decay, last_epoch=epoch_str - 2)
        self.scheduler = {'g':scheduler_g}
    
    def train(self, args, epoch):
        self.net['g'].train()
        if epoch <= self.warmup_epoch:
            for param_group in self.optim['g'].param_groups:
                param_group['lr'] = args.train.learning_rate / self.warmup_epoch * epoch

        for batch_idx, items in enumerate(self.train_loader):
            if 'voveve' in args.base_args.exp_name:
                losses = self.loss_generator2(args, items, phase="train")
            else:
                losses = self.loss_generator(args, items, phase="train")
            # train log
            if args.base_args.rank % args.base_args.ngpus_per_node == 0:
                if self.global_step % args.train.log_interval == 0:
                    print("\r[Epoch:{:3d}, {:.0f}%, Step:{}] [Loss G:{:.5f}] [{}]"
                        .format(epoch, 100.*batch_idx/self.max_iter, self.global_step, losses['gen/total'], strftime('%Y-%m-%d %H:%M:%S', gmtime())))
                    if args.base_args.test != 1:
                        self.wandb_log(losses, epoch, "train")
            if args.base_args.test:
                if batch_idx > 10:
                    break
            self.global_step += 1
            
    def validation(self, args, epoch):
        with torch.no_grad():
            self.net['g'].eval()
            for batch_idx, items in enumerate(self.valid_loader):
                losses = self.loss_generator(args, items, phase="valid")
                if args.base_args.test:
                    if batch_idx > 10:
                        break
                # validation log
                if args.base_args.rank % args.base_args.ngpus_per_node == 0:
                    print("\r[Validation Epoch:{:3d}] [Loss G:{:.5f}]".format(epoch, losses['gen/total']))
                    if args.base_args.test != 1:
                        self.wandb_log(losses, epoch, "valid")
    
    def loss_generator(self, args, items, phase="train") -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        y, label, lengths = items
        y = y.cuda(args.base_args.rank, non_blocking=True)
        label = label.cuda(args.base_args.rank, non_blocking=True)
        lengths = lengths.cuda(args.base_args.rank, non_blocking=True)
        
        class_emb, _ = self.net['g'](y)
        loss_cls = self.ce_loss(class_emb, label)
        loss_total = loss_cls
        
        if phase == "train":   
            self.optim['g'].zero_grad()
            loss_total.backward()
            self.optim['g'].step()
        
        losses = {
            'gen/cls': loss_cls.item(),
            'gen/total': loss_total.item()
            }
        return losses
                            
    def loss_generator2(self, args, items, phase="train") -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        y, label, lengths, speakers = items
        y = y.cuda(args.base_args.rank, non_blocking=True)
        label = label.cuda(args.base_args.rank, non_blocking=True)
        lengths = lengths.cuda(args.base_args.rank, non_blocking=True)
        speakers = speakers.cuda(args.base_args.rank, non_blocking=True)
        
        class_emb, speaker_emb = self.net['g'](y)
        if 'voveveve' in args.base_args.exp_name:
            loss_cls = self.bce_loss(class_emb, label)
        else:
            loss_cls = self.ce_loss(class_emb, label)
        loss_spk = self.ce_loss2(speaker_emb, speakers)
        loss_total = loss_cls + loss_spk
        
        if phase == "train":   
            self.optim['g'].zero_grad()
            loss_total.backward()
            self.optim['g'].step()
        
        losses = {
            'gen/cls': loss_cls.item(),
            'gen/total': loss_total.item(),
            'gen/speaker': loss_spk.item()
            }
        return losses
                
    def wandb_log(self, loss_dict, epoch, phase="train"):
        wandb_dict = {}
        wandb_dict.update(loss_dict)
        wandb_dict.update({"epoch":epoch})
        if phase == "train":
            wandb_dict.update({"global_step": self.global_step})
            with torch.no_grad():
                grad_norm = np.mean([
                    torch.norm(p.grad).item() for p in self.net['g'].parameters() if p.grad is not None])
                param_norm = np.mean([
                    torch.norm(p).item() for p in self.net['g'].parameters() if p.dtype == torch.float32])
            wandb_dict.update({ "common/grad-norm":grad_norm, "common/param-norm":param_norm})
            wandb_dict.update({ "common/learning-rate-g":self.optim['g'].param_groups[0]['lr']})
        elif phase == "valid":
            wandb_dict = dict(('valid/'+ key, np.mean(value)) for (key, value) in wandb_dict.items())
        self.wandb.log(wandb_dict)
