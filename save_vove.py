import os
import sys
import glob
import json
import random
import argparse
import numpy as np

import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import DataLoader

import utils
from models.timbre import ECAPA, ECAPA2, ECAPA3
from data_utils import Dataset_Main, Collate
from utils import load_filepaths_and_text, load_wav_to_torch

if __name__ == "__main__":
    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning) 
    warnings.simplefilter(action='ignore', category=UserWarning) 
    
    parser = argparse.ArgumentParser()
    # set parameters
    parser.add_argument('--write_root', type=str, default="/disk3/jaejun/libri", help='Model saving directory')
    parser.add_argument('--exp_name', default="full", type=str, help="experiment name")
    parser.add_argument('--ckpt', default="5", type=str)
    parser.add_argument('--config_name', default="ecapa", type=str, help="experiment name")
    base_args = parser.parse_args()
    
    base_args.base_dir = os.path.join(base_args.write_root, base_args.exp_name)

    configs_dir = f'configs/ecapa.json'
    with open(configs_dir, "r") as f:
        data = f.read()
    config = json.loads(data)
    args = utils.HParams(**config)
    args.base_args = base_args

    ckpt_path = os.path.join(base_args.write_root, base_args.exp_name, f'checkpoints/G_{base_args.ckpt}.pth')
            
    if 'voveve' in args.base_args.exp_name:
        net_g = ECAPA2(**args.model)
    else:
        net_g = ECAPA(**args.model)
    net_g, _, _, _ = utils.load_checkpoint(ckpt_path, net_g, None)
    net_g = net_g.to('cuda:0')
    net_g = net_g.eval()
    
    wav_root = "/disk2/vctk/modified/wav16"

    wav_dirs = glob.glob(os.path.join(wav_root, '*', '*.wav'))

    start_idx = 0
    end_idx = len(wav_dirs)
    print(end_idx-start_idx)

    cnt = 0
    for i, wav_dir in enumerate(wav_dirs):
        if i < start_idx or i > end_idx:
            continue
        save_dir = wav_dir.replace('/wav16/',f'/{args.base_args.exp_name}{args.base_args.ckpt}/').replace('.wav', '.npy')
        os.makedirs(os.path.dirname(save_dir), exist_ok=True)
        # if os.path.exists(save_dir):
            # continue

        audio, sampling_rate = load_wav_to_torch(wav_dir)
        audio_norm = audio / 32768.0
        audio_norm = audio_norm.unsqueeze(0)
        
        with torch.no_grad():
            pred, _ = net_g(audio_norm.cuda())
        np.save(save_dir, pred.detach().cpu().squeeze().numpy())
        
        print(f'idx:{str(i)}, {np.round(100*i/(end_idx-start_idx+1),1)}%', end='\r')
        
# CUDA_VISIBLE_DEVICES=2 python save_vove.py --exp_name=voveveve_full --ckpt=20
# CUDA_VISIBLE_DEVICES=3 python save_vove.py --exp_name=voveveve_full --ckpt=30

# 25.02.17
# CUDA_VISIBLE_DEVICES=10 python save_vove.py --exp_name=voveveve --ckpt=5
# CUDA_VISIBLE_DEVICES=10 python save_vove.py --exp_name=voveveve --ckpt=10
# CUDA_VISIBLE_DEVICES=11 python save_vove.py --exp_name=voveveve --ckpt=20
# CUDA_VISIBLE_DEVICES=11 python save_vove.py --exp_name=voveveve --ckpt=30
