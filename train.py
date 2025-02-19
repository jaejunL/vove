import os
import time
import wandb

import torch
import torch.multiprocessing as mp
import torch.distributed as dist

from solver import Solver
import utils

def train(args, run=None):
    wandb.require(experiment="service")
    wandb.setup()
    
    ngpus_per_node = int(torch.cuda.device_count()/args.base_args.n_nodes)
    print("use {} gpu mac hine".format(ngpus_per_node))
    args.base_args.world_size = ngpus_per_node * args.base_args.n_nodes
    
    solver = Solver(args)
    mp.spawn(worker, nprocs=ngpus_per_node, args=(solver, ngpus_per_node, args))

def worker(gpu, solver, ngpus_per_node, args):
    args.base_args.rank = args.base_args.rank * ngpus_per_node + gpu
    dist.init_process_group(backend='nccl', 
                            world_size=args.base_args.world_size,
                            init_method='env://',
                            rank=args.base_args.rank)
    args.base_args.gpu = gpu
    args.base_args.ngpus_per_node = ngpus_per_node

    solver.build_dataset(args)
    solver.build_models(args)
    solver.build_optimizers(args)

    # Loading
    try:
        _, _, _, epoch_str = utils.load_checkpoint(utils.latest_checkpoint_path(os.path.join(args.base_args.base_dir, 'checkpoints'), "G_*.pth"), solver.net['g'], solver.optim['g'])
        print("Pretrained Model is Loaded")
        solver.global_step = int(epoch_str * len(solver.train_loader))
    except:
        epoch_str = 1
    epoch_str = max(epoch_str, 1)
    
    solver.build_scheduler(args, epoch_str)

    if args.base_args.rank % ngpus_per_node == 0:
        print("start from epoch {}".format(epoch_str))

    for epoch in range(epoch_str, args.train.epochs + 1):
        start_time = time.time()
        solver.train_sampler.set_epoch(epoch)

        solver.train(args, epoch)
        solver.validation(args, epoch)
        
        solver.scheduler['g'].step()
        
        # save checkpoint
        if args.base_args.rank % ngpus_per_node == 0:
            if epoch % args.train.save_model_interval == 0 and args.base_args.test != 1:
                checkpoint_dir = os.path.join(args.base_args.base_dir, 'checkpoints')
                utils.save_checkpoint(solver.net['g'], solver.optim['g'], args.train.learning_rate, epoch,
                            os.path.join(checkpoint_dir, "G_{}.pth".format(epoch)))
            end_time = time.time()
            print(f'Training time:{end_time-start_time:.1f} sec')
        time.sleep(5)
        
if __name__ == "__main__":
    print("This is 'train.py' code")