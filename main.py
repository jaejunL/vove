import os
import json
import argparse
import datetime
from train import train
import utils

if __name__ == "__main__":
    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning) 
    warnings.simplefilter(action='ignore', category=UserWarning) 
    
    now = datetime.datetime.now()
    nowDatetime = now.strftime('%Y-%m-%d %H:%M:%S')
    
    parser = argparse.ArgumentParser()
    # set parameters
    parser.add_argument('--write_root', type=str, default="/disk3/jaejun/libri", help='Model saving directory')
    parser.add_argument('--exp_name', default="base", type=str, help="experiment name")
    parser.add_argument('--config_name', default="base", type=str, help="experiment name")    
    parser.add_argument('--test', default='false', type=utils.str2bool, help='whether test or not')
    # gpu parameters
    parser.add_argument('--gpus', nargs='+', default=None, help='gpus')
    parser.add_argument('--port', default='6000', type=str, help='port')
    parser.add_argument('--n_nodes', default=1, type=int)
    parser.add_argument('--workers', default=8, type=int) # n개의 gpu가 한 node: n개의 gpu마다 main_worker를 실행시킨다.
    parser.add_argument('--rank', default=0, type=int, help='ranking within the nodes')
    base_args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(gpu_num) for gpu_num in base_args.gpus])
    os.environ['MASTER_ADDR'] = "127.0.0.1"
    os.environ['MASTER_PORT'] = base_args.port
    if base_args.test == True:
        os.environ['WANDB_MODE'] = "dryrun"
    os.environ['WANDB_RUN_ID'] = f'Libri_{base_args.exp_name}'

    base_args.base_dir = os.path.join(base_args.write_root, base_args.exp_name)
    os.makedirs(os.path.join(base_args.base_dir, 'checkpoints'), exist_ok=True)
    configs_dir = f'configs/{base_args.config_name}.json'
    with open(configs_dir, "r") as f:
        data = f.read()
    config = json.loads(data)
    
    args = utils.HParams(**config)
    args.base_args = base_args

    train(args)

# python main.py --write_root=/disk3/jaejun/libri --exp_name=base --config_name=ecapa --gpus=0,1 --port=0001 --test=1
# python main.py --write_root=/disk3/jaejun/libri --exp_name=full --config_name=ecapa --gpus=2,3 --port=0203 --test=1
# python main.py --write_root=/disk3/jaejun/libri --exp_name=full2 --config_name=ecapa --gpus=2,3 --port=0203 --test=1

# 25.02.14
# python main.py --write_root=/disk3/jaejun/libri --exp_name=voveve --config_name=ecapa --gpus=4,5 --port=0405 --test=1
# python main.py --write_root=/disk3/jaejun/libri --exp_name=voveveve --config_name=ecapa --gpus=2,3 --port=0203 --test=1
# python main.py --write_root=/disk3/jaejun/libri --exp_name=voveve_full --config_name=ecapa --gpus=0,1 --port=0001 --test=1
# python main.py --write_root=/disk3/jaejun/libri --exp_name=voveveve_full --config_name=ecapa --gpus=10,11 --port=1011 --test=1

# python main.py --write_root=/disk3/jaejun/libri --exp_name=voveve2 --config_name=ecapa --gpus=4,5 --port=0405 --test=1

# 25.02.15
# python main.py --write_root=/disk3/jaejun/libri --exp_name=voveveve --config_name=ecapa --gpus=10,11 --port=1011 --test=1
# python main.py --write_root=/disk3/jaejun/libri --exp_name=voveveve_full --config_name=ecapa --gpus=4,5 --port=0405 --test=1

# 25.02.17
# python main.py --write_root=/disk3/jaejun/libri --exp_name=voveveve --config_name=ecapa --gpus=2,3 --port=0203 --test=1
# python main.py --write_root=/disk3/jaejun/libri --exp_name=voveveve_full --config_name=ecapa --gpus=10,11 --port=1011 --test=1
