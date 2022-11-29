import json
import time
import argparse
import random
import numpy as np
import os
import torch
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from datetime import datetime

from plb.envs import make
from plb.algorithms.logger import Logger
from plb.algorithms.discor.run_sac import train as train_sac
from plb.algorithms.ppo.run_ppo import train_ppo
from plb.algorithms.TD3.run_td3 import train_td3
from plb.optimizer.solver import solve_action
from plb.optimizer.solver_nn import solve_nn

os.environ['TI_USE_UNIFIED_MEMORY'] = '0'
os.environ['TI_DEVICE_MEMORY_FRACTION'] = '0.9'
os.environ['TI_DEVICE_MEMORY_GB'] = '4'
os.environ['TI_ENABLE_CUDA'] = '0'
os.environ['TI_ENABLE_OPENGL'] = '0'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

RL_ALGOS = ['sac', 'td3', 'ppo']
DIFF_ALGOS = ['action', 'nn']

STEPS=100

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_args():
    parser=argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, default='action')
    parser.add_argument("--env_name", type=str, default="push_box-v1")
    parser.add_argument("--task_name", type=str, default="two_roomba_deform_one_rope")
    parser.add_argument("--task_version", type=str, default="push_box")
    parser.add_argument("--picture_name", type=str, default="pose")
    parser.add_argument("--path", type=str, default='./output')
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rope_seed", type=int, default=0)
    parser.add_argument("--sdf_loss", type=float, default=500)
    parser.add_argument("--density_loss", type=float, default=500)
    parser.add_argument("--contact_loss", type=float, default=1)
    parser.add_argument("--soft_contact_loss", action='store_true')
    parser.add_argument("--real_demo", action='store_true')
    parser.add_argument("--create_grid_mass", action='store_true')
    parser.add_argument("--num_steps", type=int, default=None)
    parser.add_argument("--create_dataset", action='store_true')

    # differentiable physics parameters
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--softness", type=float, default=6666.)
    parser.add_argument("--optim", type=str, default='Adam', choices=['Adam', 'Momentum'])

    args=parser.parse_args()

    return args

def main(args):
    if args.num_steps is None:
        if args.algo in DIFF_ALGOS:
            args.num_steps = STEPS * 100
        else:
            args.num_steps = STEPS * 10000

    if args.create_grid_mass:
        output_dir = '/root/roomba_hack/pbm/plb/envs/assets'
        os.makedirs(output_dir, exist_ok=True)
        np.save(f'{output_dir}/{args.env_name}.npy', np.zeros((64, 64, 64)))
        env = make(args.env_name, nn=(args.algo=='nn'), sdf_loss=args.sdf_loss,
                            density_loss=args.density_loss, contact_loss=args.contact_loss,
                            soft_contact_loss=args.soft_contact_loss)
        env.seed(args.seed)
        env.reset()
        env.create_grid_mass()
        return

    if args.algo in ['action', 'ppo']:
        args.path = f'./{args.path}/{args.task_name}/{args.task_version}/{args.algo}/{args.picture_name}_{STEPS}_{args.num_steps}_{args.sdf_loss}_{args.density_loss}_{args.contact_loss}_{datetime.now().strftime("%Y%m%d_%H%M%S")}_{args.seed}'
    else:
        raise NotImplementedError
    
    print(f'Saving outputs to {args.path}')
    os.makedirs(args.path, exist_ok=True)
    with open(f'{args.path}/args.txt', mode="w") as f:
        json.dump(args.__dict__, f, indent=4)

    logger = Logger(args.path)
    set_random_seed(args.seed)
    env = make(args.env_name, nn=(args.algo=='nn'), sdf_loss=args.sdf_loss,
                            density_loss=args.density_loss, contact_loss=args.contact_loss,
                            soft_contact_loss=args.soft_contact_loss)
    env.seed(args.seed)
    env._max_episode_steps = STEPS

    if args.algo == 'sac':
        train_sac(env, args.path, logger, args)
    elif args.algo == 'action':
        solve_action(env, args.path, logger, args)
    elif args.algo == 'ppo':
        train_ppo(env, args.path, logger, args)
    elif args.algo == 'td3':
        train_td3(env, args.path, logger, args)
    elif args.algo == 'nn':
        solve_nn(env, args.path, logger, args)
    else:
        raise NotImplementedError

if __name__ == '__main__':
    args = get_args()
    idx = args.env_name.find('-')
    args.task_name = args.env_name[:idx]
    args.task_version = args.env_name[(idx+1):]
    main(args)
