import argparse
import os
import pickle
import random
import sys
import time
from datetime import datetime

import jax
import jax.numpy as jnp
import numpy as np
import optax
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), '../../bc/network'))
from transformer.bc_mlp.model import make_policy_networks
from transformer.bc_mlp.utils import (NormalTanhDistribution,
                                               ReplayBuffer, TrainingState,
                                               Transition, evaluate_on_env,
                                               get_d4rl_normalized_score,
                                               load_params, save_params)
from transformer.pmap import (bcast_local_devices, is_replicated,
                                       synchronize_hosts)

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from plb.algorithms.discor.run_sac import train as train_sac
from plb.algorithms.logger import Logger
from plb.algorithms.ppo.run_ppo import train_ppo
from plb.algorithms.TD3.run_td3 import train_td3
from plb.envs import make
from plb.optimizer.solver import optimize_mppi, solve_action, solve_mppi
from plb.optimizer.solver_nn import solve_nn

STEPS=100


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_args():
    parser=argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, default='action')
    parser.add_argument("--env_name", type=str, default="multi_bc_rope-v1")
    parser.add_argument("--task_name", type=str, default="multi_bc_rope")
    parser.add_argument("--task_version", type=str, default="v1")
    parser.add_argument("--picture_name", type=str, default="pose")
    parser.add_argument("--path", type=str, default='./output')
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sdf_loss", type=float, default=500)
    parser.add_argument("--density_loss", type=float, default=500)
    parser.add_argument("--contact_loss", type=float, default=1)
    parser.add_argument("--soft_contact_loss", action='store_true')
    parser.add_argument("--create_grid_mass", action='store_true')
    parser.add_argument("--num_steps", type=int, default=None)
    
    # policy
    parser.add_argument("--dataset", type=str, default='medium')
    parser.add_argument("--max_eval_ep_len", type=int, default=115)
    parser.add_argument("--num_eval_ep", type=int, default=1)
    parser.add_argument("--dataset_dir", type=str, default='data/')
    parser.add_argument("--log_dir", type=str, default='dt_runs/')
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--grad_updates_per_step", type=int, default=1)
    parser.add_argument("--max_train_iters", type=int, default=200000000000)
    parser.add_argument("--num_updates_per_iter", type=int, default=5000)
    parser.add_argument("--policy_save_iters", type=int, default=10)
    parser.add_argument("--rm_normalization", action='store_true', help='Turn off input normalization')
    parser.add_argument("--policy_params_path", type=str, default='/root/roomba_hack/pbm/bc/network/dt_runs/bc_mlpmulti_bc_rope_6/seed_0/22-10-04-09-04-39/model_4000.pt')
    parser.add_argument("--max_devices_per_host", type=int, default=None)
    parser.add_argument('--base_data_size', type=int, default=102)

    # differentiable physics parameters
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--softness", type=float, default=6666.)
    parser.add_argument("--optim", type=str, default='Adam', choices=['Adam', 'Momentum'])

    args=parser.parse_args()

    return args

def main(args):
    set_random_seed(args.seed)
    env = make(args.env_name, nn=(args.algo=='nn'), sdf_loss=args.sdf_loss,
                            density_loss=args.density_loss, contact_loss=args.contact_loss,
                            soft_contact_loss=args.soft_contact_loss)
    env.seed(args.seed)
    env._max_episode_steps = STEPS
    
    # device settings
    max_devices_per_host = args.max_devices_per_host
    process_count = jax.process_count()
    process_id = jax.process_index()
    local_device_count = jax.local_device_count()
    local_devices_to_use = local_device_count
    if max_devices_per_host:
        local_devices_to_use = min(local_devices_to_use, max_devices_per_host)
    print(f'Device count: {jax.device_count()}, process count: {process_count} (id {process_id}), local device count: {local_device_count}, devices to be used count: {local_devices_to_use}')

    # seed for jax
    seed = args.seed
    key = jax.random.PRNGKey(seed)
    global_key, local_key, test_key = jax.random.split(key, 3)
    del key
    local_key = jax.random.fold_in(local_key, process_id)
    # seed for others
    random.seed(seed)
    np.random.seed(seed)
    env.seed(seed)

    max_eval_ep_len = args.max_eval_ep_len  # max len of one episode

    lr = args.lr                            # learning rate
    hidden_size = args.hidden_size          # hidden size
    num_layers = args.num_layers            # num of layers for MLP policy

    # load data from this file
    dataset_path = f'/root/roomba_hack/pbm/bc/network/data/bc_data_all.pkl'

    # saves model and csv in this directory
    log_dir = args.policy_params_path[:-3]
    os.makedirs(log_dir, exist_ok=True)

    start_time = datetime.now().replace(microsecond=0)
    start_time_str = start_time.strftime("%y-%m-%d-%H-%M-%S")

    print("=" * 60)
    print("start time: " + start_time_str)
    print("=" * 60)
    print("dataset path: " + dataset_path)

    # load dataset
    with open(dataset_path, 'rb') as f:
        trajectories = pickle.load(f)
        print('data length-----------------------', len(trajectories))

    primitive_num = int(env.action_space.shape[0]/3)
    state_dim = (args.base_data_size*3*2+primitive_num*3)*primitive_num+args.base_data_size*3
    act_dim = int(primitive_num*2)

    # used for input normalization
    state_stats = jnp.concatenate([traj['observations'] for traj in trajectories], axis=0)
    state_mean, state_std = jnp.mean(state_stats, axis=0), jnp.std(state_stats, axis=0) + 1e-8

    policy_model = make_policy_networks(
        policy_params_size=act_dim,
        state_dim=state_dim,
        hidden_layer_sizes=tuple([hidden_size for _ in range(num_layers)]),
    )
    parametric_action_distribution = NormalTanhDistribution(event_size=act_dim)

    policy_optimizer = optax.adam(learning_rate=lr)
    policy_params = load_params(args.policy_params_path)
    policy_optimizer_state = policy_optimizer.init(policy_params)

    # count the number of parameters
    param_count = sum(x.size for x in jax.tree_leaves(policy_params))
    print(f'num_policy_param: {param_count}')

    policy_optimizer_state, policy_params = bcast_local_devices(
        (policy_optimizer_state, policy_params), local_devices_to_use)

    training_state = TrainingState(
        policy_optimizer_state=policy_optimizer_state,
        policy_params=policy_params,
        key=jnp.stack(jax.random.split(local_key, local_devices_to_use)),
        actor_steps=jnp.zeros((local_devices_to_use,)))

    results, actions = evaluate_on_env(policy_model,
                            training_state.policy_params,
                            parametric_action_distribution,
                            env,
                            None,
                            1,
                            max_eval_ep_len,
                            state_mean,
                            state_std,
                            args,
                            log_dir,
                            giff_save_iters=1)

if __name__ == '__main__':
    start_time = time.time()
    args = get_args()
    idx = args.env_name.find('-')
    args.task_name = args.env_name[:idx]
    args.task_version = args.env_name[(idx+1):]
    main(args)
    with open(f'{args.path}/time.txt', 'w') as f:
        f.write(str(time.time() - start_time))
