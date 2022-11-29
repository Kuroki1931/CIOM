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
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from PIL import Image

sys.path.append(os.path.join(os.path.dirname(__file__), '../../bc/jax_decision_transformer'))
from decision_transformer.bc_transformer_nearest_random_delay.mask_config import \
    MASK_CONFIG
from decision_transformer.bc_transformer_nearest_random_delay.model import \
    make_transformers
from decision_transformer.bc_transformer_nearest_random_delay.utils import (
    NormalTanhDistribution, ReplayBuffer, TrainingState, Transition,
    evaluate_on_env_random_delay, load_params)
from decision_transformer.pmap import (bcast_local_devices, is_replicated,
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
    parser.add_argument("--env_name", type=str, default="multi_bc_rope-v1500")
    parser.add_argument("--task_name", type=str, default="multi_bc_rope")
    parser.add_argument("--task_version", type=str, default="v1500")
    parser.add_argument("--picture_name", type=str, default="pose")
    parser.add_argument("--path", type=str, default='./output')
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sdf_loss", type=float, default=500)
    parser.add_argument("--density_loss", type=float, default=500)
    parser.add_argument("--contact_loss", type=float, default=1)
    parser.add_argument("--soft_contact_loss", action='store_true')
    parser.add_argument("--create_grid_mass", action='store_true')
    parser.add_argument("--real_demo", action='store_true')
    parser.add_argument("--num_steps", type=int, default=None)
    
    # policy
    parser.add_argument("--dataset", type=str, default='medium')
    parser.add_argument("--max_eval_ep_len", type=int, default=115)
    parser.add_argument("--num_eval_ep", type=int, default=1)
    parser.add_argument("--dataset_dir", type=str, default='data/')
    parser.add_argument("--log_dir", type=str, default='dt_runs/')
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--grad_updates_per_step", type=int, default=1)
    parser.add_argument("--max_train_iters", type=int, default=200000000000)
    parser.add_argument("--num_updates_per_iter", type=int, default=500)
    parser.add_argument("--policy_save_iters", type=int, default=10)
    parser.add_argument("--rm_normalization", action='store_true', help='Turn off input normalization')
    parser.add_argument("--policy_params_path", type=str, default='/root/roomba_hack/pbm/bc/jax_decision_transformer/dt_runs/bc_transformer_random_delay_10_multi_bc_rope_6/seed_0/22-11-25-09-28-21/model_5000.pt')
    parser.add_argument("--max_devices_per_host", type=int, default=None)
    parser.add_argument('--base_data_size', type=int, default=102)
    parser.add_argument('--select_layer', type=int, default=2)
    parser.add_argument('--select_head', type=int, default=0)
    parser.add_argument('--matrix_freq', type=int, default=20)
    parser.add_argument('--delay_time', type=int, default=10)

    # differentiable physics parameters
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--softness", type=float, default=6666.)
    parser.add_argument("--optim", type=str, default='Adam', choices=['Adam', 'Momentum'])

    args=parser.parse_args()

    return args


def visualize_mtrix(attn_weights_list, primitive_num, log_dir):
    # setting
    args.matrix_freq = 1
    mt_fig = plt.figure(figsize=(16, 16))
    layer_size = len(attn_weights_list[0])
    head_size = attn_weights_list[0][0].shape[1]
    assert args.select_layer < layer_size, f'max layer size {layer_size-1}, but you give me {args.select_layer}'
    assert args.select_head < head_size, f'max head size {head_size-1}, but you give me {args.select_head}'
    # extract matrix we focus on
    mt_list = []
    for attn_weight in attn_weights_list:
        lay_wt = attn_weight[args.select_layer]
        if args.select_head == -1:
            head_mt = jnp.mean(lay_wt[0], axis=0)
        else:
            head_mt = lay_wt[0][args.select_head]
        mt_list.append(head_mt)
    length = mt_list[0].shape[0]
    num_scale = [i for i in range(length)]
    scale = [i for i in range(primitive_num*2)]

    mt_output_path = os.path.join(log_dir, f'Matrix_head_{args.env_name}_{args.select_head}_layer{args.select_layer}.pdf')
    pp = PdfPages(mt_output_path)
    for j in range(len(mt_list) // args.matrix_freq):
        mt_part = mt_list[args.matrix_freq*j:args.matrix_freq*(j+1)]
        mt = sum(mt_part) / args.matrix_freq
        mt_ax = mt_fig.add_subplot(3, 3, j%9+1)
        mt_ax.set_xticks(num_scale)
        mt_ax.set_xticklabels(scale, rotation=90)
        mt_ax.set_yticks(num_scale)
        mt_ax.set_yticklabels(scale)
        mt_ax.set_xlabel(f'Matrix \n {args.matrix_freq*j} - {args.matrix_freq*(j+1)}')
        img = mt_ax.imshow(mt, cmap="Blues")
        mt_fig.colorbar(img, ax=mt_ax)
        if (j+1)%9 == 0:
            mt_fig.savefig(pp, format='pdf')
            mt_fig.clf()
            mt_fig = plt.figure(figsize=(16, 16))
    mt_fig.savefig(pp, format='pdf')
    mt_fig.clf()
    pp.close()


def visualize_mtrix_with_picture(attn_weights_list, primitive_num, log_dir, imgs):
    # setting
    layer_size = len(attn_weights_list[0])
    head_size = attn_weights_list[0][0].shape[1]
    assert args.select_layer < layer_size, f'max layer size {layer_size-1}, but you give me {args.select_layer}'
    assert args.select_head < head_size, f'max head size {head_size-1}, but you give me {args.select_head}'
    # extract matrix we focus on
    mt_list = []
    for attn_weight in attn_weights_list:
        lay_wt = attn_weight[args.select_layer]
        if args.select_head == -1:
            head_mt = jnp.mean(lay_wt[0], axis=0)
        else:
            head_mt = lay_wt[0][args.select_head]
        mt_list.append(head_mt)
    length = mt_list[0].shape[0]
    num_scale = [i for i in range(length)]
    scale = [i for i in range(primitive_num*2)]

    # overwrap pic
    def overwrap_picture(imgs):
        imgs = imgs[::-1]
        base_img = imgs[0].convert('RGBA')
        for i in range(1, len(imgs)):
            paste_img = imgs[i]
            red, _, _ = paste_img.split()
            red = red.point(lambda x: 255 if x > 150 else 0)

            paste_img = red.convert('RGBA')
            datas = paste_img.getdata()
            newData = []
            for item in datas:
                if item[0] == 0 and item[1] == 0 and item[2] == 0:
                    newData.append((0, 0, 0, 0))
                else:
                    newData.append((item[0], item[1], item[2], 30))
            paste_img.putdata(newData)
            base = Image.new('RGBA', base_img.size)
            base_img = Image.alpha_composite(base, base_img)
            base_img = Image.alpha_composite(base_img, paste_img)
        return base_img
    
    image_num = len(mt_list) // args.matrix_freq + 1
    fig_all, axes = plt.subplots(1, image_num, figsize=(5*image_num, 12))
    for j in range(image_num):
        fig = plt.figure(figsize=(6, 12))
        # matrix output
        mt_part = mt_list[args.matrix_freq*j:args.matrix_freq*(j+1)]
        if len(mt_part) == 0:
            break
        denominator_mt = np.zeros(mt_part[0].shape)
        for m in mt_part:
            m = np.where(m > 0, 1, 0)
            denominator_mt += m
        denominator_mt = np.where(denominator_mt == 0, 1, denominator_mt)
        mt = sum(mt_part) / denominator_mt
        mt_ax = fig.add_subplot(2, 1, 1)
        mt_ax.set_xticks(num_scale)
        mt_ax.set_xticklabels(scale, rotation=90)
        mt_ax.set_yticks(num_scale)
        mt_ax.set_yticklabels(scale)
        matrix = mt_ax.imshow(mt, cmap="Blues", interpolation='nearest')
        axis = inset_axes(
            mt_ax,
            width='80%',
            height='5%',
            loc='lower center',
            borderpad=3
        )
        fig.colorbar(matrix, cax=axis, orientation='horizontal')
        # img output
        img_ax = fig.add_subplot(2, 1, 2)
        imgs_part = imgs[args.matrix_freq*j:args.matrix_freq*(j+1)]
        img = overwrap_picture(imgs_part)
        img_ax.imshow(img)
        img_ax.axis('off')
        # combine
        fig.subplots_adjust(wspace=0, hspace=0, left=0.045, right=0.955, bottom=0.01, top=0.99)
        temp_path = f'{log_dir}/temp_{args.env_name}_{args.select_head}_layer{args.select_layer}.png'
        fig.savefig(temp_path)
        each_img = Image.open(temp_path)
        axes[j].imshow(each_img)
        if len(mt_list) > args.matrix_freq*(j+1):
            axes[j].set_title(f'Matrix \n {args.matrix_freq*j} - {args.matrix_freq*(j+1)-1}')
        else:
            axes[j].set_title(f'Matrix \n {args.matrix_freq*j} - {len(mt_list)-1}')
        axes[j].axis('off')
    output_path = os.path.join(log_dir, f'Matrix_and_picture_{args.env_name}_{args.select_head}_layer{args.select_layer}.png')
    fig_all.subplots_adjust(wspace=0, hspace=0, left=0.02, right=0.98)
    fig_all.savefig(output_path)

def visualize_mtrix_pick(attn_weights_list, primitive_num, log_dir, imgs):
    # setting
    layer_size = len(attn_weights_list[0])
    head_size = attn_weights_list[0][0].shape[1]
    assert args.select_layer < layer_size, f'max layer size {layer_size-1}, but you give me {args.select_layer}'
    assert args.select_head < head_size, f'max head size {head_size-1}, but you give me {args.select_head}'
    # extract matrix we focus on
    mt_list = []
    for attn_weight in attn_weights_list:
        lay_wt = attn_weight[args.select_layer]
        if args.select_head == -1:
            head_mt = jnp.mean(lay_wt[0], axis=0)
        else:
            head_mt = lay_wt[0][args.select_head]
        mt_list.append(head_mt)
    length = mt_list[0].shape[0]
    num_scale = [i for i in range(length)]
    scale = [i for i in range(primitive_num*2)]
    
    image_num = len(mt_list) // args.matrix_freq + 2
    fig_all, axes = plt.subplots(1, image_num, figsize=(5*image_num, 12))
    for j in range(image_num):
        fig = plt.figure(figsize=(6, 12))
        # matrix output
        if j == 0:
            mt = mt_list[0]
        elif len(mt_list) > args.matrix_freq*j:
            mt = mt_list[args.matrix_freq*j-1]
        else:
            mt = mt_list[len(mt_list)-1]
        mt_ax = fig.add_subplot(2, 1, 1)
        mt_ax.set_xticks(num_scale)
        mt_ax.set_xticklabels(scale, rotation=90)
        mt_ax.set_yticks(num_scale)
        mt_ax.set_yticklabels(scale)
        matrix = mt_ax.imshow(mt, cmap="Blues", interpolation='nearest')
        axis = inset_axes(
            mt_ax,
            width='80%',
            height='5%',
            loc='lower center',
            borderpad=3
        )
        fig.colorbar(matrix, cax=axis, orientation='horizontal')
        # img output
        img_ax = fig.add_subplot(2, 1, 2)
        if j == 0:
            img = imgs[0]
        elif len(mt_list) > args.matrix_freq*j:
            img = imgs[args.matrix_freq*j-1]
        else:
            img = imgs[len(mt_list)-1]
        img_ax.imshow(img)
        img_ax.axis('off')
        # combine
        fig.subplots_adjust(wspace=0, hspace=0, left=0.045, right=0.955, bottom=0.01, top=0.99)
        temp_path = f'{log_dir}/temp_{args.env_name}_{args.select_head}_layer{args.select_layer}.png'
        fig.savefig(temp_path)
        each_img = Image.open(temp_path)
        axes[j].imshow(each_img)
        if j == 0:
            axes[j].set_title(f'STEP \n 1')
        elif len(mt_list) > args.matrix_freq*j:
            axes[j].set_title(f'STEP \n {args.matrix_freq*j}')
        else:
            axes[j].set_title(f'STEP \n {len(mt_list)}')
        axes[j].axis('off')
    output_path = os.path.join(log_dir, f'Matrix_and_picture_{args.env_name}_{args.select_head}_layer{args.select_layer}.png')
    fig_all.subplots_adjust(wspace=0, hspace=0, left=0.02, right=0.98)
    fig_all.savefig(output_path)


def main(args):
    set_random_seed(args.seed)
    env = make(args.env_name, nn=(args.algo=='nn'), sdf_loss=args.sdf_loss,
                            density_loss=args.density_loss, contact_loss=args.contact_loss,
                            soft_contact_loss=args.soft_contact_loss)
    env.seed(args.seed)
    env._max_episode_steps = STEPS
    env_primitive_list = MASK_CONFIG[args.task_name]

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

    # load data from this file
    dataset_path = f'/root/roomba_hack/pbm/bc/jax_decision_transformer/data/bc_data_all.pkl'

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
    nearest_primitive_pos = 3
    state_dim = args.base_data_size*3*2+nearest_primitive_pos+args.base_data_size*3
    act_dim = 2
    mask_dim = primitive_num * 2
    trans_dim = state_dim + act_dim + mask_dim

    with open('/root/roomba_hack/pbm/bc/jax_decision_transformer/data/all_data_random_delay_0_10.pickle', 'rb') as f: 
        replay_buffer_data = pickle.load(f)
    
    # used for input normalization
    all_trans_data = jnp.concatenate(replay_buffer_data, axis=0)[:, :, :state_dim]
    all_trans_data = all_trans_data[:, env_primitive_list]
    state_mean, state_std = jnp.mean(all_trans_data, axis=0), jnp.std(all_trans_data, axis=0) + 1e-8

    
    policy_model, value_model = make_transformers(
        policy_params_size=act_dim,
        obs_size=state_dim,
        action_size=2,
        max_num_limb=primitive_num,
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

    results, actions, attn_weights_list, imgs = evaluate_on_env_random_delay(policy_model,
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
                                                                             giff_save_iters=1,
                                                                             method='bc_transformer_mask_delay',
                                                                             save_gif=True)
    
    visualize_mtrix_with_picture(attn_weights_list, primitive_num, log_dir, imgs)
    visualize_mtrix_pick(attn_weights_list, primitive_num, log_dir, imgs)
    visualize_mtrix(attn_weights_list, primitive_num, log_dir)


if __name__ == '__main__':
    start_time = time.time()
    args = get_args()
    idx = args.env_name.find('-')
    args.task_name = args.env_name[:idx]
    args.task_version = args.env_name[(idx+1):]
    main(args)
    with open(f'{args.path}/time.txt', 'w') as f:
        f.write(str(time.time() - start_time))
