import argparse
import csv
import json
import os
import pickle
import random
import sys
from datetime import datetime
from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import torch
from matplotlib.backends.backend_pdf import PdfPages

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from decision_transformer.bc_transformer_nearest.mask_config import MASK_CONFIG
from decision_transformer.bc_transformer_nearest.model import make_transformers
from decision_transformer.bc_transformer_nearest.utils import (
    NormalTanhDistribution, ReplayBuffer, TrainingState, Transition,
    evaluate_on_env, get_d4rl_normalized_score, save_params)
from decision_transformer.pmap import (bcast_local_devices, is_replicated,
                                       synchronize_hosts)

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
from plb.algorithms.discor.run_sac import train as train_sac
from plb.algorithms.logger import Logger
from plb.algorithms.TD3.run_td3 import train_td3
from plb.envs import make
from plb.optimizer.solver import solve_action
from plb.optimizer.solver_nn import solve_nn

STEPS=100


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def train(args):

    dataset = args.dataset          # medium / medium-replay / medium-expert
    
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
    num_eval_ep = args.num_eval_ep          # num of evaluation episodes

    batch_size = args.batch_size            # training batch size
    batch_size_per_device = batch_size // local_devices_to_use
    grad_updates_per_step = args.grad_updates_per_step

    lr = args.lr                            # learning rate

    # total updates = max_train_iters x num_updates_per_iter
    max_train_iters = args.max_train_iters
    num_updates_per_iter = args.num_updates_per_iter

    # load data from this file
    dataset_path = f'{args.dataset_dir}/{args.datapath}'

    # saves model and csv in this directory
    log_dir = args.log_dir
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    start_time = datetime.now().replace(microsecond=0)
    start_time_str = start_time.strftime("%y-%m-%d-%H-%M-%S")

    prefix = f"bc_transformer_nearest_{args.delay_time}_" + args.task_name

    log_dir = os.path.join(log_dir, prefix, f'seed_{seed}', start_time_str)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    print(f'Saving outputs to {log_dir}')
    with open(f'{log_dir}/args.txt', mode="w") as f:
        json.dump(args.__dict__, f, indent=4)

    save_model_name = "model.pt"
    save_model_path = os.path.join(log_dir, save_model_name)
    save_best_model_path = save_model_path[:-3] + "_best.pt"

    log_csv_name = "log.csv"
    log_csv_path = os.path.join(log_dir, log_csv_name)

    csv_writer = csv.writer(open(log_csv_path, 'a', 1))
    csv_header = ([
        "duration",
        "num_updates",
        "action_loss",
        "eval_avg_reward",
        "eval_avg_ep_len",
        "eval_d4rl_score"
    ])

    csv_writer.writerow(csv_header)

    print("=" * 60)
    print("start time: " + start_time_str)
    print("=" * 60)

    print("dataset path: " + dataset_path)
    print("model save path: " + save_model_path)
    print("log csv save path: " + log_csv_path)

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
    
    # apply padding
    replay_buffer_data = []
    for traj in trajectories:
        states = jnp.array(traj['observations'])

        actions = jnp.array(traj['actions'])
        
        # order data
        trans_data_list = []
        chunk_size = args.base_data_size*3*2+primitive_num*3
        for i in range(primitive_num):
            primitive_info = states[:, None, i*chunk_size:(i+1)*chunk_size]
            primitive_obs_info = primitive_info[:, :, :args.base_data_size*3*2]
            primitive_primitive_info = primitive_info[:, :, args.base_data_size*3*2:args.base_data_size*3*2+primitive_num*3]
            data = jnp.concatenate([primitive_obs_info, primitive_primitive_info, states[:, None, -args.base_data_size*3:]], axis=-1)
            trans_data_list.append(data)
        
        # delay data
        trans_data_base = jnp.concatenate(trans_data_list, axis=1)
        action_data_base = actions[:, None, :].reshape(actions.shape[0], -1, act_dim)
        if args.delay_time > 0:
            trans_padding = jnp.repeat(trans_data_base[0, :][None, :, :], args.delay_time, axis=0)
            trans_data_padding = jnp.concatenate([trans_padding, trans_data_base], axis=0)[0:-args.delay_time, :, :]
            action_padding = jnp.zeros((args.delay_time, action_data_base.shape[1], action_data_base.shape[2]))
            action_data_padding = jnp.concatenate([action_padding, action_data_base], axis=0)[0:-args.delay_time, :, :]
        elif args.delay_time == 0:
            trans_data_padding = trans_data_base.copy()
            action_data_padding = action_data_base.copy()
        else:
            raise NotImplementedError
        trans_data = jnp.concatenate([trans_data_base, trans_data_padding], axis=1)
        action_data = jnp.concatenate([action_data_base, action_data_padding], axis=1)

        # creating mask
        batch_size = trans_data.shape[0]
        primitive_num = int(trans_data.shape[1]/2)
        mask = jnp.zeros((batch_size, primitive_num*2, primitive_num*2))
        mask_myself_base = jnp.identity(primitive_num*2)
        mask_myself = mask_myself_base.at[:, primitive_num:].set(0)
        mask = jnp.where(mask_myself, 1, mask)
        big_pos = jnp.finfo(jnp.float32).max
        dist_mask_list = []
        trans_data_nearest_primitive = jnp.concatenate([trans_data[:, :, :args.base_data_size*3*2+3], trans_data[:, :, args.base_data_size*3*2+primitive_num*3:]], axis=2)
        for i in range(primitive_num*2):
            primitives_info = trans_data[:, i, args.base_data_size*3*2:args.base_data_size*3*2+primitive_num*3].reshape(batch_size, -1, 3)
            dist = primitives_info[:, :, 0]**2+primitives_info[:, :, 2]**2
            dist = dist.at[:, i%primitive_num].set(big_pos)
            dist_mask_each_primitive = (dist == jnp.min(dist, axis=1, keepdims=True)).astype(int)
            dist_index_each_primitive = jnp.argmin(dist, axis=1, keepdims=True)
            if i < primitive_num:
                dist_mask_padding = jnp.zeros((batch_size, primitive_num))
                dist_mask_each_primitive_padding = jnp.concatenate([dist_mask_padding, dist_mask_each_primitive], axis=1)
                dist_mask_list.append(dist_mask_each_primitive_padding)
            nearest_primitives_info = primitives_info[:, dist_index_each_primitive][:, 0, :, :]
            nearest_primitives_info_reshaped = nearest_primitives_info.reshape(batch_size, -1)
            trans_data_nearest_primitive = trans_data_nearest_primitive.at[:, i, args.base_data_size*3*2:args.base_data_size*3*2+3].set(nearest_primitives_info_reshaped)

        dist_mask = jnp.concatenate(dist_mask_list, axis=1).reshape(batch_size, primitive_num, -1)
        dist_mask = jnp.concatenate([dist_mask, jnp.zeros((batch_size, primitive_num, primitive_num*2))], axis=1)
        mask = jnp.where(dist_mask, 1, mask)

        trans_data = jnp.concatenate([trans_data_nearest_primitive, action_data, mask], axis=-1)
        
        assert trans_dim == trans_data.shape[-1], trans_data.shape
        replay_buffer_data.append(trans_data)
    
    # with open('/root/roomba_hack/pbm/bc/jax_decision_transformer/data/all_data_delay_0.pickle', 'rb') as f: 
    #     replay_buffer_data = pickle.load(f)

    # used for input normalization
    all_trans_data = jnp.concatenate(replay_buffer_data, axis=0)[:, :, :state_dim]

    all_act_mask_data = jnp.concatenate(replay_buffer_data, axis=0)[:, :, state_dim:]
    state_mean, state_std = jnp.mean(all_trans_data, axis=0), jnp.std(all_trans_data, axis=0) + 1e-8
    # used for input normalization
    all_trans_data = jnp.concatenate(replay_buffer_data, axis=0)[:, :, :state_dim]
    all_act_mask_data = jnp.concatenate(replay_buffer_data, axis=0)[:, :, state_dim:]
    state_mean, state_std = jnp.mean(all_trans_data, axis=0), jnp.std(all_trans_data, axis=0) + 1e-8
    # apply input normalization
    if not args.rm_normalization:
        all_trans_data = (all_trans_data - state_mean) / state_std

    replay_buffer = ReplayBuffer(
        data=jnp.concatenate([all_trans_data, all_act_mask_data], axis=2).reshape(local_devices_to_use, -1, primitive_num*2, trans_dim)
    ) # (local_devices_to_use, num_steps, trans_dim)
    
    policy_model, value_model = make_transformers(
        policy_params_size=act_dim,
        obs_size=state_dim,
        action_size=2,
        max_num_limb=primitive_num,
    )
    parametric_action_distribution = NormalTanhDistribution(event_size=act_dim)

    policy_optimizer = optax.adam(learning_rate=lr)
    policy_params = policy_model.init({'params': global_key})
    policy_optimizer_state = policy_optimizer.init(policy_params)

    # count the number of parameters
    param_count = sum(x.size for x in jax.tree_leaves(policy_params))
    print(f'num_policy_param: {param_count}')

    policy_optimizer_state, policy_params = bcast_local_devices(
        (policy_optimizer_state, policy_params), local_devices_to_use)

    def actor_loss(policy_params: Any,
                   transitions: Transition, key: jnp.ndarray) -> jnp.ndarray:
        s_t = transitions.s_t  # (batch_size_per_device, state_dim)
        a_t = transitions.a_t  # (batch_size_per_device, action_dim)
        mask = transitions.m_t # (batch_size_per_device, mask_dim)

        mask = mask[:, None, :, :]
        mask = jnp.array(mask)
        a_p, attn_weights = policy_model.apply(policy_params, s_t, mask)
        a_p = jnp.tanh(a_p[:, :primitive_num*2])
        a_t = a_t[:, :primitive_num].reshape(-1, int(primitive_num*2))
        actor_loss = jnp.mean(jnp.square(a_t - a_p))

        return actor_loss

    actor_grad = jax.jit(jax.value_and_grad(actor_loss))

    @jax.jit
    def update_step(
        state: TrainingState,
        transitions: jnp.ndarray
    ) -> Tuple[TrainingState, bool, Dict[str, jnp.ndarray]]:

        transitions = Transition(
            s_t=transitions[:, :, :state_dim],
            a_t=transitions[:, :, state_dim:state_dim+act_dim],
            m_t=transitions[:, :, state_dim+act_dim:state_dim+act_dim+mask_dim]
        )

        key, key_actor = jax.random.split(state.key, 2)

        actor_loss, actor_grads = actor_grad(state.policy_params, transitions, key_actor)
        actor_grads = jax.lax.pmean(actor_grads, axis_name='i')
        policy_params_update, policy_optimizer_state = policy_optimizer.update(
            actor_grads, state.policy_optimizer_state, state.policy_params)
        policy_params = optax.apply_updates(state.policy_params, policy_params_update)

        metrics = {'actor_loss': actor_loss}

        new_state = TrainingState(
            policy_optimizer_state=policy_optimizer_state,
            policy_params=policy_params,
            key=key,
            actor_steps=state.actor_steps + 1)
        return new_state, metrics

    def sample_data(training_state, replay_buffer):
        # num_updates_per_iter
        key1, key2 = jax.random.split(training_state.key, 2)
        idx = jax.random.randint(
            key1, (int(batch_size_per_device*grad_updates_per_step),),
            minval=0,
            maxval=replay_buffer.data.shape[0])  # from (0, num_steps)
        # (batch_size_per_device*num_updates_per_iter, trans_dim)
        transitions = jnp.take(replay_buffer.data, idx, axis=0, mode='clip')
        # (num_updates_per_iter, batch_size_per_device, context_len, trans_dim)
        transitions = jnp.reshape(transitions,
                                  [grad_updates_per_step, -1] + list(transitions.shape[1:]))

        training_state = training_state.replace(key=key1)
        return training_state, transitions

    def run_one_epoch(carry, unused_t):
        training_state, replay_buffer = carry

        training_state, transitions = sample_data(training_state, replay_buffer)
        training_state, metrics = jax.lax.scan(
            update_step, training_state, transitions, length=1)
        return (training_state, replay_buffer), metrics

    def run_training(training_state, replay_buffer):
        synchro = is_replicated(
            training_state.replace(key=jax.random.PRNGKey(0)), axis_name='i')
        (training_state, replay_buffer), metrics = jax.lax.scan(
            run_one_epoch, (training_state, replay_buffer), (),
            length=num_updates_per_iter)

        metrics = jax.tree_map(jnp.mean, metrics)
        return training_state, replay_buffer, metrics, synchro
    
    run_training = jax.pmap(run_training, axis_name='i')

    training_state = TrainingState(
        policy_optimizer_state=policy_optimizer_state,
        policy_params=policy_params,
        key=jnp.stack(jax.random.split(local_key, local_devices_to_use)),
        actor_steps=jnp.zeros((local_devices_to_use,)))
    

    max_last_iou = -1.0
    total_updates = 0

    for i_train_iter in range(max_train_iters):
        log_action_losses = []

        # optimization
        
        training_state, replay_buffer, training_metrics, synchro = run_training(
            training_state, replay_buffer)
        assert synchro[0], (current_step, training_state)
        jax.tree_map(lambda x: x.block_until_ready(), training_metrics)
        log_action_losses.append(training_metrics['actor_loss'])
  
        results, actions, attn_weights_list, imgs = evaluate_on_env(policy_model,
                                training_state.policy_params,
                                parametric_action_distribution,
                                env,
                                None,
                                num_eval_ep,
                                max_eval_ep_len,
                                state_mean,
                                state_std,
                                args,
                                log_dir,
                                i_train_iter,
                                save_gif=True)

        eval_avg_reward = results['eval/avg_reward']
        eval_avg_ep_len = results['eval/avg_ep_len']
        eval_avg_last_iou = results['eval/avg_last_iou']
        # eval_d4rl_score = get_d4rl_normalized_score(results['eval/avg_reward'], env_name) * 100

        mean_action_loss = np.mean(log_action_losses)
        time_elapsed = str(datetime.now().replace(microsecond=0) - start_time)

        total_updates += num_updates_per_iter

        log_str = ("=" * 60 + '\n' +
                   "time elapsed: " + time_elapsed  + '\n' +
                   "train iter: " + str(i_train_iter)  + '\n' +
                   "num of updates: " + str(total_updates) + '\n' +
                   "action loss: " +  format(mean_action_loss, ".5f") + '\n' +
                   "eval avg reward: " + format(eval_avg_reward, ".5f") + '\n' +
                   "eval avg ep len: " + format(eval_avg_ep_len, ".5f") + '\n' +
                   "eval avg last iou: " + format(eval_avg_last_iou, ".5f") + '\n'
                #    "eval d4rl score: " + format(eval_d4rl_score, ".5f")
                )

        print(log_str)

        log_data = [
            time_elapsed,
            total_updates,
            mean_action_loss,
            eval_avg_reward,
            eval_avg_ep_len,
            eval_avg_last_iou
            # eval_d4rl_score
        ]

        csv_writer.writerow(log_data)

        # save model
        _policy_params = jax.tree_map(lambda x: x[0], training_state.policy_params)
        print("max last iou: " + format(max_last_iou, ".5f"))
        if eval_avg_last_iou >= max_last_iou:
            print("saving max d4rl score model at: " + save_best_model_path)
            save_params(save_best_model_path, _policy_params)
            max_last_iou = eval_avg_last_iou

        if i_train_iter % args.policy_save_iters == 0 or i_train_iter == max_train_iters - 1:
            save_current_model_path = save_model_path[:-3] + f"_{total_updates}.pt"
            print("saving current model at: " + save_current_model_path)
            save_params(save_current_model_path, _policy_params)

    synchronize_hosts()

    print("=" * 60)
    print("finished training!")
    print("=" * 60)
    end_time = datetime.now().replace(microsecond=0)
    time_elapsed = str(end_time - start_time)
    end_time_str = end_time.strftime("%y-%m-%d-%H-%M-%S")
    print("started training at: " + start_time_str)
    print("finished training at: " + end_time_str)
    print("total training time: " + time_elapsed)
    print("max last iou: " + format(max_last_iou, ".5f"))
    print("saved max last iou model at: " + save_best_model_path)
    print("saved last updated model at: " + save_model_path)
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument("--algo", type=str, default='action')
    parser.add_argument("--env_name", type=str, default="multi_bc_rope-v1")
    parser.add_argument("--task_name", type=str, default="multi_bc_rope")
    parser.add_argument("--task_version", type=str, default="v1")
    parser.add_argument("--sdf_loss", type=float, default=500)
    parser.add_argument("--density_loss", type=float, default=500)
    parser.add_argument("--contact_loss", type=float, default=1)
    parser.add_argument("--soft_contact_loss", action='store_true')

    parser.add_argument('--dataset', type=str, default='medium')
    parser.add_argument('--max_eval_ep_len', type=int, default=115)
    parser.add_argument('--num_eval_ep', type=int, default=1)
    parser.add_argument('--dataset_dir', type=str, default='data/')
    parser.add_argument('--datapath', type=str, default='bc_data_all.pkl')
    parser.add_argument('--log_dir', type=str, default='dt_runs/')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--grad_updates_per_step', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--max_train_iters', type=int, default=1000)
    parser.add_argument('--num_updates_per_iter', type=int, default=1000)
    parser.add_argument('--policy_save_iters', type=int, default=1)
    parser.add_argument('--rm_normalization', action='store_true', help='Turn off input normalization')
    parser.add_argument("--policy_params_path", type=str, default='/')
    parser.add_argument('--max_devices_per_host', type=int, default=None)
    parser.add_argument('--base_data_size', type=int, default=102)
    parser.add_argument('--delay_time', type=int, default=0)

    args = parser.parse_args()

    idx = args.env_name.find('-')
    args.task_name = args.env_name[:idx]
    args.task_version = args.env_name[(idx+1):]
    
    train(args)
