import argparse
import csv
import os
import pickle
import random
import sys
from datetime import datetime
from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
import json
import numpy as np
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from decision_transformer.bc_mlp.model import make_policy_networks
from decision_transformer.bc_mlp.utils import (NormalTanhDistribution,
                                               ReplayBuffer, TrainingState,
                                               Transition, evaluate_on_env,
                                               get_d4rl_normalized_score,
                                               save_params)
from decision_transformer.pmap import (bcast_local_devices, is_replicated,
                                       synchronize_hosts)

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
from plb.algorithms.discor.run_sac import train as train_sac
from plb.algorithms.logger import Logger
from plb.algorithms.ppo.run_ppo import train_ppo
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
    hidden_size = args.hidden_size          # hidden size
    num_layers = args.num_layers            # num of layers for MLP policy

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

    prefix = "bc_mlp" + args.task_name

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
    state_dim = (args.base_data_size*3*2+primitive_num*3)*primitive_num+args.base_data_size*3
    act_dim = int(primitive_num*2)
    trans_dim = state_dim + act_dim

    # used for input normalization
    state_stats = jnp.concatenate([traj['observations'] for traj in trajectories], axis=0)
    state_mean, state_std = jnp.mean(state_stats, axis=0), jnp.std(state_stats, axis=0) + 1e-8

    # apply padding
    replay_buffer_data = []
    for traj in trajectories:
        states = jnp.array(traj['observations'])
    
        # apply input normalization
        if not args.rm_normalization:
            states = (states - state_mean) / state_std

        actions = jnp.array(traj['actions'])

        trans_data = jnp.concatenate([states, actions], axis=-1)
        
        assert trans_dim == trans_data.shape[-1], trans_data.shape
        replay_buffer_data.append(trans_data)

    replay_buffer = ReplayBuffer(
        data=jnp.concatenate(replay_buffer_data, axis=0).reshape(local_devices_to_use, -1, trans_dim)
    ) # (local_devices_to_use, num_steps, trans_dim)

    policy_model = make_policy_networks(
        policy_params_size=act_dim,
        state_dim=state_dim,
        hidden_layer_sizes=tuple([hidden_size for _ in range(num_layers)]),
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
        a_p = jnp.tanh(policy_model.apply(policy_params, s_t))

        actor_loss = jnp.mean(jnp.square(a_t - a_p))

        return actor_loss

    actor_grad = jax.jit(jax.value_and_grad(actor_loss))

    @jax.jit
    def update_step(
        state: TrainingState,
        transitions: jnp.ndarray,
    ) -> Tuple[TrainingState, bool, Dict[str, jnp.ndarray]]:

        transitions = Transition(
            s_t=transitions[:, :state_dim],
            a_t=transitions[:, state_dim:state_dim+act_dim],
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
  
        results, actions = evaluate_on_env(policy_model,
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
                                i_train_iter)

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
    parser.add_argument('--max_eval_ep_len', type=int, default=100)
    parser.add_argument('--num_eval_ep', type=int, default=1)
    parser.add_argument('--dataset_dir', type=str, default='data/')
    parser.add_argument('--datapath', type=str, default='bc_data_all.pkl')
    parser.add_argument('--log_dir', type=str, default='dt_runs/')
    parser.add_argument('--hidden_size', type=int, default=64)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--grad_updates_per_step', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--max_train_iters', type=int, default=1000)
    parser.add_argument('--num_updates_per_iter', type=int, default=1000)
    parser.add_argument('--policy_save_iters', type=int, default=1)
    parser.add_argument('--rm_normalization', action='store_true', help='Turn off input normalization')
    parser.add_argument("--policy_params_path", type=str, default='/')
    parser.add_argument('--max_devices_per_host', type=int, default=None)
    parser.add_argument('--base_data_size', type=int, default=102)

    args = parser.parse_args()

    idx = args.env_name.find('-')
    args.task_name = args.env_name[:idx]
    args.task_version = args.env_name[(idx+1):]
    
    train(args)
