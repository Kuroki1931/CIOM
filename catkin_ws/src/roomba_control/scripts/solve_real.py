#!/usr/bin/env python3
import numpy as np
import datetime
import cv2
import itertools

import rospy
import tf
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry

import argparse
import csv
import flax
import functools
import gym
import jax
import optax
import os
import pickle
import random
import sys
import pickle

import jax.numpy as jnp
import numpy as np

from datetime import datetime
from typing import Any, Dict, Tuple
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.pyplot as plt
from PIL import Image

sys.path.append(os.path.join(os.path.dirname(__file__), '/root/roomba_hack/pbm/bc/network'))

from transformer.bc_transformer_nearest.model import make_transformers
from transformer.bc_transformer_nearest.utils import ReplayBuffer, TrainingState, Transition
from transformer.bc_transformer_nearest.utils import save_params, load_params
from transformer.bc_transformer_nearest.mask_config import MASK_CONFIG
from transformer.pmap import bcast_local_devices, synchronize_hosts, is_replicated


import json
import time
import argparse
import random
import numpy as np
import os
import torch
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '/root/roomba_hack/pbm'))

from datetime import datetime

from plb.envs import make
from plb.algorithms.logger import Logger
from plb.algorithms.discor.run_sac import train as train_sac
from plb.algorithms.ppo.run_ppo import train_ppo
from plb.algorithms.TD3.run_td3 import train_td3
from plb.optimizer.solver import solve_action, solve_mppi, optimize_mppi
from plb.optimizer.solver_nn import solve_nn
from plb.algorithms.solve_bc_transformer_nearest import  visualize_mtrix_pick, visualize_mtrix


os.environ['TI_USE_UNIFIED_MEMORY'] = '0'
os.environ['TI_DEVICE_MEMORY_FRACTION'] = '0.9'
os.environ['TI_DEVICE_MEMORY_GB'] = '4'
os.environ['TI_ENABLE_CUDA'] = '0'
os.environ['TI_ENABLE_OPENGL'] = '0'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

RL_ALGOS = ['sac', 'td3', 'ppo']
DIFF_ALGOS = ['action', 'nn']

STEPS=100


class MultiRoombaController:
    def __init__(self):
        rospy.init_node('multi_roomba_controller', anonymous=True)
        
        # Publisher
        self.cmd_vel_pub1 = rospy.Publisher('/roomba1/cmd_vel', Twist, queue_size=10)
        self.cmd_vel_pub2 = rospy.Publisher('/roomba2/cmd_vel', Twist, queue_size=10)
        self.cmd_vel_pub3 = rospy.Publisher('/roomba3/cmd_vel', Twist, queue_size=10)
        self.cmd_vel_pub4 = rospy.Publisher('/roomba4/cmd_vel', Twist, queue_size=10)
        self.cmd_vel_pub5 = rospy.Publisher('/roomba5/cmd_vel', Twist, queue_size=10)

        # Subscriber
        odom_sub1 = rospy.Subscriber('/roomba1/odom', Odometry, self.callback_odom1)
        odom_sub2 = rospy.Subscriber('/roomba2/odom', Odometry, self.callback_odom2)
        odom_sub3 = rospy.Subscriber('/roomba3/odom', Odometry, self.callback_odom3)
        odom_sub4 = rospy.Subscriber('/roomba4/odom', Odometry, self.callback_odom4)
        odom_sub5 = rospy.Subscriber('/roomba5/odom', Odometry, self.callback_odom5)

        self.x1= None
        self.y1 = None
        self.yaw1 = None
        while self.x1 is None:
            rospy.sleep(0.1)
        self.x2 = None
        self.y2 = None
        self.ya2 = None
        while self.x2 is None:
            rospy.sleep(0.1)
        self.x3 = None
        self.y3 = None
        self.yaw3 = None
        while self.x3 is None:
            rospy.sleep(0.1)
        self.x4 = None
        self.y4 = None
        self.yaw4 = None
        while self.x4 is None:
            rospy.sleep(0.1)
        self.x5 = None
        self.y5 = None
        self.yaw5 = None
        while self.x5 is None:
            rospy.sleep(0.1)

    def callback_odom1(self, data):
        self.x1 = data.pose.pose.position.x
        self.y1 = data.pose.pose.position.y
        self.yaw1 = self.get_yaw_from_quaternion(data.pose.pose.orientation)
    
    def callback_odom2(self, data):
        self.x2 = data.pose.pose.position.x
        self.y2 = data.pose.pose.position.y
        self.yaw2 = self.get_yaw_from_quaternion(data.pose.pose.orientation)
    
    def callback_odom3(self, data):
        self.x3 = data.pose.pose.position.x
        self.y3 = data.pose.pose.position.y
        self.yaw3 = self.get_yaw_from_quaternion(data.pose.pose.orientation)
    
    def callback_odom4(self, data):
        self.x4 = data.pose.pose.position.x
        self.y4 = data.pose.pose.position.y
        self.yaw4 = self.get_yaw_from_quaternion(data.pose.pose.orientation)
    
    def callback_odom5(self, data):
        self.x5 = data.pose.pose.position.x
        self.y5 = data.pose.pose.position.y
        self.yaw5 = self.get_yaw_from_quaternion(data.pose.pose.orientation)

    def time_control(self, velocity1, velocity2, velocity3, velocity4, velocity5, yawrate, time):
        vel1 = Twist()
        vel2 = Twist()
        vel3 = Twist()
        vel4 = Twist()
        vel5 = Twist()
        start_time = rospy.get_rostime().secs
        while(rospy.get_rostime().secs-start_time<time):
            print('move')
            vel1.linear.x = velocity1
            vel1.angular.z = yawrate
            vel2.linear.x = velocity2
            vel2.angular.z = yawrate
            vel3.linear.x = velocity3
            vel3.angular.z = yawrate
            vel4.linear.x = velocity4
            vel4.angular.z = yawrate
            vel5.linear.x = velocity5
            vel5.angular.z = yawrate
            self.cmd_vel_pub1.publish(vel1)
            self.cmd_vel_pub2.publish(vel2)
            self.cmd_vel_pub3.publish(vel3)
            self.cmd_vel_pub4.publish(vel4)
            self.cmd_vel_pub5.publish(vel5)
            rospy.sleep(0.1)
        return

    def turn(self, yaw1, yaw2, yaw3, yaw4, yaw5, yawrate1, yawrate2, yawrate3, yawrate4, yawrate5):
        vel1 = Twist()
        vel2 = Twist()
        vel3 = Twist()
        vel4 = Twist()
        vel5 = Twist()
        yaw_initial1 = self.yaw1
        yaw_initial2 = self.yaw2
        yaw_initial3 = self.yaw3
        yaw_initial4 = self.yaw4
        yaw_initial5 = self.yaw5
        done1 = False
        done2 = False
        done3 = False
        done4 = False
        done5 = False
        while not (done1 & done2 & done3 & done4 & done5):
            print('rotate')
            vel1.linear.x = 0.0
            vel1.angular.z = yawrate1
            vel2.linear.x = 0.0
            vel2.angular.z = yawrate2
            vel3.linear.x = 0.0
            vel3.angular.z = yawrate3
            vel4.linear.x = 0.0
            vel4.angular.z = yawrate4
            vel5.linear.x = 0.0
            vel5.angular.z = yawrate5
            self.cmd_vel_pub1.publish(vel1)
            self.cmd_vel_pub2.publish(vel2)
            self.cmd_vel_pub3.publish(vel3)
            self.cmd_vel_pub4.publish(vel4)
            self.cmd_vel_pub5.publish(vel5)
            if not (abs(self.yaw1-yaw_initial1)<np.deg2rad(yaw1)):
                done1 = True
                yawrate1 = 0
            if not (abs(self.yaw2-yaw_initial2)<np.deg2rad(yaw2)):
                done2 = True
                yawrate2 = 0
            if not (abs(self.yaw3-yaw_initial3)<np.deg2rad(yaw3)):
                done3 = True
                yawrate3 = 0
            if not (abs(self.yaw4-yaw_initial4)<np.deg2rad(yaw4)):
                done4 = True
                yawrate4 = 0
            if not (abs(self.yaw5-yaw_initial5)<np.deg2rad(yaw5)):
                done5 = True
                yawrate5 = 0
            rospy.sleep(0.1)
        return

    def get_yaw_from_quaternion(self, quaternion):
        e = tf.transformations.euler_from_quaternion(
                (quaternion.x, quaternion.y, quaternion.z, quaternion.w))
        return e[2]


class ThreeRoombaController:
    def __init__(self):
        rospy.init_node('multi_roomba_controller', anonymous=True)
        
        # Publisher
        self.cmd_vel_pub1 = rospy.Publisher('/roomba1/cmd_vel', Twist, queue_size=10)
        self.cmd_vel_pub2 = rospy.Publisher('/roomba2/cmd_vel', Twist, queue_size=10)
        self.cmd_vel_pub3 = rospy.Publisher('/roomba3/cmd_vel', Twist, queue_size=10)

        # Subscriber
        odom_sub1 = rospy.Subscriber('/roomba1/odom', Odometry, self.callback_odom1)
        odom_sub2 = rospy.Subscriber('/roomba2/odom', Odometry, self.callback_odom2)
        odom_sub3 = rospy.Subscriber('/roomba3/odom', Odometry, self.callback_odom3)

        self.x1= None
        self.y1 = None
        self.yaw1 = None
        while self.x1 is None:
            rospy.sleep(0.1)
        self.x2 = None
        self.y2 = None
        self.ya2 = None
        while self.x2 is None:
            rospy.sleep(0.1)
        self.x3 = None
        self.y3 = None
        self.yaw3 = None
        while self.x3 is None:
            rospy.sleep(0.1)

    def callback_odom1(self, data):
        self.x1 = data.pose.pose.position.x
        self.y1 = data.pose.pose.position.y
        self.yaw1 = self.get_yaw_from_quaternion(data.pose.pose.orientation)
    
    def callback_odom2(self, data):
        self.x2 = data.pose.pose.position.x
        self.y2 = data.pose.pose.position.y
        self.yaw2 = self.get_yaw_from_quaternion(data.pose.pose.orientation)
    
    def callback_odom3(self, data):
        self.x3 = data.pose.pose.position.x
        self.y3 = data.pose.pose.position.y
        self.yaw3 = self.get_yaw_from_quaternion(data.pose.pose.orientation)

    def time_control(self, velocity1, velocity2, velocity3, yawrate, time):
        vel1 = Twist()
        vel2 = Twist()
        vel3 = Twist()
        start_time = rospy.get_rostime().secs
        while(rospy.get_rostime().secs-start_time<time):
            print('move')
            vel1.linear.x = velocity1
            vel1.angular.z = yawrate
            vel2.linear.x = velocity2
            vel2.angular.z = yawrate
            vel3.linear.x = velocity3
            vel3.angular.z = yawrate
            self.cmd_vel_pub1.publish(vel1)
            self.cmd_vel_pub2.publish(vel2)
            self.cmd_vel_pub3.publish(vel3)
            rospy.sleep(0.1)
        return

    def turn(self, yaw1, yaw2, yaw3, yawrate1, yawrate2, yawrate3):
        vel1 = Twist()
        vel2 = Twist()
        vel3 = Twist()
        yaw_initial1 = self.yaw1
        yaw_initial2 = self.yaw2
        yaw_initial3 = self.yaw3
        done1 = False
        done2 = False
        done3 = False
        while not (done1 & done2 & done3):
            print('rotate')
            vel1.linear.x = 0.0
            vel1.angular.z = yawrate1
            vel2.linear.x = 0.0
            vel2.angular.z = yawrate2
            vel3.linear.x = 0.0
            vel3.angular.z = yawrate3
            self.cmd_vel_pub1.publish(vel1)
            self.cmd_vel_pub2.publish(vel2)
            self.cmd_vel_pub3.publish(vel3)
            if not (abs(self.yaw1-yaw_initial1)<np.deg2rad(yaw1)):
                done1 = True
                yawrate1 = 0
            if not (abs(self.yaw2-yaw_initial2)<np.deg2rad(yaw2)):
                done2 = True
                yawrate2 = 0
            if not (abs(self.yaw3-yaw_initial3)<np.deg2rad(yaw3)):
                done3 = True
                yawrate3 = 0
            rospy.sleep(0.1)
        return

    def get_yaw_from_quaternion(self, quaternion):
        e = tf.transformations.euler_from_quaternion(
                (quaternion.x, quaternion.y, quaternion.z, quaternion.w))
        return e[2]


def organaize_state(rope_xzy, primitives_xzy, env, args, primitive_num, state_mean, state_std, delay_data_base, delay_roomba_pos=None, t=0):
    ## change observation
    goal_state = env.get_goal_obs(args)
    seg_size = goal_state.shape[0]
    step_size = int((seg_size/3) // 100)
    goal_state = goal_state.reshape(-1, 3)[::step_size, :]

    if t:
        delay_roomba_pos = np.concatenate([delay_roomba_pos, primitives_xzy[None, :]], axis=0)
        primitives_xzy_5_delay = delay_roomba_pos[t, :]
    else:
        if args.delay_time > 0:
            delay_roomba_pos = np.repeat(primitives_xzy[None, :], args.delay_time, axis=0)
            delay_roomba_pos = np.concatenate([delay_roomba_pos, primitives_xzy[None, :]], axis=0)
        elif args.delay_time == 0:
            delay_roomba_pos = primitives_xzy[None, :].copy()
        else:
            raise NotImplementedError
        primitives_xzy_5_delay = delay_roomba_pos[0, :]

    obs_list = []
    # primitive to rope, goal, primitive
    for i in range(int(primitive_num)):
        primitive_xzy = primitives_xzy[i*3:(i+1)*3]
        primitive_to_rope = (rope_xzy - primitive_xzy).reshape(-1)
        primitive_to_goal = (goal_state - primitive_xzy).reshape(-1)
        primitive_to_primitive = (primitives_xzy_5_delay - np.tile(primitive_xzy,(primitive_num)))
        primitive_to_primitive[i*3:(i+1)*3] = 0
        obs_list.append(primitive_to_rope)
        obs_list.append(primitive_to_goal)
        obs_list.append(primitive_to_primitive)

    # rope to goal
    rope_to_goal = (goal_state - rope_xzy).reshape(-1)
    obs_list.append(rope_to_goal)
    state = np.concatenate(obs_list)
    
    # order obs, create mask
    trans_data_list = []
    batch_size = 1
    chunk_size = args.base_data_size*3*2+primitive_num*3
    mask = jnp.zeros((1, primitive_num*2, primitive_num*2))
    mask_myself_base = jnp.identity(primitive_num*2)
    mask_myself = mask_myself_base.at[:, primitive_num:].set(0)
    mask = jnp.where(mask_myself, 1, mask)
    big_pos = jnp.finfo(jnp.float32).max
    dist_mask_list = []
    for i in range(primitive_num):
        primitive_info = state[None, None, i*chunk_size:(i+1)*chunk_size]
        primitive_obs_info = primitive_info[:, :, :args.base_data_size*3*2]
        primitive_primitive_info = primitive_info[:, :, args.base_data_size*3*2:args.base_data_size*3*2+primitive_num*3].reshape(primitive_info.shape[0], -1, 3)
        
        dist = primitive_primitive_info[:, :, 0]**2+primitive_primitive_info[:, :, 2]**2
        dist[0, i] = big_pos
        dist_mask_each_primitive = (dist == jnp.min(dist, axis=1, keepdims=True)).astype(int)
        dist_index_each_primitive = jnp.argmin(dist, axis=1, keepdims=True)

        dist_mask_padding = jnp.zeros((1, primitive_num))
        dist_mask_each_primitive_padding = jnp.concatenate([dist_mask_padding, dist_mask_each_primitive], axis=1)
        dist_mask_list.append(dist_mask_each_primitive_padding)

        nearest_primitive_info = primitive_primitive_info[:, dist_index_each_primitive][0]
        data = jnp.concatenate([primitive_obs_info, nearest_primitive_info, state[None, None, -args.base_data_size*3:]], axis=-1)
        trans_data_list.append(data)

    dist_mask = jnp.concatenate(dist_mask_list, axis=1).reshape(batch_size, primitive_num, -1)
    dist_mask = jnp.concatenate([dist_mask, jnp.zeros((batch_size, primitive_num, primitive_num*2))], axis=1)
    mask = jnp.where(dist_mask, 1, mask)[None, :, :, :]

    state_base = jnp.concatenate(trans_data_list, axis=1)
    if args.delay_time > 0:
            delay_data_base = jnp.concatenate([delay_data_base, state_base], axis=0)
    elif args.delay_time == 0:
        delay_data_base = state_base.copy()
    else:
        raise NotImplementedError
    state = jnp.concatenate([state_base, delay_data_base[t, :, :][None, :, :]], axis=1)
    
    # normalize
    state = (state - state_mean) / state_std

    return state, delay_data_base, delay_roomba_pos, mask

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
    parser.add_argument("--mppi_horizon", type=int, default=0)
    parser.add_argument("--parallel_index", type=int, default=None)
    parser.add_argument("--each_parallel_steps", type=int, default=None)
    parser.add_argument("--mppi_index", type=int, default=None)
    parser.add_argument("--mppi_steps", type=int, default=None)
    parser.add_argument("--mppi_output", action='store_true')
    parser.add_argument("--num_steps", type=int, default=None)
    base_path = '/home/robot_dev4/kuroki/DifftaichiSim2Real/pbm/bc/network/dt_runs/bc_transformer_nearest_delay_0_multi_bc_rope_6/seed_0/22-10-11-13-09-42'

    # policy
    parser.add_argument("--dataset", type=str, default='medium')
    parser.add_argument("--max_eval_ep_len", type=int, default=120)
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
    parser.add_argument("--policy_params_path", type=str, default='/root/roomba_hack/pbm/bc/network/dt_runs/bc_transformer_nearest_delay_0_multi_bc_rope_6/seed_0/22-10-11-13-09-42/model_3000.pt')
    parser.add_argument("--max_devices_per_host", type=int, default=None)
    parser.add_argument('--base_data_size', type=int, default=102)
    parser.add_argument('--select_layer', type=int, default=2)
    parser.add_argument('--select_head', type=int, default=0)
    parser.add_argument('--matrix_freq', type=int, default=20)
    parser.add_argument('--delay_time', type=int, default=0)

    # differentiable physics parameters
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--softness", type=float, default=6666.)
    parser.add_argument("--optim", type=str, default='Adam', choices=['Adam', 'Momentum'])

    args=parser.parse_args()

    return args

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

if __name__=='__main__':
    args = get_args()
    idx = args.env_name.find('-')
    args.task_name = args.env_name[:idx]
    args.task_version = args.env_name[(idx+1):]
    
    if args.task_name in ['multi_bc_rope_5_center']:
        multi_roomba_controller = MultiRoombaController()
    elif args.task_name in ['multi_bc_rope_4_center']:
        multi_roomba_controller = TwoRoombaController()
    elif args.task_name in ['multi_bc_rope_3']:
        print('set three_roomba_controller')
        multi_roomba_controller = ThreeRoombaController()
    
    print('set env')
    set_random_seed(args.seed)
    env = make(args.env_name, nn=(args.algo=='nn'), sdf_loss=args.sdf_loss,
                            density_loss=args.density_loss, contact_loss=args.contact_loss,
                            soft_contact_loss=args.soft_contact_loss)
    env.seed(args.seed)
    init_iou = env.taichi_env.loss.init_iou
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

    max_eval_ep_len = args.max_eval_ep_len  # max len of one episode

    lr = args.lr                            # learning rate

    # load data from this file
    dataset_path = f'/root/roomba_hack/pbm/bc/network/data/bc_data_all.pkl'

    start_time = datetime.now().replace(microsecond=0)
    start_time_str = start_time.strftime("%y-%m-%d-%H-%M-%S")

    # saves model and csv in this directory
    log_dir = args.policy_params_path[:-3] + '_' + args.task_version + '_' + start_time_str
    os.makedirs(log_dir, exist_ok=True)

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
    
    with open('/root/roomba_hack/pbm/bc/network/data/all_data_delay_0.pickle', 'rb') as f: 
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

    policy_optimizer = optax.adam(learning_rate=lr)
    policy_params = load_params(args.policy_params_path)
    policy_optimizer_state = policy_optimizer.init(policy_params)

    policy_optimizer_state, policy_params = bcast_local_devices(
        (policy_optimizer_state, policy_params), local_devices_to_use)
    
    policy_params = jax.tree_map(lambda x: x[0], policy_params)
    policy_model_apply = jax.jit(policy_model.apply)

    primitive_num = int(env.action_space.shape[0]/3)
    nearest_primitive_pos = 3
    state_dim = args.base_data_size*3*2+nearest_primitive_pos+args.base_data_size*3

    state_mean = jnp.array(state_mean)
    state_std = jnp.array(state_std)

    eval_ep_len = 0
    attn_weights_list = []
    while eval_ep_len * 5 < args.max_eval_ep_len:
        print('timestep', eval_ep_len)
        eval_ep_len += 1
        
        from tasks.multi_bc_rope.multi_bc_rope_real2sim import multi_bc_rope_real2sim
        roomba_info = multi_bc_rope_real2sim(real_demo=True)
        roomba_vec = roomba_info.get_roomba_first_vec()
        particles = roomba_info.get_obj_particle()
        step_size = len(particles) // 100
        rope_xzy = particles[::step_size]
        roomba_position = roomba_info.get_roomba_init()
        roomba_position = [i[1:-1].split(', ') for i in roomba_position]
        roomba_position = list(itertools.chain.from_iterable(roomba_position))
        primitives_xzy = np.array([float(i) for i in roomba_position])

        ## change observation
        delay_data_base = jnp.zeros((args.delay_time, primitive_num, state_dim))
        state, delay_data_base, delay_roomba_pos, mask = organaize_state(rope_xzy, primitives_xzy, env, args, primitive_num, state_mean, state_std, delay_data_base)

        action, attn_weights = policy_model.apply(policy_params, state, mask)
        attn_weights_list.append(attn_weights)
        action = jnp.tanh(action[:, :primitive_num*2]).reshape(-1)
        for j in range(int(len(action)/2)):
            action = np.insert(action, j*3+1, 0)

        def excute_command(seq, first_vec):
            command_list = []
            bf_vec = first_vec
            vec = np.delete(seq, 1) / 15
            v = (vec[0]**2 + vec[1]**2)**(1/2)
            product =np.dot(vec,bf_vec)
            normu = np.linalg.norm(vec)
            normv = np.linalg.norm(bf_vec)
            cost = product /(normu * normv)
            angle = np.rad2deg(np.arccos(cost))
            cross = np.cross(vec, bf_vec)
            bf_vec = vec
            if cross > 0:
                yawrate = 0.7
            else:
                yawrate = -0.7
            if angle < 5:
                yawrate = 0
                angle = 0
            command_list.append([v, yawrate, angle])
            return command_list

        excute_command_list = []
        for i in range(int(action.shape[0]/3)):
            seq = action[i*3:(i+1)*3]
            command_list = excute_command(seq, roomba_vec[i])
            excute_command_list.append(command_list)
        
        
        for r1, r2, r3 in zip(excute_command_list[0], excute_command_list[1], excute_command_list[2]):
            v1, yawrate1, angle1 = r1
            v2, yawrate2, angle2 = r2
            v3, yawrate3, angle3 = r3
            multi_roomba_controller.turn(angle1, angle2, angle3, yawrate1, yawrate2, yawrate3)
            multi_roomba_controller.time_control(v1, v2, v3, 0, 0.0000001)

    set_random_seed(args.seed)
    env = make(args.env_name, nn=(args.algo=='nn'), sdf_loss=args.sdf_loss,
                            density_loss=args.density_loss, contact_loss=args.contact_loss,
                            soft_contact_loss=args.soft_contact_loss)
    env.seed(args.seed)
    state = env.reset()
    env.taichi_env.loss.init_iou = init_iou
    loss_info = env.compute_loss()
    last_iou = loss_info['incremental_iou']
    img = env.render(mode='rgb_array')
    cv2.imwrite(f"{log_dir}/last_sim_state_{last_iou}.png", img[..., ::-1])
    with open(f'{log_dir}/iou_{args.env_name}_{last_iou}_{eval_ep_len}_last.txt', 'w') as f:
        f.write(str(last_iou))

    visualize_mtrix(attn_weights_list, primitive_num, log_dir, args)