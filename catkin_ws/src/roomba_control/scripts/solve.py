#!/usr/bin/env python3
import argparse
import datetime
import os
import pickle
import random
import sys
from datetime import datetime

import cv2
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import rospy
import tf
import torch
from geometry_msgs.msg import Twist
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from nav_msgs.msg import Odometry
from PIL import Image

sys.path.append(os.path.join(os.path.dirname(__file__), '/root/roomba_hack/pbm/bc/jax_decision_transformer'))
from decision_transformer.bc_transformer_nearest.mask_config import MASK_CONFIG
from decision_transformer.bc_transformer_nearest.model import make_transformers
from decision_transformer.bc_transformer_nearest.utils import (
    NormalTanhDistribution, ReplayBuffer, TrainingState, Transition,
    evaluate_on_env, get_d4rl_normalized_score, load_params, save_params)
from decision_transformer.pmap import (bcast_local_devices, is_replicated,
                                       synchronize_hosts)

sys.path.append(os.path.join(os.path.dirname(__file__), '/root/roomba_hack/pbm'))
from plb.algorithms.discor.run_sac import train as train_sac
from plb.algorithms.logger import Logger
from plb.algorithms.ppo.run_ppo import train_ppo
from plb.algorithms.solve_bc_transformer_nearest import (visualize_mtrix,
                                                         visualize_mtrix_pick)
from plb.algorithms.TD3.run_td3 import train_td3
from plb.envs import make
from plb.optimizer.solver import optimize_mppi, solve_action, solve_mppi
from plb.optimizer.solver_nn import solve_nn

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


class FourRoombaController:
    def __init__(self):
        rospy.init_node('multi_roomba_controller', anonymous=True)
        
        # Publisher
        self.cmd_vel_pub1 = rospy.Publisher('/roomba1/cmd_vel', Twist, queue_size=10)
        self.cmd_vel_pub2 = rospy.Publisher('/roomba2/cmd_vel', Twist, queue_size=10)
        self.cmd_vel_pub3 = rospy.Publisher('/roomba3/cmd_vel', Twist, queue_size=10)
        self.cmd_vel_pub4 = rospy.Publisher('/roomba4/cmd_vel', Twist, queue_size=10)

        # Subscriber
        odom_sub1 = rospy.Subscriber('/roomba1/odom', Odometry, self.callback_odom1)
        odom_sub2 = rospy.Subscriber('/roomba2/odom', Odometry, self.callback_odom2)
        odom_sub3 = rospy.Subscriber('/roomba3/odom', Odometry, self.callback_odom3)
        odom_sub4 = rospy.Subscriber('/roomba4/odom', Odometry, self.callback_odom4)

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

    def time_control(self, velocity1, velocity2, velocity3, velocity4, yawrate, time):
        vel1 = Twist()
        vel2 = Twist()
        vel3 = Twist()
        vel4 = Twist()
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
            self.cmd_vel_pub1.publish(vel1)
            self.cmd_vel_pub2.publish(vel2)
            self.cmd_vel_pub3.publish(vel3)
            self.cmd_vel_pub4.publish(vel4)
            rospy.sleep(0.1)
        return

    def turn(self, yaw1, yaw2, yaw3, yaw4, yawrate1, yawrate2, yawrate3, yawrate4):
        vel1 = Twist()
        vel2 = Twist()
        vel3 = Twist()
        vel4 = Twist()
        yaw_initial1 = self.yaw1
        yaw_initial2 = self.yaw2
        yaw_initial3 = self.yaw3
        yaw_initial4 = self.yaw4
        done1 = False
        done2 = False
        done3 = False
        done4 = False
        done5 = False
        while not (done1 & done2 & done3 & done4):
            print('rotate')
            vel1.linear.x = 0.0
            vel1.angular.z = yawrate1
            vel2.linear.x = 0.0
            vel2.angular.z = yawrate2
            vel3.linear.x = 0.0
            vel3.angular.z = yawrate3
            vel4.linear.x = 0.0
            vel4.angular.z = yawrate4
            self.cmd_vel_pub1.publish(vel1)
            self.cmd_vel_pub2.publish(vel2)
            self.cmd_vel_pub3.publish(vel3)
            self.cmd_vel_pub4.publish(vel4)
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

class TwoRoombaController:
    def __init__(self):
        rospy.init_node('multi_roomba_controller', anonymous=True)
        
        # Publisher
        self.cmd_vel_pub1 = rospy.Publisher('/roomba1/cmd_vel', Twist, queue_size=10)
        self.cmd_vel_pub2 = rospy.Publisher('/roomba2/cmd_vel', Twist, queue_size=10)

        # Subscriber
        odom_sub1 = rospy.Subscriber('/roomba1/odom', Odometry, self.callback_odom1)
        odom_sub2 = rospy.Subscriber('/roomba2/odom', Odometry, self.callback_odom2)

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

    def callback_odom1(self, data):
        self.x1 = data.pose.pose.position.x
        self.y1 = data.pose.pose.position.y
        self.yaw1 = self.get_yaw_from_quaternion(data.pose.pose.orientation)
    
    def callback_odom2(self, data):
        self.x2 = data.pose.pose.position.x
        self.y2 = data.pose.pose.position.y
        self.yaw2 = self.get_yaw_from_quaternion(data.pose.pose.orientation)

    def time_control(self, velocity1, velocity2, yawrate, time):
        vel1 = Twist()
        vel2 = Twist()
        start_time = rospy.get_rostime().secs
        while(rospy.get_rostime().secs-start_time<time):
            print('move')
            vel1.linear.x = velocity1
            vel1.angular.z = yawrate
            vel2.linear.x = velocity2
            vel2.angular.z = yawrate
            self.cmd_vel_pub1.publish(vel1)
            self.cmd_vel_pub2.publish(vel2)
            rospy.sleep(0.1)
        return

    def turn(self, yaw1, yaw2, yawrate1, yawrate2):
        vel1 = Twist()
        vel2 = Twist()
        yaw_initial1 = self.yaw1
        yaw_initial2 = self.yaw2
        done1 = False
        done2 = False
        while not (done1 & done2):
            print('rotate')
            vel1.linear.x = 0.0
            vel1.angular.z = yawrate1
            vel2.linear.x = 0.0
            vel2.angular.z = yawrate2
            self.cmd_vel_pub1.publish(vel1)
            self.cmd_vel_pub2.publish(vel2)
            if not (abs(self.yaw1-yaw_initial1)<np.deg2rad(yaw1)):
                done1 = True
                yawrate1 = 0
            if not (abs(self.yaw2-yaw_initial2)<np.deg2rad(yaw2)):
                done2 = True
                yawrate2 = 0
            rospy.sleep(0.1)
        return

    def get_yaw_from_quaternion(self, quaternion):
        e = tf.transformations.euler_from_quaternion(
                (quaternion.x, quaternion.y, quaternion.z, quaternion.w))
        return e[2]


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
    parser.add_argument("--policy_params_path", type=str, default='/root/roomba_hack/pbm/bc/jax_decision_transformer/dt_runs/bc_transformer_nearest_delay_0_multi_bc_rope_6/seed_0/22-10-11-13-09-42/model_3000.pt')
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
    
    if args.task_name in ['multi_bc_rope_5_center', 'multi_bc_rope_5_side']:
        multi_roomba_controller = MultiRoombaController()
    elif args.task_name in ['multi_bc_rope_4_center']:
        print('set four_roomba_controller')
        multi_roomba_controller = FourRoombaController()
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
    dataset_path = f'/root/roomba_hack/pbm/bc/jax_decision_transformer/data/bc_data_all.pkl'

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
    
    with open('/root/roomba_hack/pbm/bc/jax_decision_transformer/data/all_data_delay_0.pickle', 'rb') as f: 
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
    
    
    eval_ep_len = 0
    each_eval_step = 25
    attn_weights_list = []
    while eval_ep_len < args.max_eval_ep_len:
        print('timestep', eval_ep_len)
        # initialize new state
        results, actions, attn_weights, imgs = evaluate_on_env(policy_model,
                                                                    training_state.policy_params,
                                                                    parametric_action_distribution,
                                                                    env,
                                                                    None,
                                                                    1,
                                                                    each_eval_step,
                                                                    state_mean,
                                                                    state_std,
                                                                    args,
                                                                    log_dir,
                                                                    giff_save_iters=1,
                                                                    method='bc_transformer_mask_delay',
                                                                    save_gif=True,
                                                                    real_demo=True,
                                                                    total_eval_step=eval_ep_len,
                                                                    init_iou=init_iou)
        eval_ep_len += each_eval_step
        attn_weights_list.extend(attn_weights)
        
        if len(actions) == 0:
            break

        def excute_command(seq, first_vec, split):
            command_list = []
            bf_vec = first_vec
            for i in range(int(len(seq)/split)):
                vec = np.delete(sum(seq[i*split:(i+1)*split]), 1)/(split*12)
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
                if angle < 8:
                    yawrate = 0
                    angle = 0
                command_list.append([v, yawrate, angle])
            return command_list
        
        from tasks.multi_bc_rope.multi_bc_rope_real2sim import \
            multi_bc_rope_real2sim
        roomba_info = multi_bc_rope_real2sim(real_demo=True)
        roomba_vec = roomba_info.get_roomba_first_vec()
        print(roomba_vec)
        split = 5

        excute_command_list = []
        for i in range(int(actions.shape[1]/3)):
            seq = actions[:, i*3:(i+1)*3]
            command_list = excute_command(seq, roomba_vec[i], split)
            excute_command_list.append(command_list)

        for r1, r2, r3, r4, r5 in zip(excute_command_list[0], excute_command_list[2], excute_command_list[1], excute_command_list[3], excute_command_list[4]):
            v1, yawrate1, angle1 = r1
            v2, yawrate2, angle2 = r2
            v3, yawrate3, angle3 = r3
            v4, yawrate4, angle4 = r4
            v5, yawrate5, angle5 = r5
            multi_roomba_controller.turn(angle1, angle2, angle3, angle4, angle5, yawrate1, yawrate2, yawrate3, yawrate4, yawrate5)
            multi_roomba_controller.time_control(v1, v2, v3, v4, v5, 0, 0.0000001)

        env.reset()
        env.taichi_env.loss.init_iou = init_iou
        frames = []
        for i in range(len(actions)):
            obs, r, done, loss_info = env.step(actions[i])
            if i % 5 == 0:
                img = env.render(mode='rgb_array')
                pimg = Image.fromarray(img)
                frames.append(pimg)
                last_iou = loss_info['incremental_iou']
                print(last_iou)
        frames[0].save(f'{args.path}/{args.env_name}_{len(actions)}_{last_iou}.gif', save_all=True, append_images=frames[1:], loop=0)
        
        if eval_ep_len < args.max_eval_ep_len:
            env.reset_new_env()
            env.reset()
            env.taichi_env.loss.init_iou = init_iou
            env.seed(args.seed)
            loss_info = env.compute_loss()
            last_iou = loss_info['incremental_iou']
            cv2.imwrite(f"{log_dir}/last_sim_state_{last_iou}_{eval_ep_len}.png", img[..., ::-1])
    
    set_random_seed(args.seed)
    env = make(args.env_name, nn=(args.algo=='nn'), sdf_loss=args.sdf_loss,
                            density_loss=args.density_loss, contact_loss=args.contact_loss,
                            soft_contact_loss=args.soft_contact_loss)
    state = env.reset()
    env.taichi_env.loss.init_iou = init_iou
    env.seed(args.seed)
    loss_info = env.compute_loss()
    last_iou = loss_info['incremental_iou']
    img = env.render(mode='rgb_array')
    cv2.imwrite(f"{log_dir}/last_sim_state_{last_iou}.png", img[..., ::-1])
    with open(f'{log_dir}/iou_{args.env_name}_{last_iou}_{eval_ep_len}_last.txt', 'w') as f:
        f.write(str(last_iou))
    visualize_mtrix_pick(attn_weights_list, primitive_num, log_dir, imgs, args)
    visualize_mtrix(attn_weights_list, primitive_num, log_dir, args)