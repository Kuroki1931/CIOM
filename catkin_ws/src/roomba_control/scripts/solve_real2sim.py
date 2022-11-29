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

sys.path.append(os.path.join(os.path.dirname(__file__), '/root/roomba_hack/pbm/bc/network'))
from transformer.bc_transformer_nearest.model import make_transformers
from transformer.bc_transformer_nearest.utils import ReplayBuffer, TrainingState, Transition
from transformer.bc_transformer_nearest.utils import evaluate_on_env, save_params, load_params
from transformer.bc_transformer_nearest.mask_config import MASK_CONFIG
from transformer.pmap import bcast_local_devices, synchronize_hosts, is_replicated

sys.path.append(os.path.join(os.path.dirname(__file__), '/root/roomba_hack/pbm'))
from plb.envs import make
from plb.algorithms.logger import Logger
from plb.algorithms.discor.run_sac import train as train_sac
from plb.algorithms.ppo.run_ppo import train_ppo
from plb.algorithms.TD3.run_td3 import train_td3
from plb.optimizer.solver import solve_action, solve_mppi, optimize_mppi
from plb.optimizer.solver_nn import solve_nn

os.environ['TI_USE_UNIFIED_MEMORY'] = '0'
os.environ['TI_DEVICE_MEMORY_FRACTION'] = '0.9'
os.environ['TI_DEVICE_MEMORY_GB'] = '4'
os.environ['TI_ENABLE_CUDA'] = '0'
os.environ['TI_ENABLE_OPENGL'] = '0'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


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


class OneRoombaController:
    def __init__(self):
        rospy.init_node('multi_roomba_controller', anonymous=True)
        
        # Publisher
        self.cmd_vel_pub1 = rospy.Publisher('/roomba1/cmd_vel', Twist, queue_size=10)

        # Subscriber
        odom_sub1 = rospy.Subscriber('/roomba1/odom', Odometry, self.callback_odom1)

        self.x1= None
        self.y1 = None
        self.yaw1 = None
        while self.x1 is None:
            rospy.sleep(0.1)

    def callback_odom1(self, data):
        self.x1 = data.pose.pose.position.x
        self.y1 = data.pose.pose.position.y
        self.yaw1 = self.get_yaw_from_quaternion(data.pose.pose.orientation)

    def time_control(self, velocity1, yawrate, time):
        vel1 = Twist()
        start_time = rospy.get_rostime().secs
        while(rospy.get_rostime().secs-start_time<time):
            print('move')
            vel1.linear.x = velocity1
            vel1.angular.z = yawrate
            self.cmd_vel_pub1.publish(vel1)
            rospy.sleep(0.1)
        return

    def turn(self, yaw1, yawrate1):
        vel1 = Twist()
        yaw_initial1 = self.yaw1
        done1 = False
        while not (done1):
            print('rotate')
            vel1.linear.x = 0.0
            vel1.angular.z = yawrate1
            self.cmd_vel_pub1.publish(vel1)
            if not (abs(self.yaw1-yaw_initial1)<np.deg2rad(yaw1)):
                done1 = True
                yawrate1 = 0
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
    parser.add_argument("--max_eval_ep_len", type=int, default=80)
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

STEPS = 180
EPISODES = 20

if __name__=='__main__':
    args = get_args()
    idx = args.env_name.find('-')
    args.task_name = args.env_name[:idx]
    args.task_version = args.env_name[(idx+1):]
    args.num_steps = STEPS * EPISODES
    args.path = f'/root/roomba_hack/pbm/{args.path}/{args.task_name}/{args.task_version}/{args.algo}/{args.picture_name}_{STEPS}_{args.num_steps}_{args.sdf_loss}_{args.density_loss}_{args.contact_loss}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    
    if args.task_name in ['push_box']:
        multi_roomba_controller = OneRoombaController()
    elif args.task_name in ['two_roomba_deform_one_rope']:
        multi_roomba_controller = TwoRoombaController()
    elif args.task_name in ['two_roomba_wrap_rope']:
        multi_roomba_controller = TwoRoombaController()
    elif args.task_name in ['two_roomba_rotate_one_box']:
        multi_roomba_controller = TwoRoombaController()

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
    init_iou = env.taichi_env.loss.init_iou
    primitive_num = int(env.action_space.shape[0]/3)
    
    steps = copy(STEPS)
    half = int(STEPS/2)
    base = 0
    while base < STEPS:
        print('base half', base, half)
        base += half
        args.num_steps = half * EPISODES
        env._max_episode_steps = half
        steps -= half
        
        actions = solve_action(env, args.path, logger, args)
         
        if args.task_name in ['push_box']:
            from tasks.one_roomba_one_box.one_roomba_one_box_real2sim import one_roomba_one_box_real2sim
            real2sim = one_roomba_one_box_real2sim(args.create_grid_mass, args.task_version)
        elif args.task_name in ['two_roomba_deform_one_rope']:
            from tasks.two_roomba_one_rope.two_roomba_one_rope_real2sim import two_roomba_one_rope_real2sim
            real2sim = two_roomba_one_rope_real2sim(args.create_grid_mass, args.task_version, args.picture_name)
        elif args.task_name in ['two_roomba_wrap_rope']:
            from tasks.two_roomba_one_rope_one_obj.two_roomba_one_rope_one_obj_real2sim import two_roomba_one_rope_one_obj_real2sim
            real2sim = two_roomba_one_rope_one_obj_real2sim(args.create_grid_mass, args.task_version, args.picture_name)
        elif args.task_name in ['two_roomba_rotate_one_box']:
            from tasks.two_roomba_one_box.two_roomba_one_box_real2sim import two_roomba_one_box_real2sim
            real2sim = two_roomba_one_box_real2sim(args.create_grid_mass, args.task_version)

        roomba_vec = real2sim.get_roomba_first_vec()
        
        def excute_command(seq, first_vec, split):
            command_list = []
            bf_vec = first_vec
            for i in range(int(len(seq)/split)):
                vec = np.delete(sum(seq[i*split:(i+1)*split]), 1)/(5*25)
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
                print(v, yawrate, angle)
            return command_list
        
        excute_command_list = []
        for i in range(int(actions.shape[1]/3)):
            seq = actions[:, i*3:(i+1)*3]
            split = int(len(seq)/20)
            command_list = excute_command(seq, roomba_vec[i], 5)
            excute_command_list.append(command_list)
        
        for r1, r2 in zip(excute_command_list[0], excute_command_list[1]):
                v1, yawrate1, angle1 = r1
                v2, yawrate2, angle2 = r2
                multi_roomba_controller.turn(angle1, angle2, yawrate1, yawrate2)
                multi_roomba_controller.time_control(v1, v2, 0, 0.0000001)

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
        frames[0].save(f'{args.path}/{args.env_name}_{len(actions)}_{base}_{last_iou}.gif', save_all=True, append_images=frames[1:], loop=0)
        
        set_random_seed(args.seed)
        env = make(args.env_name, nn=(args.algo=='nn'), sdf_loss=args.sdf_loss,
                                density_loss=args.density_loss, contact_loss=args.contact_loss,
                                soft_contact_loss=args.soft_contact_loss)
        env.seed(args.seed)
        env.taichi_env.loss.init_iou = init_iou
        img = env.render(mode='rgb_array')
        cv2.imwrite(f"{args.path}/last_sim_state_{len(actions)}.png", img[..., ::-1])

    env.reset()
    env.taichi_env.loss.init_iou = init_iou
    loss_info = env.compute_loss()
    last_iou = loss_info['incremental_iou']
    img = env.render(mode='rgb_array')
    cv2.imwrite(f"{args.path}/last_sim_state_{last_iou}.png", img[..., ::-1])
    with open(f'{args.path}/iou_{args.env_name}_{last_iou}_{STEPS}_last.txt', 'w') as f:
        f.write(str(last_iou))
        