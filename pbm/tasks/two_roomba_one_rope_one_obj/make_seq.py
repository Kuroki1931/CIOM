import math
import numpy as np
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from mpc.two_roomba_one_rope_one_obj.two_roomba_one_rope_one_obj_real2sim import two_roomba_one_rope_one_obj_real2sim


split = 5
mpc = 10000000

def make_command(seq, first_vec):
    bf_vec = first_vec
    for i in range(int(len(seq)/split)):
        vec = np.delete(sum(seq[i*split:(i+1)*split]), 1)/(split*50)
        v = (vec[0]**2 + vec[1]**2)**(1/2)

        product =np.dot(vec,bf_vec)
        normu = np.linalg.norm(vec)
        normv = np.linalg.norm(bf_vec)
        cost = product /(normu * normv)
        angle = np.rad2deg(np.arccos(cost))
        cross = np.cross(vec, bf_vec)
        bf_vec = vec
        # print('while True:')
        # print('    dt = datetime.datetime.now()')
        # print('    if dt.second % sec == 0:')
        # print('        break')
        print(f'simple_controller1.time_control(simple_controller1.cmd_vel_pub, {v}, 0.0, 0.00001)')
        if cross >= 0:
            print(f'simple_controller1.turn_left({angle})')
        if cross < 0:
            print(f'simple_controller1.turn_right({angle})')
        print(f'simple_controller2.time_control(simple_controller2.cmd_vel_pub, {v}, 0.0, 0.00001)')
        if cross >= 0:
            print(f'simple_controller2.turn_left({angle})')
        if cross < 0:
            print(f'simple_controller2.turn_right({angle})')
        if i == mpc / split:
            break


obj = two_roomba_one_rope_one_obj_real2sim()
roomba_vec = obj.get_roomba_first_vec()
seq_base = np.load('mpc/two_roomba_one_rope_one_obj/1st/action.npy')
print(seq_base)
left_seq = seq_base[:, :3]
print('left_seq')
make_command(left_seq, roomba_vec[0])
print('-'*60)
right_seq = seq_base[:, 3:]
print('right_seq')
make_command(right_seq, roomba_vec[1])



    