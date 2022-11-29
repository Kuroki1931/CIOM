import math
import numpy as np
from mpc.two_roomba_one_rope.two_roomba_one_rope_real2sim import two_roomba_one_rope_real2sim

split = 5

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
        print(f'simple_controller.time_control(simple_controller.cmd_vel_pub, {v}, 0.0, 0.00001)')
        if cross >= 0:
            print(f'simple_controller.turn_left({angle})')
        if cross < 0:
            print(f'simple_controller.turn_right({angle})')


obj = two_roomba_one_rope_real2sim()
roomba_vec = obj.get_roomba_first_vec()
seq_base = np.load('mpc/two_roomba_one_rope/action.npy')
left_seq = seq_base[:, :3]
print('left_seq')
make_command(left_seq, roomba_vec[0])
print('-'*60)
right_seq = seq_base[:, 3:]
print('right_seq')
make_command(right_seq, roomba_vec[1])



    