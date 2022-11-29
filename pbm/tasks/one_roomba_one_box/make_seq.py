import math
import numpy as np
from one_roomba_one_box_real2sim import one_roomba_one_box_real2sim

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
        print(f'simple_controller1.time_control(simple_controller1.cmd_vel_pub, {v}, 0.0, 0.0000001)')
        if cross >= 0:
            print(f'simple_controller1.turn_left({angle})')
        if cross < 0:
            print(f'simple_controller1.turn_right({angle})')
        # print(f'simple_controller2.time_control(simple_controller2.cmd_vel_pub, {v}, 0.0, 0.0000001)')
        # if cross >= 0:
        #     print(f'simple_controller2.turn_left({angle})')
        # if cross < 0:
        #     print(f'simple_controller2.turn_right({angle})')


obj = one_roomba_one_box_real2sim()
roomba_vec = obj.get_roomba_first_vec()
for i in ['action_gb', 'action_ppo', 'action_mppi']:
    print(i, '-'*60)
    actions = np.load(f'actions/{i}.npy')
    make_command(actions, roomba_vec[0])




    