import glob
from os import lseek
import pickle
import json
import numpy as np
import matplotlib.pyplot as plt


iou_standard = 0.65
particle_observe_size = 100
num_success_sample = 0

remake_data_list = []
files = glob.glob('/root/roomba_hack/pbm/output/multi_bc_rope/traj/*.pickle')
goal_states_list = []
for path in files:
    with open(path, 'rb') as f:
        data = pickle.load(f)
    print(path[65:72], '------', path[62:66])
    iou = float(path[78:84].split('_')[-1]) #TODO take 1 trickey
    version = path[75:79].split('_')[0]

    loss_info = data['loss_info_list']
    for i in range(len(loss_info)):
        loss = loss_info[i]
        if loss['incremental_iou'] > 1:
            valid_data_length = i
            break
        valid_data_length = i

    seg_size = data['goal_state'].shape[0]
    step_size = int((seg_size/3) // particle_observe_size)

    actions = data['action'][:valid_data_length]
    rewards = data['rewards'][:valid_data_length]
    goal_state_all = data['goal_state'].reshape(-1, 3)
    plt.scatter(goal_state_all[:, 0], goal_state_all[:, 2], s=0.001)
    goal_state = goal_state_all[::step_size, :]

    # obs separate to rope and primitives
    observations_base = data['observation'][:valid_data_length, :]
    obs_reshape = observations_base[:, :seg_size*2].reshape(valid_data_length, -1, 3)
    rope_xzy = obs_reshape[:, ::2, :][:, ::step_size, :]
    obs_primitives = observations_base[:, seg_size*2:]

    # primitives separate to primitive
    primitive_xzy_list = []
    for i in range(int(obs_primitives.shape[1]/7)):
        primitive_xzy = obs_primitives[:, i*7:(i+1)*7][:, :3]
        primitive_xzy_list.append(primitive_xzy)
    primitives_xzy = np.concatenate(primitive_xzy_list, axis=1)

    obs_list = []

    # primitive to 
    primitive_num = primitives_xzy.shape[1]/3
    for i in range(int(primitive_num)):
        primitive_xzy = primitives_xzy[:, i*3:(i+1)*3]
        primitive_to_rope = (rope_xzy - primitive_xzy[:, None, :]).reshape(valid_data_length, -1)
        primitive_to_goal = (goal_state - primitive_xzy[:, None, :]).reshape(valid_data_length, -1)
        primitive_to_primitive = (primitives_xzy - np.tile(primitive_xzy,(6)))
        obs_list.append(primitive_to_rope)
        obs_list.append(primitive_to_goal)
        obs_list.append(primitive_to_primitive)

    # rope to goal
    rope_to_goal = (goal_state - rope_xzy).reshape(valid_data_length, -1)
    obs_list.append(rope_to_goal)
    obs = np.concatenate(obs_list, axis=1)

    remake_data = {
        'actions': actions[:, [0, 2, 3, 5, 6, 8, 9, 11, 12, 14, 15, 17]],
        'rewards': rewards,
        'goal_state': goal_state,
        'observations': obs,
    }
    remake_data_list.append(remake_data)
    print(remake_data['actions'].shape, remake_data['rewards'].shape, remake_data['goal_state'].shape, remake_data['observations'].shape)
    print('obssum', np.sum(remake_data['observations']))
    print('actsum', np.sum(remake_data['actions']))
    print('goalsum', np.sum(remake_data['goal_state']))
    
with open(f'/root/roomba_hack/pbm/bc/jax_decision_transformer/data/bc_data.pkl', 'wb') as f:
    pickle.dump(remake_data_list, f)