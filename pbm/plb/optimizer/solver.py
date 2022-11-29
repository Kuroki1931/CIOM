import os
import cv2
import pdb
import yaml
import pickle
import codecs
import datetime
import numpy as np
import taichi as ti
from PIL import Image
import matplotlib.pyplot as plt

from plb.algorithms.logger import Logger
from yacs.config import CfgNode as CN
from .optim import Optimizer, Adam, Momentum
from ..engine.taichi_env import TaichiEnv
from ..config.utils import make_cls_config

OPTIMS = {
    'Adam': Adam,
    'Momentum': Momentum
}

class Solver:
    def __init__(self, env: TaichiEnv, logger=None, cfg=None, **kwargs):
        self.cfg = make_cls_config(self, cfg, **kwargs)
        self.optim_cfg = self.cfg.optim
        self.env = env
        self.logger = logger

    def solve(self, path, init_actions=None, callbacks=()):
        env = self.env
        if init_actions is None:
            init_actions = self.init_actions(env, self.cfg)
        # initialize ...
        optim = OPTIMS[self.optim_cfg.type](init_actions, self.optim_cfg)
        # set softness ..
        env_state = env.get_state()
        with open(f'{path}/initial_state.json', 'wb') as fp:
            pickle.dump(env_state, fp)
        self.total_steps = 0

        def forward(sim_state, action):
            if self.logger is not None:
                self.logger.reset()

            env.set_state(sim_state, self.cfg.softness, False)
            with ti.Tape(loss=env.loss.loss):
                for i in range(len(action)):
                    env.step(action[i])
                    self.total_steps += 1
                    loss_info = env.compute_loss()
                    if self.logger is not None:
                        self.logger.step(None, action, loss_info['reward'], None, i==len(action)-1, loss_info)
            loss = env.loss.loss[None]
            return loss, env.primitives.get_grad(len(action))

        best_action = None
        best_loss = 1e10
        best_iter = None

        actions = init_actions
        for iter in range(self.cfg.n_iters):
            self.params = actions.copy()
            loss, grad = forward(env_state['state'], actions)
            
            if loss < best_loss:
                best_loss = loss
                best_iter = iter
                best_action = actions.copy()
            actions = optim.step(grad)
            for callback in callbacks:
                callback(self, optim, loss, grad)

            if not os.path.exists(f'{path}/actions'):
                os.makedirs(f'{path}/actions', exist_ok=True)
            np.save(f"{path}/actions/action_iter{iter}-{self.cfg.n_iters-1}.npy", actions)
        np.save(f"{path}/best_action.npy", best_action)
        return best_action

    @staticmethod
    def init_actions(env, cfg):
        action_dim = env.primitives.action_dim
        horizon = cfg.horizon
        if cfg.init_sampler == 'uniform':
            return np.random.uniform(-cfg.init_range, cfg.init_range, size=(horizon, action_dim))
        else:
            raise NotImplementedError
    
    @staticmethod
    def add_init_actions(env, cfg, action):
        action_dim = env.primitives.action_dim
        horizon = cfg.horizon - len(action)
        if cfg.init_sampler == 'uniform':
            random_action = np.random.uniform(-cfg.init_range, cfg.init_range, size=(horizon, action_dim))
            return np.concatenate([action, random_action])
        else:
            raise NotImplementedError

    @classmethod
    def default_config(cls):
        cfg = CN()
        cfg.optim = Optimizer.default_config()
        cfg.n_iters = 100
        cfg.softness = 666.
        cfg.horizon = 50

        cfg.init_range = 0.
        cfg.init_sampler = 'uniform'
        return cfg


def solve_action(env, path, logger, args, actions=None):
    env.reset()
    img = env.render(mode='rgb_array')
    cv2.imwrite(f"{path}/init.png", img[..., ::-1])
    taichi_env: TaichiEnv = env.unwrapped.taichi_env
    T = env._max_episode_steps
    solver = Solver(taichi_env, logger, None,
                    n_iters=(args.num_steps + T-1)//T, softness=args.softness, horizon=T,
                    **{"optim.lr": args.lr, "optim.type": args.optim, "init_range": 0.001})
    
    action = solver.solve(path, actions)

    # create dataset for bc
    env.reset()
    action_list = []
    current_obs_list = []
    next_obs_list = []
    reward_list = []
    loss_info_list = []
    last_iou_list = []
    
    goal_state = env.get_goal_obs(args)
    frames = []
    for i in range(len(action)):
        action_list.append(action[i])
        current_obs = list(env.get_obs(i))
        obs, r, done, loss_info = env.step(action[i])
        if i % 10 == 0 or i+1 == len(action):
            img = env.render(mode='rgb_array')
            pimg = Image.fromarray(img)
            frames.append(pimg)
        next_obs = list(env.get_obs(i+1))
        last_iou = loss_info['incremental_iou']
        current_obs_list.append(current_obs)
        next_obs_list.append(next_obs)
        reward_list.append(r)
        loss_info_list.append(loss_info)
        last_iou_list.append(last_iou)
    
    if not os.path.exists(f'{path}/../../../traj'):
        os.makedirs(f'{path}/../../../traj', exist_ok=True)

    print('length', i, 'r', r, 'last_iou', last_iou)
    bc_data = {
        'action': np.array(action_list),
        'rewards': np.array(reward_list),
        'env_name': args.env_name,
        'goal_state': goal_state,
        'observation': np.array(current_obs_list),
        'next_observation': np.array(next_obs_list),
        'loss_info_list': loss_info_list
    }
    
    print(action.shape, np.array(reward_list).shape, goal_state.shape, np.array(current_obs_list).shape, np.array(next_obs_list).shape)
    with open(f'{path}/../../../traj/{args.env_name}_{last_iou}_{args.num_steps}_traj.pickle', 'wb') as f:
        pickle.dump(bc_data, f)
    with open(f'{path}/../../../traj/iou_{args.env_name}_{last_iou}.txt', 'w') as f:
        f.write(str(last_iou))
    with open(f'{path}/{args.env_name}_{last_iou}_{args.num_steps}_traj.pickle', 'wb') as f:
            pickle.dump(bc_data, f)
    frames[0].save(f'{path}/{args.env_name}_{last_iou}_{len(action)}.gif', save_all=True, append_images=frames[1:], loop=0)