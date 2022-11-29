"""Reference: https://github.com/google/brax/blob/main/brax/training/distribution.py
   Probability distributions in JAX.
"""
import os
import abc
import flax
import jax
import optax
import pickle
import random
import time

import jax.numpy as jnp
import numpy as np

from typing import Any

from PIL import Image

class ParametricDistribution(abc.ABC):
    """Abstract class for parametric (action) distribution."""

    def __init__(self, param_size, postprocessor, event_ndims, reparametrizable):
        """Abstract class for parametric (action) distribution.

        Specifies how to transform distribution parameters (i.e. actor output)
        into a distribution over actions.

        Args:
          param_size: size of the parameters for the distribution
          postprocessor: bijector which is applied after sampling (in practice, it's
            tanh or identity)
          event_ndims: rank of the distribution sample (i.e. action)
          reparametrizable: is the distribution reparametrizable
        """
        self._param_size = param_size
        self._postprocessor = postprocessor
        self._event_ndims = event_ndims  # rank of events
        self._reparametrizable = reparametrizable
        assert event_ndims in [0, 1]

    @abc.abstractmethod
    def create_dist(self, parameters):
        """Creates distribution from parameters."""
        pass

    @property
    def param_size(self):
        return self._param_size

    @property
    def reparametrizable(self):
        return self._reparametrizable

    def postprocess(self, event):
        return self._postprocessor.forward(event)

    def inverse_postprocess(self, event):
        return self._postprocessor.inverse(event)

    def sample_no_postprocessing(self, parameters, seed):
        return self.create_dist(parameters).sample(seed=seed)

    def sample(self, parameters, seed):
        """Returns a sample from the postprocessed distribution."""
        return self.postprocess(self.sample_no_postprocessing(parameters, seed))

    def log_prob(self, parameters, actions):
        """Compute the log probability of actions."""
        dist = self.create_dist(parameters)
        log_probs = dist.log_prob(actions)
        log_probs -= self._postprocessor.forward_log_det_jacobian(actions)
        if self._event_ndims == 1:
            log_probs = jnp.sum(log_probs, axis=-1)  # sum over action dimension
        return log_probs

    def entropy(self, parameters, seed):
        """Return the entropy of the given distribution."""
        dist = self.create_dist(parameters)
        entropy = dist.entropy()
        entropy += self._postprocessor.forward_log_det_jacobian(
            dist.sample(seed=seed))
        if self._event_ndims == 1:
            entropy = jnp.sum(entropy, axis=-1)
        return entropy


class NormalDistribution:
    """Normal distribution."""

    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale

    def sample(self, seed):
        return jax.random.normal(seed, shape=self.loc.shape) * self.scale + self.loc

    def log_prob(self, x):
        log_unnormalized = -0.5 * jnp.square(x / self.scale - self.loc / self.scale)
        log_normalization = 0.5 * jnp.log(2. * jnp.pi) + jnp.log(self.scale)
        return log_unnormalized - log_normalization

    def entropy(self):
        log_normalization = 0.5 * jnp.log(2. * jnp.pi) + jnp.log(self.scale)
        entropy = 0.5 + log_normalization
        return entropy * jnp.ones_like(self.loc)


class TanhBijector:
  """Tanh Bijector."""

  def forward(self, x):
      return jnp.tanh(x)

  def inverse(self, y):
      return jnp.arctanh(y)

  def forward_log_det_jacobian(self, x):
      return 2. * (jnp.log(2.) - x - jax.nn.softplus(-2. * x))


class NormalTanhDistribution(ParametricDistribution):
    """Normal distribution followed by tanh."""

    def __init__(self, event_size, min_std=0.001):
        """Initialize the distribution.

        Args:
          event_size: the size of events (i.e. actions).
          min_std: minimum std for the gaussian.
        """
        # We apply tanh to gaussian actions to bound them.
        # Normally we would use TransformedDistribution to automatically
        # apply tanh to the distribution.
        # We can't do it here because of tanh saturation
        # which would make log_prob computations impossible. Instead, most
        # of the code operate on pre-tanh actions and we take the postprocessor
        # jacobian into account in log_prob computations.
        super().__init__(
            param_size=2 * event_size,
            postprocessor=TanhBijector(),
            event_ndims=1,
            reparametrizable=True)
        self._min_std = min_std

    def create_dist(self, parameters):
        loc, scale = jnp.split(parameters, 2, axis=-1)
        scale = jax.nn.softplus(scale) + self._min_std
        dist = NormalDistribution(loc=loc, scale=scale)
        return dist

    def deterministic(self, parameters):
        loc, _ = jnp.split(parameters, 2, axis=-1)
        return self._postprocessor.forward(loc)


def evaluate_on_env_random_delay(policy_model,
                                policy_params,
                                distribution,
                                env,
                                key=None,
                                num_eval_ep=10,
                                max_test_ep_len=100,
                                state_mean=None,
                                state_std=None,
                                args=None,
                                log_dir=None,
                                i_train_iter=0,
                                giff_save_iters=1,
                                method='bc_transformer_mask_delay',
                                save_gif=False,
                                real_demo=False,
                                total_eval_step=None,
                                init_iou=None):

    attn_weights_list = []
    policy_params = jax.tree_map(lambda x: x[0], policy_params)
    policy_model_apply = jax.jit(policy_model.apply)

    results = {}
    total_reward = 0
    total_timesteps = 0

    primitive_num = int(env.action_space.shape[0]/3)
    nearest_primitive_pos = 3
    state_dim = args.base_data_size*3*2+nearest_primitive_pos+args.base_data_size*3

    if state_mean is None:
        state_mean = jnp.zeros((state_dim,))
    else:
        state_mean = jnp.array(state_mean)

    if state_std is None:
        state_std = jnp.ones((state_dim,))
    else:
        state_std = jnp.array(state_std)

    _key = jax.random.PRNGKey(999) if key is None else key
    
    num_test_loop = 0
    total_success = 0

    for _, key_i in enumerate(jax.random.split(_key, num=num_eval_ep)):
        # init episode
        num_test_loop += 1
        state = env.reset()
        if real_demo:
            env.taichi_env.loss.init_iou = init_iou

        ## change observation
        randam_sample_range = args.delay_time
        delay_data_base = jnp.zeros((randam_sample_range, primitive_num, state_dim))
        state, delay_data_base, delay_roomba_pos, mask = organaize_state_random_delay(state, env, args, primitive_num, state_mean, state_std, delay_data_base)

        running_reward = 0
        sum_last_iou = 0
        
        loss_info = env.compute_loss()
        last_iou = loss_info['incremental_iou']
        if (real_demo) & (last_iou > 0.75):
            total_success += 1
            print('real_demo success!')
            break

        imgs = []
        actions_list = []
        for t in range(max_test_ep_len):
            total_timesteps += 1
            
            act, attn_weights = policy_model.apply(policy_params, state, mask)
            attn_weights_list.append(attn_weights)
            act = jnp.tanh(act[:, :primitive_num*2]).reshape(-1)
            
            for j in range(int(len(act)/2)):
                act = np.insert(act, j*3+1, 0)
            actions_list.append(act)
            state, running_reward, done, loss_info = env.step(np.array(act))
            state, delay_data_base, delay_roomba_pos, mask = organaize_state_random_delay(state, env, args, primitive_num, state_mean, state_std, delay_data_base, delay_roomba_pos, t+1)

            total_reward += running_reward
            last_iou = loss_info['incremental_iou']
           
            # save_gif.
            if (save_gif) & (i_train_iter % giff_save_iters == 0 and num_test_loop == 1) & ((t+1)%1000==0 or t == 0):
                print(f"Saving gif at {t} steps")
                imgs.append(Image.fromarray(env.render(mode='rgb_array')))
                # pass
            
            if (not real_demo) & (last_iou > 0.75):
                total_success += 1
                break

        sum_last_iou += last_iou

        if (save_gif) & (i_train_iter % giff_save_iters == 0 and num_test_loop == 1):
            # imgs[0].save(f"{log_dir}/{args.env_name}_{last_iou}_{t}.gif", save_all=True, append_images=imgs[1:], loop=0)
            with open(f'{log_dir}/iou_{args.env_name}_{last_iou}_{i_train_iter}_{t}.txt', 'w') as f:
                f.write(str(last_iou))
            print("Saved!!")
            # pass

    results['eval/avg_reward'] = total_reward / num_eval_ep
    results['eval/avg_success'] = total_success / num_test_loop
    results['eval/avg_ep_len'] = total_timesteps / num_eval_ep
    results['eval/avg_last_iou'] = sum_last_iou / num_test_loop

    return results, np.array(actions_list), attn_weights_list, imgs


@flax.struct.dataclass
class ReplayBuffer:
    """Contains data related to a replay buffer."""
    data: jnp.ndarray


@flax.struct.dataclass
class Transition:
    """Contains data for contextual-BC training step."""
    s_t: jnp.ndarray
    a_t: jnp.ndarray
    m_t: jnp.ndarray


@flax.struct.dataclass
class TrainingState:
    """Contains training state for the learner."""
    policy_optimizer_state: optax.OptState
    policy_params: Any
    key: jnp.ndarray
    actor_steps: jnp.ndarray


class File:
    """General purpose file resource."""
    def __init__(self, fileName: str, mode='r'):
        self.f = None
        if not self.f:
            self.f = open(fileName, mode)

    def __enter__(self):
        return self.f

    def __exit__(self, exc_type, exc_value, traceback):
        self.f.close()


def save_params(path: str, params: Any):
    """Saves parameters in Flax format."""
    with File(path, 'wb') as fout:
        fout.write(pickle.dumps(params))


def load_params(path: str) -> Any:
  with File(path, 'rb') as fin:
    buf = fin.read()
  return pickle.loads(buf)


def organaize_state_random_delay(state, env, args, primitive_num, state_mean, state_std, delay_data_base, delay_roomba_pos=None, t=0):
    ## change observation
    goal_state = env.get_goal_obs(args)
    seg_size = goal_state.shape[0]
    step_size = int((seg_size/3) // 100)
    goal_state = goal_state.reshape(-1, 3)[::step_size, :]
    unit_length = goal_state.shape[0]

    # obs separate to rope and primitives
    obs_reshape = state[:unit_length*6].reshape(-1, 3)
    rope_xzy = obs_reshape[::2, :]
    obs_primitives = state[unit_length*6:]

    # primitives separate to primitive
    primitive_xzy_list = []
    for i in range(int(obs_primitives.shape[0]/7)):
        primitive_xzy = obs_primitives[i*7:(i+1)*7][:3]
        primitive_xzy_list.append(primitive_xzy)
    primitives_xzy = np.concatenate(primitive_xzy_list)
    if t:
        delay_roomba_pos = np.concatenate([delay_roomba_pos, primitives_xzy[None, :]], axis=0)
        primitives_xzy_5_delay = delay_roomba_pos[t, :]
    else:        
        delay_roomba_pos = np.repeat(primitives_xzy[None, :], args.delay_time, axis=0)
        delay_roomba_pos = np.concatenate([delay_roomba_pos, primitives_xzy[None, :]], axis=0)
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
    delay_data_base = jnp.concatenate([delay_data_base, state_base], axis=0)

    random_delay_data_base = delay_data_base[t:t+args.delay_time]
    random_delay_list = []
    for i in range(primitive_num):
        row = random.randint(0, args.delay_time)
        random_delay_list.append(random_delay_data_base[row][i][None, :])
    random_delay_data = np.concatenate(random_delay_list)
    state = jnp.concatenate([state_base, random_delay_data[None, :, :]], axis=1)
    
    # normalize
    state = (state - state_mean) / state_std

    return state, delay_data_base, delay_roomba_pos, mask