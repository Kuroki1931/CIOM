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


def evaluate_on_env(policy_model,
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
                    giff_save_iters=3,
                    method='bc',
                    render=False,
                    save_gif=False,
                    task_name="rope"):

    
    policy_params = jax.tree_map(lambda x: x[0], policy_params)
    policy_model_apply = jax.jit(policy_model.apply)

    results = {}
    total_reward = 0
    total_timesteps = 0

    primitive_num = int(env.action_space.shape[0]/3)
    state_dim = (args.base_data_size*3*2+primitive_num*3)*primitive_num+args.base_data_size*3
 
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
        
        goal_state = env.get_goal_obs(args)
        state = organaize_state(state, goal_state, primitive_num, state_mean, state_std)

        running_reward = 0
        sum_last_iou = 0

        imgs = []
        actions_list = []
        for t in range(max_test_ep_len):

            total_timesteps += 1
            act = jnp.tanh(policy_model.apply(policy_params, state))
        
            for j in range(int(len(act)/2)):
                act = np.insert(act, j*3+1, 0)
            actions_list.append(act)
            state, running_reward, done, loss_info = env.step(np.array(act))
            state = organaize_state(state, goal_state, primitive_num, state_mean, state_std)

            total_reward += running_reward
            last_iou = loss_info['incremental_iou']
           
            # save_gif.
            if ((save_gif and num_test_loop == 1) | (i_train_iter % giff_save_iters == 0 and num_test_loop == 1)) & ((t+1)%5==0 or t == 0):
                print(f"Saving gif at {t} steps")
                imgs.append(Image.fromarray(env.render(mode='rgb_array')))
            if render:
                env.render()
            if last_iou > 0.75:
                total_success += 1
                break
        sum_last_iou += last_iou

        if (save_gif and num_test_loop == 1) | (i_train_iter % giff_save_iters == 0 and num_test_loop == 1):
            imgs[0].save(f"{log_dir}/{args.env_name}_{last_iou}_{i_train_iter}_{t}.gif", save_all=True, append_images=imgs[1:], loop=0)
            with open(f'{log_dir}/iou_{args.env_name}_{last_iou}_{i_train_iter}_{t}.txt', 'w') as f:
                f.write(str(last_iou))
            print("Saved!!")

        
    results['eval/avg_reward'] = total_reward / num_eval_ep
    results['eval/avg_success'] = total_success / num_eval_ep
    results['eval/avg_ep_len'] = total_timesteps / num_eval_ep
    results['eval/avg_last_iou'] = sum_last_iou / num_eval_ep

    return results, np.array(actions_list)


@flax.struct.dataclass
class ReplayBuffer:
    """Contains data related to a replay buffer."""
    data: jnp.ndarray


@flax.struct.dataclass
class Transition:
    """Contains data for contextual-BC training step."""
    s_t: jnp.ndarray
    a_t: jnp.ndarray


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


def organaize_state(state, goal_state, primitive_num, state_mean, state_std):
    ## change observation
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
    
    obs_list = []
    # primitive to rope, goal, primitive
    for i in range(int(primitive_num)):
        primitive_xzy = primitives_xzy[i*3:(i+1)*3]
        primitive_to_rope = (rope_xzy - primitive_xzy).reshape(-1)
        primitive_to_goal = (goal_state - primitive_xzy).reshape(-1)
        primitive_to_primitive = (primitives_xzy - np.tile(primitive_xzy,(primitive_num)))
        obs_list.append(primitive_to_rope)
        obs_list.append(primitive_to_goal)
        obs_list.append(primitive_to_primitive)

    # rope to goal
    rope_to_goal = (goal_state - rope_xzy).reshape(-1)
    obs_list.append(rope_to_goal)
    state = np.concatenate(obs_list)
    state = (state - state_mean) / state_std
    return state