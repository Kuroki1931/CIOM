import dataclasses
import jax
import jax.numpy as jnp

from flax import linen
from typing import Any, Callable, Sequence, Tuple


@dataclasses.dataclass
class FeedForwardModel:
    init: Any
    apply: Any


class MLP(linen.Module):
    """MLP module."""
    layer_sizes: Sequence[int]
    activation: Callable[[jnp.ndarray], jnp.ndarray] = linen.relu
    kernel_init: Callable[..., Any] = jax.nn.initializers.lecun_uniform()
    activate_final: bool = False
    bias: bool = True

    @linen.compact
    def __call__(self, data: jnp.ndarray):
        hidden = data
        for i, hidden_size in enumerate(self.layer_sizes):
            hidden = linen.Dense(
                hidden_size,
                name=f'hidden_{i}',
                kernel_init=self.kernel_init,
                use_bias=self.bias)(hidden)
            if i != len(self.layer_sizes) - 1 or self.activate_final:
                hidden = self.activation(hidden)
        return hidden


def make_policy_networks(policy_params_size: int,
                         state_dim: int,
                         hidden_layer_sizes: Tuple[int, ...] = (256, 256),
                         ) -> FeedForwardModel:
    """Creates a MLP policy for BC."""
    policy_module = MLP(
        layer_sizes=hidden_layer_sizes + (policy_params_size,),
        activation=linen.relu,
        kernel_init=jax.nn.initializers.lecun_uniform())
    dummy_obs = jnp.zeros((1, state_dim))
    policy = FeedForwardModel(
        init=lambda key: policy_module.init(key, dummy_obs),
        apply=policy_module.apply)
    return policy
