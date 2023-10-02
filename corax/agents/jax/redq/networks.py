# type: ignore
import dataclasses
import functools
from typing import Callable, Optional, Sequence

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

from corax import specs
from corax.jax import networks as networks_lib
from corax.jax import types as jax_types
from corax.jax import utils as jax_utils


def default_init(scale=1.0):
    return hk.initializers.VarianceScaling(scale, "fan_avg", "uniform")


@dataclasses.dataclass
class MLP(hk.Module):
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = jax.nn.relu
    activate_final: bool = False
    use_layer_norm: bool = False
    scale_final: Optional[float] = None
    use_pnorm: bool = False
    name: Optional[str] = None

    def __call__(self, x: jax.Array) -> jax.Array:
        for i, size in enumerate(self.hidden_dims):
            if i + 1 == len(self.hidden_dims) and self.scale_final is not None:
                x = hk.Linear(size, w_init=default_init(self.scale_final))(x)
            else:
                x = hk.Linear(size, w_init=default_init())(x)

            if i + 1 < len(self.hidden_dims) or self.activate_final:
                if self.use_layer_norm:
                    x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(x)
                x = self.activations(x)

        if self.use_pnorm:
            x /= jnp.linalg.norm(x, axis=-1, keepdims=True).clip(1e-10)

        return x


def ensemble_init(
    base_init: Callable[[jax_types.PRNGKey], hk.Params],
    key: jax_types.PRNGKey,
    ensemble_size: int,
):
    keys = jax.random.split(key, ensemble_size)
    return jax.vmap(base_init)(keys)


def ensemble_all_apply(base_apply, params, *args):
    return jax.vmap(base_apply, in_axes=(0,) + (None,) * len(args))(params, *args)


def subsample_ensemble_params(params, key, subset_size):
    ensemble_size = jax.tree_util.tree_leaves(params)[0].shape[0]
    ensemble_indices = jnp.arange(ensemble_size, dtype=jnp.int32)
    subset_indices = jax.random.choice(
        key, a=ensemble_indices, shape=(subset_size,), replace=False
    )
    return jax.tree_util.tree_map(lambda x: x[subset_indices], params)


def ensemble_subsample_apply(base_apply, subset_size: int, params, key, *args):
    subset_params = subsample_ensemble_params(params, key, subset_size)
    return ensemble_all_apply(base_apply, subset_params, *args)


@dataclasses.dataclass
class REDQNetworks:
    policy_network: networks_lib.FeedForwardNetwork
    critic_network: networks_lib.FeedForwardNetwork
    num_min_qs: int


def make_networks(
    spec: specs.EnvironmentSpec,
    hidden_sizes: Sequence[int] = (256, 256),
    num_qs: int = 2,
    num_min_qs: Optional[int] = None,
    critic_layer_norm: bool = True,
):
    num_dimensions = np.prod(spec.actions.shape, dtype=int)
    num_min_qs = num_min_qs or num_qs

    # TODO(yl): Support stochastic networks (e.g., dropout)
    def _actor_fn(obs):
        torso = MLP(
            hidden_sizes,
            use_layer_norm=False,
            activate_final=True,
            use_pnorm=False,
        )
        network = hk.Sequential(
            [torso, networks_lib.NormalTanhDistribution(num_dimensions)]
        )
        return network(obs)

    def _critic_fn(obs, action):
        inputs = jnp.concatenate([obs, action], axis=-1)
        network = MLP(
            list(hidden_sizes) + [1],
            use_layer_norm=critic_layer_norm,
            activate_final=False,
            use_pnorm=False,
        )
        return jnp.squeeze(network(inputs), axis=-1)

    policy = hk.without_apply_rng(hk.transform(_actor_fn))
    critic = hk.without_apply_rng(hk.transform(_critic_fn))
    dummy_obs = jax_utils.add_batch_dim(jax_utils.zeros_like(spec.observations))
    dummy_act = jax_utils.add_batch_dim(jax_utils.zeros_like(spec.actions))

    ensemble_critic_init = lambda key: ensemble_init(
        lambda rng: critic.init(rng, dummy_obs, dummy_act), key, num_qs
    )
    ensemble_critic_apply = functools.partial(ensemble_all_apply, critic.apply)

    return REDQNetworks(
        policy_network=networks_lib.FeedForwardNetwork(
            init=lambda key: policy.init(key, dummy_obs), apply=policy.apply
        ),
        critic_network=networks_lib.FeedForwardNetwork(
            init=ensemble_critic_init,
            apply=ensemble_critic_apply,
        ),
        num_min_qs=num_min_qs,
    )


def target_entropy_from_spec(action_spec: specs.BoundedArray) -> float:
    num_dimensions = action_spec.shape[-1]
    return -num_dimensions / 2


def apply_policy_and_sample(networks: REDQNetworks, evaluation: bool):
    def policy_network(params, key, observations):
        dist = networks.policy_network.apply(params, observations)
        if evaluation:
            return dist.mode()
        return dist.sample(seed=key)

    return policy_network
