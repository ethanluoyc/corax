# type: ignore
"""Networks definitions for the CQL agent."""
import dataclasses
from typing import Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

from corax import specs
from corax.agents.jax import actor_core as actor_core_lib
from corax.jax import networks as networks_lib
from corax.jax import utils


@dataclasses.dataclass
class CQLNetworks:
    """Network and pure functions for the CQL agent."""

    policy_network: networks_lib.FeedForwardNetwork
    critic_network: networks_lib.FeedForwardNetwork
    log_prob: networks_lib.LogProbFn
    sample: networks_lib.SampleFn
    sample_eval: networks_lib.SampleFn
    environment_specs: specs.EnvironmentSpec


def apply_and_sample_n(
    key: networks_lib.PRNGKey,
    networks: CQLNetworks,
    params: networks_lib.Params,
    obs: jnp.ndarray,
    num_samples: int,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Applies the policy and samples num_samples actions."""
    dist_params = networks.policy_network.apply(params, obs)
    sampled_actions = jnp.array(
        [
            networks.sample(dist_params, key_n)
            for key_n in jax.random.split(key, num_samples)
        ]
    )
    sampled_log_probs = networks.log_prob(dist_params, sampled_actions)
    return sampled_actions, sampled_log_probs


def make_networks(
    spec: specs.EnvironmentSpec,
    policy_hidden_sizes=(256, 256),
    critic_hidden_sizes=(256, 256, 256),
) -> CQLNetworks:
    num_dimensions = np.prod(spec.actions.shape, dtype=int)

    def _actor_fn(obs):
        network = hk.Sequential(
            [
                hk.nets.MLP(
                    list(policy_hidden_sizes),
                    w_init=hk.initializers.VarianceScaling(1.0, "fan_in", "uniform"),
                    activation=jax.nn.relu,
                    activate_final=True,
                ),
                networks_lib.NormalTanhDistribution(num_dimensions),
            ]
        )
        return network(obs)

    def _critic_fn(obs, action):
        network1 = hk.Sequential(
            [
                hk.nets.MLP(
                    list(critic_hidden_sizes) + [1],
                    w_init=hk.initializers.VarianceScaling(1.0, "fan_in", "uniform"),
                    activation=jax.nn.relu,
                ),
            ]
        )
        network2 = hk.Sequential(
            [
                hk.nets.MLP(
                    list(critic_hidden_sizes) + [1],
                    w_init=hk.initializers.VarianceScaling(1.0, "fan_in", "uniform"),
                    activation=jax.nn.relu,
                ),
            ]
        )
        input_ = jnp.concatenate([obs, action], axis=-1)
        value1 = network1(input_)
        value2 = network2(input_)
        return jnp.concatenate([value1, value2], axis=-1)

    policy = hk.without_apply_rng(hk.transform(_actor_fn))
    critic = hk.without_apply_rng(hk.transform(_critic_fn))

    # Create dummy observations and actions to create network parameters.
    dummy_action = utils.zeros_like(spec.actions)
    dummy_obs = utils.zeros_like(spec.observations)
    dummy_action = utils.add_batch_dim(dummy_action)
    dummy_obs = utils.add_batch_dim(dummy_obs)

    return CQLNetworks(
        policy_network=networks_lib.FeedForwardNetwork(
            lambda key: policy.init(key, dummy_obs), policy.apply
        ),
        critic_network=networks_lib.FeedForwardNetwork(
            lambda key: critic.init(key, dummy_obs, dummy_action), critic.apply
        ),
        log_prob=lambda params, actions: params.log_prob(actions),
        sample=lambda params, key: params.sample(seed=key),
        sample_eval=lambda params, key: params.mode(),
        environment_specs=spec,
    )


def apply_policy_and_sample(
    networks: CQLNetworks, eval_mode: bool = False
) -> actor_core_lib.FeedForwardPolicy:
    """Returns a function that computes actions."""
    sample_fn = networks.sample if not eval_mode else networks.sample_eval
    if not sample_fn:
        raise ValueError("sample function is not provided")

    def apply_and_sample(params, key, obs):
        return sample_fn(networks.policy_network.apply(params, obs), key)

    return apply_and_sample
