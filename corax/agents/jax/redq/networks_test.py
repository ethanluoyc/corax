"""Tests for the REDQ agent."""

from absl.testing import absltest
import chex
import jax

import corax
from corax.agents.jax.redq import networks as red_networks
from corax.jax import utils as jax_utils
from corax.testing import fakes


class NetworkTest(absltest.TestCase):
    def test_network(self):
        env = fakes.ContinuousEnvironment(bounded=True)
        env_spec = corax.make_environment_spec(env)
        networks = red_networks.make_networks(
            env_spec, hidden_sizes=(8, 8), num_qs=5, num_min_qs=2
        )

        key = jax.random.PRNGKey(0)
        dummy_obs = jax_utils.tile_nested(
            jax_utils.zeros_like(env_spec.observations), 2
        )
        dummy_act = jax_utils.tile_nested(jax_utils.zeros_like(env_spec.actions), 2)

        critic_params = networks.critic_network.init(key)

        qs = networks.critic_network.apply(critic_params, dummy_obs, dummy_act)
        chex.assert_shape(qs, (5, 2))

        subset_params = red_networks.subsample_ensemble_params(critic_params, key, 2)
        subset_qs = networks.critic_network.apply(subset_params, dummy_obs, dummy_act)

        chex.assert_shape(subset_qs, (2, 2))


if __name__ == "__main__":
    absltest.main()
