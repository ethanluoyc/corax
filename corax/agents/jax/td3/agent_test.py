"""Tests for the REDQ agent."""

import jax
import optax
from absl.testing import absltest

import corax
from corax.agents.jax import td3
from corax.testing import fakes


class TD3Test(absltest.TestCase):
    def test_learner_core_step(self):
        env = fakes.ContinuousEnvironment(bounded=True)
        env_spec = corax.make_environment_spec(env)
        networks = td3.make_networks(env_spec, hidden_layer_sizes=(8, 8))
        batch_size = 2
        learner_core = td3.TD3LearnerCore(
            networks,
            discount=0.99,
            policy_optimizer=optax.adam(1e-4),
            critic_optimizer=optax.adam(1e-4),
            twin_critic_optimizer=optax.adam(1e-4),
        )
        iterator = fakes.transition_iterator_from_spec(env_spec)(batch_size)
        key = jax.random.PRNGKey(0)
        state = learner_core.init(key)
        state, _ = learner_core.step(state, next(iterator))


if __name__ == "__main__":
    absltest.main()
