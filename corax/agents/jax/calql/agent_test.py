"""Tests for the CQL agent."""

from absl.testing import absltest
import jax
import optax

from corax import specs
from corax.agents.jax.calql import learning
from corax.agents.jax.calql import networks as networks_lib
from corax.testing import fakes


class CQLTest(absltest.TestCase):
    def test_train(self):
        seed = 0
        num_iterations = 3
        batch_size = 10

        # Create a fake environment to test with.
        environment = fakes.ContinuousEnvironment(
            episode_length=10, bounded=True, action_dim=6
        )
        spec = specs.make_environment_spec(environment)

        # Construct the agent.
        networks = networks_lib.make_networks(
            spec,
            policy_hidden_sizes=(8, 8),
            critic_hidden_sizes=(8, 8),
        )
        dataset = fakes.transition_iterator(environment)
        key = jax.random.PRNGKey(seed)
        learner = learning.CalQLLearner(
            batch_size,
            networks,
            key,
            demonstrations=dataset(batch_size),
            policy_optimizer=optax.adam(3e-5),
            critic_optimizer=optax.adam(3e-4),
            fixed_cql_coefficient=5.0,
            cql_lagrange_threshold=None,
            cql_num_samples=2,
            target_entropy=0.1,
            num_bc_iters=2,
            num_sgd_steps_per_step=1,
        )

        # Train the agent
        for _ in range(num_iterations):
            learner.step()


if __name__ == "__main__":
    absltest.main()
