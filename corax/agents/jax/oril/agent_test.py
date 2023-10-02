"""Tests for ORIL agent."""
import jax
import optax
from absl.testing import absltest

from corax import specs
from corax.agents.jax import iql
from corax.agents.jax import learners
from corax.agents.jax import oril
from corax.testing import fakes


class ORILLearnerTest(absltest.TestCase):
    """Integration tests for ORIL learner"""

    def test_oril_learner(self):
        # Create a fake environment to test with.
        environment = fakes.ContinuousEnvironment(
            action_dim=2, observation_dim=3, episode_length=10, bounded=True
        )
        environment_spec = specs.make_environment_spec(environment)
        expert_iterator = fakes.transition_iterator(environment)(32)
        unlabeled_iterator = fakes.transition_iterator(environment)(32)
        reward_network = oril.make_reward_network(environment_spec)
        key = jax.random.PRNGKey(0)
        oril_key, learner_key = jax.random.split(key)

        def make_oril_iterator(expert_iterator, unlabeled_iterator, offline_iterator):
            while True:
                # pylint: disable=stop-iteration-return
                expert_sample = next(expert_iterator)
                unlabeled_sample = next(unlabeled_iterator)
                offline_rl_sample = next(offline_iterator)
                yield oril.ORILSample(
                    expert_sample, unlabeled_sample, offline_rl_sample
                )

        def make_offline_learner(iterator):
            iql_networks = iql.make_networks(environment_spec)
            learner_core = iql.IQLLearnerCore(
                iql_networks,
                policy_optimizer=optax.adam(1e-4),
                critic_optimizer=optax.adam(1e-4),
                value_optimizer=optax.adam(1e-4),
            )
            return learners.DefaultJaxLearner(learner_core, learner_key, iterator)

        learner = oril.ORILLearner(
            make_oril_iterator(expert_iterator, unlabeled_iterator, unlabeled_iterator),
            make_offline_learner,
            reward_network,
            loss_fn=oril.oril_loss,
            random_key=oril_key,
            optimizer=optax.adam(1e-4),
        )
        learner.step()


if __name__ == "__main__":
    absltest.main()
