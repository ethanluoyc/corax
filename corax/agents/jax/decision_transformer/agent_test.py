from absl.testing import absltest
import jax
import optax

from corax import specs
from corax.agents.jax import decision_transformer
from corax.agents.jax.decision_transformer import dataset as dataset_lib
from corax.testing import fakes


class RLDSDataset(absltest.TestCase):
    def test_dataset_generator(self):
        scale = 1.0
        env = fakes.ContinuousEnvironment(action_dim=2, observation_dim=2, bounded=True)
        env_specs = specs.make_environment_spec(env)

        episode_dataset = fakes.rlds_dataset_from_env_spec(env_specs)
        observation_mean_std = dataset_lib.get_observation_mean_std(episode_dataset)
        dataset = (
            dataset_lib.transform_decision_transformer_input(
                episode_dataset,
                sequence_length=10,
                scale=scale,
                observation_mean_std=observation_mean_std,
            )
            .repeat()
            .batch(10)
        )

        networks = decision_transformer.make_gym_networks(
            env_specs, 32, 2, 2, 0.0, 1000
        )

        optimizer = optax.adam(1e-3)
        key = jax.random.PRNGKey(0)
        learner = decision_transformer.DecisionTransformerLearner(
            model=networks,
            key=key,
            dataset=dataset.as_numpy_iterator(),
            optimizer=optimizer,
        )
        learner.step()


if __name__ == "__main__":
    absltest.main()
