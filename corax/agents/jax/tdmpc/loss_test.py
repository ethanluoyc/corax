import jax
import jax.numpy as jnp
import optax
import reverb
from absl.testing import absltest

from corax import specs
from corax.adders import reverb as adders_reverb
from corax.agents.jax import tdmpc
from corax.agents.jax.tdmpc import learning
from corax.jax import utils
from corax.testing import fakes


class LossTest(absltest.TestCase):
    def test_loss_gradients(self):
        """Test that single pass loss computes gradients correctly."""
        environment = fakes.ContinuousEnvironment(bounded=True)

        environment_spec = specs.make_environment_spec(environment)

        networks = tdmpc.make_networks(
            environment_spec,
            mlp_hidden_size=10,
            latent_size=10,
            encoder_hidden_size=10,
            zero_init=False,
        )

        learner = learning.TDMPCLearner(
            environment_spec,
            networks,
            jax.random.PRNGKey(0),
            iterator=None,  # type: ignore
            replay_client=None,  # type: ignore
            optimizer=optax.adam(1e-3),
        )

        dummy_values = utils.zeros_like(environment_spec)
        horizon = 3

        sequences = adders_reverb.Step(
            observation=utils.add_batch_dim(
                utils.tile_nested(dummy_values.observations, horizon)
            ),
            reward=utils.add_batch_dim(
                utils.tile_nested(dummy_values.rewards, horizon)
            ),
            action=utils.add_batch_dim(
                utils.tile_nested(dummy_values.actions, horizon)
            ),
            discount=jnp.ones((1, horizon)),
            start_of_episode=jnp.ones((1, horizon)),
        )
        sequences = jax.tree_map(lambda x: x + 1, sequences)

        batch = reverb.ReplaySample(
            info=reverb.SampleInfo(
                probability=jnp.ones((1,)),
                key=None,
                table_size=1,
                priority=1,
                times_sampled=1.0,
            ),
            data=sequences,
        )

        params = learner._state.params

        def get_grad(name):
            def loss_fn(params):
                _, (_, logging_dict) = learner._loss_fn(
                    params, params, batch, jax.random.PRNGKey(0)
                )
                return logging_dict[name]

            return jax.grad(loss_fn)(params)

        policy_grads = get_grad("policy_loss")
        self.assertEqual(
            optax.global_norm((policy_grads._replace(policy_params={}))), 0
        )
        model_grads = get_grad("model/model_loss")
        self.assertEqual(optax.global_norm((model_grads.policy_params)), 0.0)


if __name__ == "__main__":
    absltest.main()
