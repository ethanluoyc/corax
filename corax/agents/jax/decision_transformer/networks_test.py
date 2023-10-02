# type: ignore
import chex
import haiku as hk
import jax
import jax.numpy as jnp
from absl.testing import absltest

from corax.agents.jax.decision_transformer import networks


class NetworksTest(absltest.TestCase):
    def test_dt_run(self):
        batch_size = 10
        sequence_size = 5
        num_heads = 2
        num_layers = 2
        dropout_rate = 0.0
        state_size = 3
        action_size = 5
        hidden_size = 8

        def forward_fn(data, is_training: bool = True):
            dt = networks.DecisionTransformer(
                state_size=state_size,
                action_size=action_size,
                hidden_size=hidden_size,
                num_heads=num_heads,
                num_layers=num_layers,
                dropout_rate=dropout_rate,
                action_tanh=True,
            )
            output_embeddings = dt(
                data["states"],
                data["actions"],
                data["rewards"],
                data["returns_to_go"],
                data["timesteps"],
                attention_mask=data["mask"],
                is_training=is_training,
            )
            return output_embeddings

        forward = hk.transform(forward_fn)
        examples_input = {
            "states": jnp.zeros((batch_size, sequence_size, state_size)),
            "actions": jnp.zeros((batch_size, sequence_size, action_size)),
            "rewards": jnp.zeros((batch_size, sequence_size, 1)),
            "returns_to_go": jnp.zeros((batch_size, sequence_size, 1)),
            "timesteps": jnp.zeros((batch_size, sequence_size), dtype=jnp.int32),
            "mask": jnp.ones((batch_size, sequence_size), dtype=jnp.bool_),
        }
        key = jax.random.PRNGKey(0)
        key_init, key_apply = jax.random.split(key)
        del key
        params = forward.init(key_init, examples_input, True)
        state_preds, action_preds, return_preds = forward.apply(
            params, key_apply, examples_input, True
        )
        chex.assert_equal_shape((state_preds, examples_input["states"]))
        chex.assert_equal_shape((action_preds, examples_input["actions"]))
        chex.assert_equal_shape((return_preds, examples_input["rewards"]))

    def test_transformer_forward(self):
        batch_size = 10
        sequence_size = 5
        input_size = 4
        num_heads = 2
        num_layers = 2
        dropout_rate = 0.0

        def forward_fn(data, is_training=True):
            tokens = data["inputs"]
            input_embeddings = tokens
            # input_mask = jnp.greater(tokens, 0)
            input_mask = None

            # Run the transformer over the inputs.
            transformer = networks.Transformer(
                num_heads=num_heads, num_layers=num_layers, dropout_rate=dropout_rate
            )
            output_embeddings = transformer(input_embeddings, input_mask, is_training)
            return output_embeddings

        forward = hk.transform(forward_fn)
        examples_input = {"inputs": jnp.zeros((batch_size, sequence_size, input_size))}
        key = jax.random.PRNGKey(0)
        key_init, key_apply = jax.random.split(key)
        del key
        params = forward.init(key_init, examples_input, True)
        output = forward.apply(params, key_apply, examples_input, True)
        self.assertEqual(output.shape, (batch_size, sequence_size, input_size))


if __name__ == "__main__":
    absltest.main()
