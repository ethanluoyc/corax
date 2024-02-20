# type: ignore
"""Tests for ORIL loss functions."""
from absl.testing import absltest
import jax
import jax.numpy as jnp
import tree

from corax import types
from corax.agents.jax.oril import losses


class ORILLossTest(absltest.TestCase):
    def setUp(self):
        def dummy_rewarder_fn(params, obs):
            del params
            return jnp.square(obs)

        self.rewarder_fn = dummy_rewarder_fn

        zero_transition = types.Transition(0.1, 0.0, 0.0, 0.0, 0.0)
        self.unlabeled_transitions = tree.map_structure(
            lambda x: jnp.expand_dims(x, axis=0), zero_transition
        )

        one_transition = types.Transition(1.0, 0.0, 0.0, 0.0, 0.0)
        self.expert_transitions = tree.map_structure(
            lambda x: jnp.expand_dims(x, axis=0), one_transition
        )

    def test_oril_loss(self):
        loss, _ = losses.oril_loss(
            self.rewarder_fn,
            {},
            self.expert_transitions,
            self.unlabeled_transitions,  # type: ignore
        )

        r_e = jax.nn.sigmoid(
            self.rewarder_fn(None, self.expert_transitions.observation)  # type: ignore
        )
        r_u = jax.nn.sigmoid(
            self.rewarder_fn(None, self.unlabeled_transitions.observation)  # type: ignore
        )

        expected_loss = -jnp.log(r_e) - jnp.log(1 - r_u)

        self.assertAlmostEqual(loss, expected_loss, places=6)

    def test_oril_pu_loss(self):
        gamma = 0.7
        loss, _ = losses.oril_pu_loss(
            self.rewarder_fn,
            {},
            self.expert_transitions,
            self.unlabeled_transitions,
            gamma=gamma,
        )

        r_e = jax.nn.sigmoid(
            self.rewarder_fn(None, self.expert_transitions.observation)  # type: ignore
        )
        r_u = jax.nn.sigmoid(
            self.rewarder_fn(None, self.unlabeled_transitions.observation)  # type: ignore
        )

        expected_loss = (
            gamma * (-jnp.log(r_e)) + -jnp.log(1 - r_u) - gamma * -jnp.log(1 - r_e)
        )

        self.assertAlmostEqual(loss, expected_loss, places=6)


if __name__ == "__main__":
    absltest.main()
