# type: ignore
"""Network definitions for DrQ-v2."""
import dataclasses
from typing import Callable

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as onp

from corax import specs
from corax import types
from corax.agents.jax import actor_core
from corax.jax import networks as networks_lib
from corax.jax import utils


class DrQTorso(hk.Module):
    """DrQ Torso used in DrQ-v2."""

    def __init__(self, name: str = "drq_torso"):
        super().__init__(name)
        # pylint: disable=use-dict-literal
        conv_kwargs = dict(
            kernel_shape=(3, 3),
            output_channels=32,
            padding="VALID",
            # This follows from the reference implementation, the scale accounts for
            # using the ReLU activation.
            w_init=hk.initializers.Orthogonal(2**0.5),
        )
        self._network = hk.Sequential(
            [
                hk.Conv2D(stride=2, **conv_kwargs),
                jax.nn.relu,
                hk.Conv2D(stride=1, **conv_kwargs),
                jax.nn.relu,
                hk.Conv2D(stride=1, **conv_kwargs),
                jax.nn.relu,
                hk.Conv2D(stride=1, **conv_kwargs),
                jax.nn.relu,
                hk.Flatten(),
            ]
        )

    def __call__(self, inputs: jnp.ndarray):
        if not jnp.issubdtype(inputs.dtype, jnp.uint8):
            raise ValueError("Expect inputs to be uint8 pixel values between 0 to 255.")
        if inputs.ndim != 4:
            raise ValueError(
                "Input array should have 4 dimensions (for "
                "batch size, height, width, and channels), but it has "
                f"{inputs.ndim}"
            )
        # Floatify the image.
        preprocessed_inputs = inputs.astype(jnp.float32) / 255.0 - 0.5
        return self._network(preprocessed_inputs)


@dataclasses.dataclass
class DrQV2Networks:
    encoder_network: networks_lib.FeedForwardNetwork
    policy_network: networks_lib.FeedForwardNetwork
    critic_network: networks_lib.FeedForwardNetwork
    add_policy_noise: Callable[
        [types.NestedArray, networks_lib.PRNGKey, float, float], types.NestedArray
    ]
    get_policy_feature: Callable


def apply_policy_and_sample(
    networks: DrQV2Networks, action_specs: specs.BoundedArray, sigma: float
) -> actor_core.FeedForwardPolicy:
    """Create a pure policy function for actor_core."""

    def policy(params, key, obs):
        feature_map = networks.encoder_network.apply(params["encoder"], obs)
        action = networks.policy_network.apply(params["policy"], feature_map)
        noise = jax.random.normal(key, shape=action.shape) * sigma
        return jnp.clip(action + noise, action_specs.minimum, action_specs.maximum)

    return policy


def make_networks(
    spec: specs.EnvironmentSpec,
    hidden_size: int = 1024,
    latent_size: int = 50,
) -> DrQV2Networks:
    """Create networks for the DrQ-v2 agent."""
    action_size = onp.prod(spec.actions.shape, dtype=int)

    def _encoder_fn(obs):
        return DrQTorso()(obs)

    def _trunk():
        w_init = hk.initializers.Orthogonal(1.0)
        return hk.Sequential(
            [
                hk.Linear(latent_size, w_init=w_init),
                hk.LayerNorm(axis=-1, create_scale=True, create_offset=True),
                jnp.tanh,
            ]
        )

    def _critic_fn(obs, action):
        w_init = hk.initializers.Orthogonal(1.0)
        embed = _trunk()(obs)
        # NOTE: Actions are concatenated after the trunk
        embed = jnp.concatenate([embed, action], axis=-1)
        critic_network1 = hk.nets.MLP(
            output_sizes=(hidden_size, hidden_size, 1),
            w_init=w_init,
            activate_final=False,
        )
        critic_network2 = hk.nets.MLP(
            output_sizes=(hidden_size, hidden_size, 1),
            w_init=w_init,
            activate_final=False,
        )
        q1 = critic_network1(embed)
        q2 = critic_network2(embed)
        q1 = jnp.squeeze(q1, axis=-1)
        q2 = jnp.squeeze(q2, axis=-1)
        return q1, q2

    def _policy_fn(obs):
        embed = _trunk()(obs)
        head = hk.Sequential(
            [
                hk.nets.MLP(
                    output_sizes=(hidden_size, hidden_size),
                    w_init=hk.initializers.Orthogonal(1.0),
                    activate_final=True,
                ),
                hk.Linear(action_size, w_init=hk.initializers.Orthogonal(1.0)),
                jnp.tanh,
            ]
        )
        return head(embed)

    def _policy_features_fn(obs):
        return _trunk()(obs)

    def add_policy_noise(
        action: types.NestedArray,
        key: networks_lib.PRNGKey,
        sigma: float,
        noise_clip: float,
    ) -> types.NestedArray:
        """Adds action noise to bootstrapped Q-value estimate in critic loss."""
        noise = jax.random.normal(key=key, shape=action.shape) * sigma
        noise = jnp.clip(noise, -noise_clip, noise_clip)
        action = action + noise
        clipped_action = jnp.clip(action, spec.actions.minimum, spec.actions.maximum)
        return (
            action
            - jax.lax.stop_gradient(action)
            + jax.lax.stop_gradient(clipped_action)
        )

    transform_without_rng = lambda f: hk.without_apply_rng(hk.transform(f))

    policy = transform_without_rng(_policy_fn)
    critic = transform_without_rng(_critic_fn)
    encoder = transform_without_rng(_encoder_fn)
    policy_feature = transform_without_rng(_policy_features_fn)

    dummy_action = utils.add_batch_dim(utils.zeros_like(spec.actions))
    dummy_obs = utils.add_batch_dim(utils.zeros_like(spec.observations))
    dummy_encoded = jax.eval_shape(encoder.init, jax.random.PRNGKey(0), dummy_obs)
    dummy_encoded = jax.eval_shape(encoder.apply, dummy_encoded, dummy_obs)
    dummy_encoded = jnp.zeros(shape=dummy_encoded.shape, dtype=dummy_encoded.dtype)

    return DrQV2Networks(
        encoder_network=networks_lib.FeedForwardNetwork(
            lambda key: encoder.init(key, dummy_obs), encoder.apply
        ),
        policy_network=networks_lib.FeedForwardNetwork(
            lambda key: policy.init(key, dummy_encoded), policy.apply
        ),
        critic_network=networks_lib.FeedForwardNetwork(
            lambda key: critic.init(key, dummy_encoded, dummy_action), critic.apply
        ),
        add_policy_noise=add_policy_noise,
        get_policy_feature=policy_feature.apply,
    )


def make_state_networks(spec: specs.EnvironmentSpec):
    """Create networks for the DrQ-v2 agent."""
    action_size = onp.prod(spec.actions.shape, dtype=int)

    def _encoder_fn(obs):
        return obs

    def _critic_fn(obs, action):
        inputs = jnp.concatenate([obs, action], axis=-1)
        critic_network1 = networks_lib.LayerNormMLP([256, 256] + [1])
        critic_network2 = networks_lib.LayerNormMLP([256, 256] + [1])
        q1 = critic_network1(inputs)
        q2 = critic_network2(inputs)
        q1 = jnp.squeeze(q1, axis=-1)
        q2 = jnp.squeeze(q2, axis=-1)
        return q1, q2

    def _policy_fn(obs):
        network = hk.Sequential(
            [
                networks_lib.LayerNormMLP([256, 256], activate_final=True),
                networks_lib.NearZeroInitializedLinear(action_size),
                networks_lib.TanhToSpec(spec.actions),
            ]
        )
        return network(obs)

    def _policy_features_fn(obs):
        return obs

    def add_policy_noise(
        action: types.NestedArray,
        key: networks_lib.PRNGKey,
        sigma: float,
        noise_clip: float,
    ) -> types.NestedArray:
        """Adds action noise to bootstrapped Q-value estimate in critic loss."""
        noise = jax.random.normal(key=key, shape=action.shape) * sigma
        noise = jnp.clip(noise, -noise_clip, noise_clip)
        action = action + noise
        clipped_action = jnp.clip(action, spec.actions.minimum, spec.actions.maximum)
        return (
            action
            - jax.lax.stop_gradient(action)
            + jax.lax.stop_gradient(clipped_action)
        )

    transform_without_rng = lambda f: hk.without_apply_rng(hk.transform(f))

    policy = transform_without_rng(_policy_fn)
    critic = transform_without_rng(_critic_fn)
    encoder = transform_without_rng(_encoder_fn)
    policy_feature = transform_without_rng(_policy_features_fn)

    dummy_action = utils.add_batch_dim(utils.zeros_like(spec.actions))
    dummy_obs = utils.add_batch_dim(utils.zeros_like(spec.observations))
    dummy_encoded = jax.eval_shape(encoder.init, jax.random.PRNGKey(0), dummy_obs)
    dummy_encoded = jax.eval_shape(encoder.apply, dummy_encoded, dummy_obs)
    dummy_encoded = jnp.zeros(shape=dummy_encoded.shape, dtype=dummy_encoded.dtype)

    return DrQV2Networks(
        encoder_network=networks_lib.FeedForwardNetwork(
            lambda key: encoder.init(key, dummy_obs), encoder.apply
        ),
        policy_network=networks_lib.FeedForwardNetwork(
            lambda key: policy.init(key, dummy_encoded), policy.apply
        ),
        critic_network=networks_lib.FeedForwardNetwork(
            lambda key: critic.init(key, dummy_encoded, dummy_action), critic.apply
        ),
        add_policy_noise=add_policy_noise,
        get_policy_feature=policy_feature.apply,
    )
