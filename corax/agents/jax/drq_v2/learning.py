"""Learner component for DrQV2."""
import time
from typing import Iterator, NamedTuple, Optional

import jax
import jax.numpy as jnp
import optax
import reverb
import rlax

from corax import core
from corax import types as acme_types
from corax.agents.jax.drq_v2 import augmentations
from corax.agents.jax.drq_v2 import networks as drq_v2_networks
from corax.jax import networks as networks_lib
from corax.jax import types as jax_types
from corax.jax import utils
from corax.utils import counting
from corax.utils import loggers


class TrainingState(NamedTuple):
    """Holds training state for the DrQ learner."""

    policy_params: networks_lib.Params
    policy_opt_state: optax.OptState

    encoder_params: networks_lib.Params
    # There is not target encoder parameters in v2.
    encoder_opt_state: optax.OptState

    critic_params: networks_lib.Params
    critic_target_params: networks_lib.Params
    critic_opt_state: optax.OptState

    key: jax_types.PRNGKey
    steps: int


class DrQV2Learner(core.Learner):
    """Learner for DrQ-v2"""

    def __init__(
        self,
        random_key: jax_types.PRNGKey,
        dataset: Iterator[reverb.ReplaySample],
        networks: drq_v2_networks.DrQV2Networks,
        sigma_schedule: optax.Schedule,
        augmentation: augmentations.DataAugmentation,
        policy_optimizer: optax.GradientTransformation,
        critic_optimizer: optax.GradientTransformation,
        encoder_optimizer: optax.GradientTransformation,
        noise_clip: float = 0.3,
        critic_soft_update_rate: float = 0.005,
        discount: float = 0.99,
        num_sgd_steps_per_step: int = 1,
        clipping: bool = False,
        bc_alpha: Optional[float] = None,
        counter: Optional[counting.Counter] = None,
        logger: Optional[loggers.Logger] = None,
    ):
        def critic_loss_fn(
            critic_params: networks_lib.Params,
            encoder_params: networks_lib.Params,
            critic_target_params: networks_lib.Params,
            policy_params: networks_lib.Params,
            transitions: acme_types.Transition,
            key: jax_types.PRNGKey,
            sigma: jnp.ndarray,
        ):
            next_encoded = networks.encoder_network.apply(
                encoder_params, transitions.next_observation
            )
            next_action = networks.policy_network.apply(policy_params, next_encoded)
            next_action = networks.add_policy_noise(next_action, key, sigma, noise_clip)  # type: ignore
            next_q1, next_q2 = networks.critic_network.apply(
                critic_target_params, next_encoded, next_action
            )
            # Calculate q target values
            next_q = jnp.minimum(next_q1, next_q2)
            target_q = transitions.reward + transitions.discount * discount * next_q
            target_q = jax.lax.stop_gradient(target_q)
            # Calculate predicted Q
            encoded = networks.encoder_network.apply(
                encoder_params, transitions.observation
            )
            q1, q2 = networks.critic_network.apply(
                critic_params, encoded, transitions.action
            )
            loss_critic = (jnp.square(target_q - q1) + jnp.square(target_q - q2)).mean(
                axis=0
            )
            return loss_critic, {"q1": q1.mean(), "q2": q2.mean()}

        def critic_fn(params, h, a):
            q1, q2 = networks.critic_network.apply(params, h, a)
            q = jnp.minimum(q1, q2)
            return q

        def policy_loss_fn(
            policy_params: networks_lib.Params,
            critic_params: networks_lib.Params,
            encoder_params: networks_lib.Params,
            transition: acme_types.Transition,
            sigma: jnp.ndarray,
            key,
        ):
            encoded = networks.encoder_network.apply(
                encoder_params, transition.observation
            )
            action = networks.policy_network.apply(policy_params, encoded)
            action = networks.add_policy_noise(action, key, sigma, noise_clip)  # type: ignore
            if bc_alpha is None:
                grad_critic = jax.vmap(
                    jax.grad(critic_fn, argnums=2), in_axes=(None, 0, 0)
                )
                dq_da = grad_critic(critic_params, encoded, action)
                dqda_clipping = 1.0 if clipping else None
                batch_dpg_learning = jax.vmap(rlax.dpg_loss, in_axes=(0, 0, None))
                loss = batch_dpg_learning(action, dq_da, dqda_clipping)
                return jnp.mean(loss), {}
            else:
                q = critic_fn(critic_params, encoded, action)
                lmbda = jax.lax.stop_gradient(bc_alpha / jnp.mean(jnp.abs(q)))
                loss = -lmbda * jnp.mean(q) + jnp.mean(
                    jnp.square(action - transition.action)
                )
                return loss, {}

        policy_grad_fn = jax.value_and_grad(policy_loss_fn, has_aux=True)
        critic_grad_fn = jax.value_and_grad(
            critic_loss_fn, argnums=(0, 1), has_aux=True
        )

        def update_step(
            state: TrainingState,
            transitions: acme_types.Transition,
        ):
            key_aug1, key_aug2, key_policy, key_critic, key = jax.random.split(
                state.key, 5
            )
            sigma = sigma_schedule(state.steps)
            # Perform data augmentation on o_tm1 and o_t
            observation_aug = augmentation(key_aug1, transitions.observation)
            next_observation_aug = augmentation(key_aug2, transitions.next_observation)
            transitions = transitions._replace(
                observation=observation_aug,
                next_observation=next_observation_aug,
            )
            # Update critic
            (critic_loss, critic_aux), (critic_grad, encoder_grad) = critic_grad_fn(
                state.critic_params,
                state.encoder_params,
                state.critic_target_params,
                state.policy_params,
                transitions,
                key_critic,
                sigma,
            )
            (policy_loss, policy_aux), actor_grad = policy_grad_fn(
                state.policy_params,
                state.critic_params,
                state.encoder_params,
                transitions,
                sigma,
                key_policy,
            )
            encoder_update, encoder_opt_state = encoder_optimizer.update(
                encoder_grad, state.encoder_opt_state
            )
            critic_update, critic_opt_state = critic_optimizer.update(
                critic_grad, state.critic_opt_state
            )
            encoder_params = optax.apply_updates(state.encoder_params, encoder_update)
            critic_params = optax.apply_updates(state.critic_params, critic_update)
            # Update policy
            policy_update, policy_opt_state = policy_optimizer.update(
                actor_grad, state.policy_opt_state
            )
            policy_params = optax.apply_updates(state.policy_params, policy_update)

            critic_target_params = optax.incremental_update(
                critic_params, state.critic_target_params, critic_soft_update_rate
            )
            metrics = {
                "policy_loss": policy_loss,
                "critic_loss": critic_loss,
                "sigma": sigma,
                **critic_aux,
                **policy_aux,
            }
            new_state = TrainingState(
                policy_params=policy_params,
                policy_opt_state=policy_opt_state,
                encoder_params=encoder_params,
                encoder_opt_state=encoder_opt_state,
                critic_params=critic_params,
                critic_target_params=critic_target_params,
                critic_opt_state=critic_opt_state,
                key=key,
                steps=state.steps + 1,
            )
            return new_state, metrics

        self._iterator = dataset
        self._counter = counter or counting.Counter()
        self._logger = logger or loggers.make_default_logger(
            label="learner",
            asynchronous=True,
            serialize_fn=utils.fetch_devicearray,
        )
        self._update_step = utils.process_multiple_batches(
            update_step, num_sgd_steps_per_step
        )
        self._update_step = jax.jit(self._update_step)

        # Initialize training state
        def make_initial_state(key: jax_types.PRNGKey):
            key_encoder, key_critic, key_policy, key = jax.random.split(key, 4)
            encoder_init_params = networks.encoder_network.init(key_encoder)
            encoder_init_opt_state = encoder_optimizer.init(encoder_init_params)

            critic_init_params = networks.critic_network.init(key_critic)
            critic_init_opt_state = critic_optimizer.init(critic_init_params)

            policy_init_params = networks.policy_network.init(key_policy)
            policy_init_opt_state = policy_optimizer.init(policy_init_params)

            return TrainingState(
                policy_params=policy_init_params,
                policy_opt_state=policy_init_opt_state,
                encoder_params=encoder_init_params,
                critic_params=critic_init_params,
                critic_target_params=critic_init_params,
                encoder_opt_state=encoder_init_opt_state,
                critic_opt_state=critic_init_opt_state,
                key=key,
                steps=0,
            )

        # Create initial state.
        self._state = make_initial_state(random_key)

        # Do not record timestamps until after the first learning step is done.
        # This is to avoid including the time it takes for actors to come online and
        # fill the replay buffer.
        self._timestamp = None

    def step(self):
        # Get the next batch from the replay iterator
        sample = next(self._iterator)
        transitions = acme_types.Transition(*sample.data)

        # Perform a single learner step
        self._state, metrics = self._update_step(self._state, transitions)

        # Compute elapsed time
        timestamp = time.time()
        elapsed_time = timestamp - self._timestamp if self._timestamp else 0
        self._timestamp = timestamp

        # Increment counts and record the current time
        counts = self._counter.increment(steps=1, walltime=elapsed_time)

        # Attempts to write the logs.
        self._logger.write({**metrics, **counts})

    def get_variables(self, names):
        variables = {
            "policy": {
                "encoder": self._state.encoder_params,
                "policy": self._state.policy_params,
            },
        }
        return [variables[name] for name in names]

    def save(self) -> TrainingState:
        return self._state

    def restore(self, state: TrainingState) -> None:
        self._state = state
