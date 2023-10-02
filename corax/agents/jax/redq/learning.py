from typing import Dict, List, NamedTuple, Sequence, Tuple

import jax
import jax.numpy as jnp
import optax

from corax import types
from corax.agents.jax import learner_core
from corax.agents.jax.redq import networks as red_networks
from corax.jax import networks as networks_lib
from corax.jax import types as jax_types

Metrics = Dict[str, jax.Array]


class TrainingState(NamedTuple):
    policy_params: networks_lib.Params
    critic_params: networks_lib.Params
    target_critic_params: networks_lib.Params
    log_temperature_params: networks_lib.Params

    policy_opt_state: optax.OptState
    critic_opt_state: optax.OptState
    temperature_opt_state: optax.OptState

    key: jax_types.PRNGKey


class REDQLearnerCore(learner_core.LearnerCore):
    def __init__(
        self,
        networks: red_networks.REDQNetworks,
        *,
        policy_optimizer: optax.GradientTransformation,
        critic_optimizer: optax.GradientTransformation,
        temperature_optimizer: optax.GradientTransformation,
        target_entropy: float,
        discount: float = 0.99,
        tau: float = 0.005,
        init_temperature: float = 1.0,
        backup_entropy: bool = True,
        utd_ratio: int = 1,
        reward_scale: float = 1.0,
        reward_bias: float = 0.0,
    ) -> None:
        self._networks = networks
        self._policy_optimizer = policy_optimizer
        self._critic_optimizer = critic_optimizer
        self._temperature_optimizer = temperature_optimizer
        self._tau = tau
        self._target_entropy = target_entropy
        self._discount = discount
        self._init_temperature = init_temperature
        self._backup_entropy = backup_entropy
        self._utd_ratio = utd_ratio
        self._reward_scale = reward_scale
        self._reward_bias = reward_bias

        def actor_loss_fn(
            policy_params: networks_lib.Params,
            critic_params: networks_lib.Params,
            log_temperature: networks_lib.Params,
            transitions: types.Transition,
            key: jax_types.PRNGKey,
        ) -> Tuple[jax.Array, Metrics]:
            dist = self._networks.policy_network.apply(
                policy_params, transitions.observation
            )
            actions = dist.sample(seed=key)
            log_probs = dist.log_prob(actions)
            qs = self._networks.critic_network.apply(
                critic_params, transitions.observation, actions
            )
            q = jnp.mean(qs, axis=0)
            temperature = jnp.exp(log_temperature)
            actor_loss = log_probs * temperature - q
            actor_loss = jnp.mean(actor_loss)
            return actor_loss, {"actor_loss": actor_loss, "entropy": -log_probs.mean()}

        def critic_loss_fn(
            critic_params: networks_lib.Params,
            state: TrainingState,
            transitions: types.Transition,
            key: jax_types.PRNGKey,
        ) -> Tuple[jax.Array, Metrics]:
            dist = self._networks.policy_network.apply(
                state.policy_params, transitions.next_observation
            )

            policy_key, critic_key = jax.random.split(key)
            next_actions = dist.sample(seed=policy_key)
            # [M, ...]
            subsampled_target_params = red_networks.subsample_ensemble_params(
                state.target_critic_params, critic_key, self._networks.num_min_qs
            )
            # [M, B]
            next_qs = self._networks.critic_network.apply(
                subsampled_target_params, transitions.next_observation, next_actions
            )
            # [B]
            next_q = next_qs.min(axis=0)
            target_q = (
                transitions.reward + self._discount * transitions.discount * next_q
            )

            if self._backup_entropy:
                next_log_probs = dist.log_prob(next_actions)
                temperature = jnp.exp(state.log_temperature_params)
                target_q = target_q - (
                    self._discount * transitions.discount * temperature * next_log_probs
                )

            qs = self._networks.critic_network.apply(
                critic_params, transitions.observation, transitions.action
            )
            critic_loss = jnp.mean(jnp.square(qs - target_q))
            return critic_loss, {"critic_loss": critic_loss, "q": qs.mean()}

        def temperature_loss_fn(
            log_temperature: jax.Array, entropy: jax.Array
        ) -> Tuple[jax.Array, Metrics]:
            temperature = jnp.exp(log_temperature)
            temp_loss = temperature * jnp.mean(entropy - self._target_entropy)
            return temp_loss, {
                "temperature": temperature,
                "temperature_loss": temp_loss,
            }

        self._actor_loss = actor_loss_fn
        self._critic_loss = critic_loss_fn
        self._temperature_loss = temperature_loss_fn

    def init(self, key: jax_types.PRNGKey) -> TrainingState:
        policy_key, critic_key, train_key = jax.random.split(key, 3)

        policy_params = self._networks.policy_network.init(policy_key)
        critic_params = self._networks.critic_network.init(critic_key)
        temperature_params = jnp.log(self._init_temperature)

        policy_opt_state = self._policy_optimizer.init(policy_params)
        critic_opt_state = self._critic_optimizer.init(critic_params)
        temperature_opt_state = self._temperature_optimizer.init(temperature_params)

        return TrainingState(
            policy_params=policy_params,
            critic_params=critic_params,
            target_critic_params=critic_params,
            log_temperature_params=temperature_params,
            policy_opt_state=policy_opt_state,
            critic_opt_state=critic_opt_state,
            temperature_opt_state=temperature_opt_state,
            key=train_key,
        )

    def _update_critic(
        self, state: TrainingState, transitions: types.Transition
    ) -> Tuple[TrainingState, Metrics]:
        grad_fn = jax.grad(self._critic_loss, has_aux=True)

        key, update_key = jax.random.split(state.key)

        critic_grads, critic_metrics = grad_fn(
            state.critic_params,
            state,
            transitions,
            update_key,
        )

        critic_updates, critic_opt_state = self._critic_optimizer.update(
            critic_grads, state.critic_opt_state
        )

        critic_params = optax.apply_updates(state.critic_params, critic_updates)

        target_critic_params = optax.incremental_update(
            critic_params,
            state.target_critic_params,
            self._tau,
        )

        new_state = state._replace(
            critic_params=critic_params,
            critic_opt_state=critic_opt_state,
            target_critic_params=target_critic_params,
            key=key,
        )

        return (new_state, critic_metrics)

    def _update_actor(
        self, state: TrainingState, transitions: types.Transition
    ) -> Tuple[TrainingState, Metrics]:
        grad_fn = jax.grad(self._actor_loss, has_aux=True)

        key, update_key = jax.random.split(state.key)

        policy_grads, actor_metrics = grad_fn(
            state.policy_params,
            state.critic_params,
            state.log_temperature_params,
            transitions,
            update_key,
        )

        policy_updates, policy_opt_state = self._policy_optimizer.update(
            policy_grads, state.policy_opt_state
        )

        policy_params = optax.apply_updates(state.policy_params, policy_updates)

        new_state = state._replace(
            policy_params=policy_params, policy_opt_state=policy_opt_state, key=key
        )

        return (new_state, actor_metrics)

    def _update_temperature(
        self, state: TrainingState, entropy: jax.Array
    ) -> Tuple[TrainingState, Metrics]:
        grad_fn = jax.grad(self._temperature_loss, has_aux=True)
        grads, metrics = grad_fn(state.log_temperature_params, entropy)

        temperature_updates, opt_state = self._temperature_optimizer.update(
            grads, state.temperature_opt_state
        )

        log_temperature_params = optax.apply_updates(
            state.log_temperature_params, temperature_updates
        )

        new_state = state._replace(
            log_temperature_params=log_temperature_params,
            temperature_opt_state=opt_state,
        )
        return new_state, metrics

    def step(
        self, state: TrainingState, transitions: types.Transition
    ) -> Tuple[TrainingState, Dict[str, jax.Array]]:
        # Maybe rescale reward
        transitions = transitions._replace(
            reward=transitions.reward * self._reward_scale + self._reward_bias
        )

        # We simply concatenate the replay + offline when making the iterator
        # So we need to shuffle the transitions here
        # This could alternatively be done in the iterator
        # Note that shuffling is crucial since otherwise
        # some minibatches below would only contain either replay or offline data.
        key, subkey = jax.random.split(state.key)
        # Reuse the same key to ensure all leaves are shuffled in the same way.
        transitions = jax.tree_util.tree_map(
            lambda a: jax.random.permutation(subkey, a, axis=0, independent=False),
            transitions,
        )
        state = state._replace(key=key)

        batch_size = transitions.reward.shape[0]
        if batch_size % self._utd_ratio != 0:
            raise ValueError("Total batch size to be divisible by utd_ratio")

        num_batches = self._utd_ratio
        minibatches = jax.tree_util.tree_map(
            lambda a: jnp.reshape(a, (num_batches, -1, *a.shape[1:])), transitions
        )

        state, critic_metrics = jax.lax.scan(
            self._update_critic, state, minibatches, length=num_batches
        )
        critic_metrics = jax.tree_map(jnp.mean, critic_metrics)

        # Update policy and temperature with last minibatch
        batch = jax.tree_util.tree_map(lambda a: a[-1], minibatches)

        state, actor_metrics = self._update_actor(state, batch)
        state, temperature_metrics = self._update_temperature(
            state, actor_metrics["entropy"]
        )
        return state, {**critic_metrics, **actor_metrics, **temperature_metrics}

    def get_variables(
        self, state: TrainingState, names: Sequence[str]
    ) -> List[types.NestedArray]:
        variables = {
            "policy": state.policy_params,
            "critic": state.critic_params,
            "target_critic": state.target_critic_params,
        }
        return [variables[name] for name in names]
