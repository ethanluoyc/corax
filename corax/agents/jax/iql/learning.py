"""Implementations of Implicit Q Learning (IQL) learner component."""
from typing import Any, Dict, Iterator, NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp
import optax

from corax import types as core_types
from corax.agents.jax import learner_core
from corax.agents.jax import learners
from corax.agents.jax.iql import networks as iql_networks
from corax.jax import networks as networks_lib
from corax.jax import types as jax_types
from corax.utils import counting
from corax.utils import loggers

_Metrics = Dict[str, jnp.ndarray]


class TrainingState(NamedTuple):
    policy_params: networks_lib.Params
    policy_opt_state: optax.OptState
    value_params: networks_lib.Params
    value_opt_state: optax.OptState
    critic_params: networks_lib.Params
    critic_opt_state: optax.OptState
    target_critic_params: networks_lib.Params
    key: jax_types.PRNGKey
    steps: int


def expectile_loss(diff, expectile=0.8):
    weight = jnp.where(diff > 0, expectile, (1 - expectile))
    return weight * (diff**2)


class IQLLearnerCore(learner_core.LearnerCore[TrainingState, core_types.Transition]):
    """IQL Learner core."""

    def __init__(
        self,
        networks: iql_networks.IQLNetworks,
        policy_optimizer: optax.GradientTransformation,
        critic_optimizer: optax.GradientTransformation,
        value_optimizer: optax.GradientTransformation,
        discount: float = 0.99,
        tau: float = 0.005,
        expectile: float = 0.8,
        temperature: float = 0.1,
    ):
        """Create an instance of the IQLLearner.

        Args:
            random_key (types.PRNGKey): random seed used by the learner.
            networks (iql_networks.IQLNetworks): networks used by the learner.
            policy_optimizer (optax.GradientTransformation): optimizer for policy.
            critic_optimizer (optax.GradientTransformation): optimizer for critic.
            value_optimizer (optax.GradientTransformation): optimizer for value critic.
            discount (float, optional): additional discount. Defaults to 0.99.
            tau (float, optional): target soft update rate. Defaults to 0.005.
            expectile (float, optional): expectile for training critic. Defaults to 0.8.
            temperature (float, optional): temperature for the AWR. Defaults to 0.1.

        Returns:
            An instance of IQLLearner
        """

        policy_network = networks.policy_network
        value_network = networks.value_network
        critic_network = networks.critic_network

        def awr_actor_loss_fn(
            policy_params: networks_lib.Params,
            key: jax_types.PRNGKey,
            target_critic_params: networks_lib.Params,
            value_params: networks_lib.Params,
            batch: core_types.Transition,
        ) -> Tuple[jnp.ndarray, Any]:
            v = value_network.apply(value_params, batch.observation)
            q1, q2 = critic_network.apply(
                target_critic_params, batch.observation, batch.action
            )
            q = jnp.minimum(q1, q2)
            exp_a = jnp.exp((q - v) * temperature)
            exp_a = jnp.minimum(exp_a, 100.0)
            dist = policy_network.apply(
                policy_params, batch.observation, is_training=True, key=key
            )
            log_probs = dist.log_prob(batch.action)
            actor_loss = -(exp_a * log_probs).mean()

            return actor_loss, {"actor_loss": actor_loss, "advantage": jnp.mean(q - v)}

        def value_loss_fn(
            value_params: networks_lib.Params,
            target_critic_params: networks_lib.Params,
            batch: core_types.Transition,
        ) -> Tuple[jnp.ndarray, Any]:
            q1, q2 = critic_network.apply(
                target_critic_params, batch.observation, batch.action
            )
            q = jnp.minimum(q1, q2)
            v = value_network.apply(value_params, batch.observation)
            value_loss = expectile_loss(q - v, expectile).mean()
            return value_loss, {"value_loss": value_loss, "value": v.mean()}

        def critic_loss_fn(
            critic_params: networks_lib.Params,
            target_value_params: networks_lib.Params,
            batch: core_types.Transition,
        ) -> Tuple[jnp.ndarray, Any]:
            next_v = value_network.apply(target_value_params, batch.next_observation)
            target_q = batch.reward + discount * batch.discount * next_v
            q1, q2 = critic_network.apply(
                critic_params, batch.observation, batch.action
            )
            critic_loss = ((q1 - target_q) ** 2 + (q2 - target_q) ** 2).mean()
            return critic_loss, {
                "critic_loss": critic_loss,
                "q1": q1.mean(),
                "q2": q2.mean(),
            }

        actor_grad_fn = jax.grad(awr_actor_loss_fn, has_aux=True)
        value_grad_fn = jax.grad(value_loss_fn, has_aux=True)
        critic_grad_fn = jax.grad(critic_loss_fn, has_aux=True)

        def update_step(
            state: TrainingState, batch: core_types.Transition
        ) -> Tuple[TrainingState, _Metrics]:
            # Update value network first
            policy_key, key = jax.random.split(state.key)
            value_grads, value_metrics = value_grad_fn(
                state.value_params, state.target_critic_params, batch
            )
            value_updates, value_opt_state = value_optimizer.update(
                value_grads, state.value_opt_state
            )
            value_params = optax.apply_updates(state.value_params, value_updates)
            # Update policy network
            policy_grads, policy_metrics = actor_grad_fn(
                state.policy_params,
                policy_key,
                state.target_critic_params,
                value_params,
                batch,
            )
            policy_updates, policy_opt_state = policy_optimizer.update(
                policy_grads, state.policy_opt_state
            )
            policy_params = optax.apply_updates(state.policy_params, policy_updates)
            # Update critic network
            critic_grads, critic_metrics = critic_grad_fn(
                state.critic_params, value_params, batch
            )
            critic_updates, critic_opt_state = critic_optimizer.update(
                critic_grads, state.critic_opt_state
            )
            critic_params = optax.apply_updates(state.critic_params, critic_updates)

            target_critic_params = optax.incremental_update(
                critic_params, state.target_critic_params, tau
            )
            state = TrainingState(
                policy_params=policy_params,
                policy_opt_state=policy_opt_state,
                critic_params=critic_params,
                critic_opt_state=critic_opt_state,
                value_params=value_params,
                value_opt_state=value_opt_state,
                target_critic_params=target_critic_params,
                key=key,
                steps=state.steps + 1,
            )
            return state, {**critic_metrics, **value_metrics, **policy_metrics}

        self._update_step = jax.jit(update_step)

        def make_initial_state(key):
            policy_key, critic_key, value_key, key = jax.random.split(key, 4)
            policy_params = policy_network.init(policy_key)
            policy_opt_state = policy_optimizer.init(policy_params)
            critic_params = critic_network.init(critic_key)
            critic_opt_state = critic_optimizer.init(critic_params)
            value_params = value_network.init(value_key)
            value_opt_state = value_optimizer.init(value_params)
            state = TrainingState(
                policy_params=policy_params,
                policy_opt_state=policy_opt_state,
                critic_params=critic_params,
                critic_opt_state=critic_opt_state,
                target_critic_params=critic_params,
                value_params=value_params,
                value_opt_state=value_opt_state,
                key=key,
                steps=0,
            )
            return state

        self._init_fn = make_initial_state

    def init(self, key: jax_types.PRNGKey) -> TrainingState:
        return self._init_fn(key)

    def step(self, state: TrainingState, transitions: core_types.Transition):
        # Perform a single learner step
        state, metrics = self._update_step(state, transitions)
        return state, metrics

    def get_variables(self, state, names):
        variables = {
            "policy": state.policy_params,
            "critic": state.critic_params,
        }
        return [variables[name] for name in names]


class IQLLearner(learners.DefaultJaxLearner[TrainingState, core_types.Transition]):
    def __init__(
        self,
        networks: iql_networks.IQLNetworks,
        dataset: Iterator[core_types.Transition],
        random_key: jax_types.PRNGKey,
        policy_optimizer: optax.GradientTransformation,
        critic_optimizer: optax.GradientTransformation,
        value_optimizer: optax.GradientTransformation,
        discount: float = 0.99,
        tau: float = 0.005,
        expectile: float = 0.8,
        temperature: float = 0.1,
        counter: Optional[counting.Counter] = None,
        logger: Optional[loggers.Logger] = None,
    ):
        core = IQLLearnerCore(
            networks=networks,
            policy_optimizer=policy_optimizer,
            critic_optimizer=critic_optimizer,
            value_optimizer=value_optimizer,
            discount=discount,
            tau=tau,
            expectile=expectile,
            temperature=temperature,
        )
        super().__init__(
            core, random_key, iterator=dataset, logger=logger, counter=counter
        )
