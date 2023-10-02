# type: ignore
from typing import Any, Callable, NamedTuple, Optional

import dm_env
import jax
import jax.numpy as jnp
import tree

from corax import specs
from corax.jax import networks as networks_lib
from corax.jax import running_statistics
from corax.jax import utils
from corax.jax import variable_utils


def _append(context, update):
    # Alternatively, this works too.
    # return jnp.concatenate([context[1:], jnp.expand_dims(update, 0)], axis=0)
    new_context = jnp.roll(context, shift=-1, axis=0)
    new_context = new_context.at[-1].set(update)
    return new_context


class ActorState(NamedTuple):
    observations: jnp.ndarray
    actions: jnp.ndarray
    rewards: jnp.ndarray
    returns_to_go: jnp.ndarray
    timesteps: jnp.ndarray
    masks: jnp.ndarray

    running_rtg: jnp.ndarray
    running_time: jnp.ndarray


def initial_state(
    environment_spec: specs.EnvironmentSpec,
    target_return: float,
    context_size: int,
) -> ActorState:
    zero_values = utils.tile_nested(utils.zeros_like(environment_spec), context_size)
    return ActorState(
        observations=zero_values.observations,
        actions=zero_values.actions,
        rewards=zero_values.rewards,
        returns_to_go=zero_values.rewards,
        timesteps=jnp.zeros((context_size,), dtype=jnp.int32),
        masks=jnp.zeros((context_size,), dtype=jnp.bool_),
        running_rtg=target_return,
        running_time=jnp.zeros((), dtype=jnp.int32),
    )


class DecisionTransformerActor:
    def __init__(
        self,
        spec: specs.EnvironmentSpec,
        random_key: networks_lib.PRNGKey,
        forward_fn: Callable[[Any], Any],
        target_return: float,
        context_length: int,
        return_scale: float,
        variable_client: variable_utils.VariableClient,
        observation_mean_std: Optional[running_statistics.NestedMeanStd] = None,
        mode: str = "normal",
    ):
        self._key = random_key
        self._max_length = context_length
        self._forward_fn = forward_fn
        self._variable_client = variable_client
        self._target_return = target_return
        self._state = None

        def init(target_return) -> ActorState:
            return initial_state(spec, target_return / return_scale, context_length)

        def select_action(
            params: networks_lib.Params,
            state: ActorState,
            key: jnp.ndarray,
            observation: networks_lib.Observation,
        ):
            if observation_mean_std is not None:
                observation = running_statistics.normalize(
                    observation, observation_mean_std
                )
            new_state = state._replace(
                observations=_append(state.observations, observation),
                timesteps=_append(state.timesteps, state.running_time),
                returns_to_go=_append(state.returns_to_go, state.running_rtg),
                rewards=_append(state.actions, utils.zeros_like(spec.rewards)),
                actions=_append(state.actions, utils.zeros_like(spec.actions)),
                masks=_append(state.masks, True),
            )
            # Run dt_policy
            inputs = {
                "states": new_state.observations,
                "actions": new_state.actions,
                "timesteps": new_state.timesteps,
                "returns_to_go": jnp.expand_dims(new_state.returns_to_go, axis=-1),
                "attention_mask": new_state.masks,
            }
            inputs = tree.map_structure(lambda x: jnp.expand_dims(x, axis=0), inputs)
            key_forward, key = jax.random.split(key)
            _, action_preds, _ = forward_fn(
                params,
                key_forward,
                inputs["states"],
                inputs["actions"],
                None,
                inputs["returns_to_go"],
                inputs["timesteps"],
                attention_mask=inputs["attention_mask"],
                is_training=False,
            )
            return action_preds[0, -1], (new_state, key)

        def observe(state: ActorState, action: networks_lib.Action, reward: float):
            if mode != "delayed":
                running_rtg = state.running_rtg - reward / return_scale
            else:
                running_rtg = state.running_rtg
            return state._replace(
                actions=state.actions.at[-1].set(action),
                rewards=state.rewards.at[-1].set(reward),
                running_time=state.running_time + 1,
                running_rtg=running_rtg,
            )

        self._init = jax.jit(init)
        self._select_action = jax.jit(select_action)
        self._observe = jax.jit(observe)

    @property
    def _params(self):
        return self._variable_client.params

    def select_action(self, observation: networks_lib.Observation):
        action, (self._state, self._key) = self._select_action(
            self._params, self._state, self._key, observation
        )
        action = utils.to_numpy(action)
        return action

    def observe_first(self, timestep: dm_env.TimeStep):
        del timestep
        self._state = self._init(self._target_return)

    def observe(self, action: networks_lib.Action, next_timestep: dm_env.TimeStep):
        self._state = self._observe(self._state, action, next_timestep.reward)

    def update(self, wait: bool = False):
        self._variable_client.update(wait)
