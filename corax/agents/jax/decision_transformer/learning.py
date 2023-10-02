import time
from typing import NamedTuple, Sequence

import jax
import jax.numpy as jnp
import optax

import corax
from corax.jax import networks as networks_lib
from corax.utils import counting
from corax.utils import loggers


class TrainState(NamedTuple):
    params: networks_lib.Params
    opt_state: optax.OptState
    key: networks_lib.PRNGKey


class DecisionTransformerLearner(corax.Learner):
    def __init__(self, model, key, dataset, optimizer, logger=None, counter=None):
        def loss_fn(params, key, inputs):
            states = inputs["observation"]
            actions = inputs["action"]
            rewards = None
            # discounts = inputs["discount"]
            rtg = inputs["return_to_go"]
            timesteps = inputs["timestep"]
            attention_mask = inputs["mask"]

            action_target = actions
            _, action_preds, _ = model.apply(
                params,
                key,
                states,
                actions,
                rewards,
                rtg,
                timesteps,
                attention_mask=attention_mask,
            )

            act_dim = action_preds.shape[2]
            action_preds = action_preds.reshape(-1, act_dim)
            target = action_target.reshape(-1, act_dim)
            return jnp.mean(
                ((action_preds - target) ** 2)
                * jnp.expand_dims(attention_mask.reshape(-1) > 0, -1).astype(
                    jnp.float32
                )
            )

        def sgd_step(state: TrainState, inputs):
            key_step, key = jax.random.split(state.key)
            loss_value, grads = jax.value_and_grad(loss_fn)(
                state.params, key_step, inputs
            )
            updates, opt_state = optimizer.update(grads, state.opt_state, state.params)
            params = optax.apply_updates(state.params, updates)
            return TrainState(params, opt_state, key), {"loss": loss_value}

        self._sgd_step = jax.jit(sgd_step)
        self._dataset = dataset

        init_key, key = jax.random.split(key)
        init_params = model.init(init_key)
        init_opt_state = optimizer.init(init_params)

        self._state = TrainState(init_params, init_opt_state, key)
        self._logger = logger or loggers.make_default_logger(
            "learner",
            asynchronous=True,
        )
        self._counter = counter or counting.Counter()
        self._timestamp = None

    def step(self):
        inputs = next(self._dataset)

        self._state, metrics = self._sgd_step(self._state, inputs)

        # Compute elapsed time.
        timestamp = time.time()
        elapsed_time = timestamp - self._timestamp if self._timestamp else 0
        self._timestamp = timestamp

        # Increment counts
        counts = self._counter.increment(steps=1, walltime=elapsed_time)

        self._logger.write({**metrics, **counts})

    def get_variables(self, names: Sequence[str]):
        variables = {"model": self._state.params}
        return [variables[name] for name in names]
