# type: ignore
import time
from typing import Iterator, List, NamedTuple, Optional, Tuple

import chex
import haiku as hk
import jax
import jax.numpy as jnp
import optax
import reverb
import rlax
import tree

import corax
from corax import specs
from corax import types
from corax.adders import reverb as adders_reverb
from corax.agents.jax.tdmpc import networks as tdmpc_networks
from corax.agents.jax.tdmpc import types as tdmpc_types
from corax.jax import networks as networks_lib
from corax.jax import utils as jax_utils
from corax.utils import async_utils
from corax.utils import counting
from corax.utils import loggers

TDMPCReplaySample = jax_utils.PrefetchingSplit


def _squared_error(predictions: chex.Array, targets: chex.Array) -> chex.Array:
    chex.assert_type([predictions], float)
    chex.assert_equal_shape((predictions, targets))
    errors = predictions - targets
    return jnp.square(errors)


def _l1_loss(predictions: chex.Array, targets: chex.Array) -> chex.Array:
    chex.assert_type([predictions], float)
    chex.assert_equal_shape((predictions, targets))
    errors = predictions - targets
    return jnp.abs(errors)


class TrainingState(NamedTuple):
    """Contains training state for the learner."""

    params: tdmpc_networks.TDMPCParams
    target_params: tdmpc_networks.TDMPCParams
    opt_state: optax.OptState
    steps: int
    key: jax.random.PRNGKeyArray


class ModelOutputs(NamedTuple):
    """Contains model outputs."""

    z_prediction: types.NestedArray
    z_prior: types.NestedArray
    reward_pred: types.NestedArray
    q_value: types.NestedArray


class LossTargets(NamedTuple):
    """Contains loss targets."""

    td_target: types.NestedArray
    z_target: types.NestedArray
    reward: types.NestedArray


class TDMPCLearner(corax.Learner):
    """TDMPC Learner."""

    _state: TrainingState

    def __init__(
        self,
        spec: specs.EnvironmentSpec,
        networks: tdmpc_networks.TDMPCNetworks,
        random_key: jax.random.PRNGKeyArray,
        replay_client: reverb.Client,
        iterator: Iterator[TDMPCReplaySample],
        *,
        optimizer: optax.GradientTransformation,
        discount: float = 0.99,
        min_std: float = 0.05,
        loss_scale: Optional[tdmpc_types.LossScalesConfig] = None,
        value_tx_pair: rlax.TxPair = rlax.IDENTITY_PAIR,
        importance_sampling_exponent: float = 0.4,
        target_update_rate: float = 0.01,
        rho: float = 0.5,
        logger: Optional[loggers.Logger] = None,
        counter: Optional[counting.Counter] = None,
    ):
        """Initializes the learner.

        Args:
            spec: A description of the actions, observations, etc.
            networks: networks used by the learner.
            random_key: A random key for initializing the learner.
            replay_client: A Reverb client for updating priorities.
            iterator: A dataset iterator.
            optimizer: The optimizer to use for training.
            discount: Discount to use for TD updates.
            min_std: Minimum standard deviation to use for the policy.
            loss_scale: Loss scales to use for the TDMPC losses.
            importance_sampling_exponent: Exponent to use for importance sampling.
            target_update_rate: Rate at which to update the target networks.
            rho: The rho parameter for the TDMPC loss.
            logger: Logger object for writing training statistics to.
            counter: Counter object for keeping track of iterations.
        """
        if loss_scale is None:
            loss_scale = tdmpc_types.LossScalesConfig()
        self._discount = discount
        self._min_std = min_std
        self._target_update_rate = target_update_rate
        self._loss_scale = loss_scale
        self._rho = rho
        self._networks = networks
        self._optimizer = optimizer
        self._importance_sampling_exponent = importance_sampling_exponent
        self._replay_client = replay_client
        self._tx_pair = value_tx_pair

        # Initialize parameters.
        param_key, key = jax.random.split(random_key)
        params = tdmpc_networks.init_params(self._networks, spec, param_key)

        # Initialize optimizer state.
        opt_state = self._optimizer.init(params)

        # Initialize learner state.
        self._state = TrainingState(
            params=params,
            target_params=params,
            opt_state=opt_state,
            key=key,
            steps=0,
        )

        self._update_step = jax.jit(self._sgd_step)

        self._counter = counter or counting.Counter()
        self._logger = logger

        def update_priorities(keys_and_priorities: Tuple[jnp.ndarray, jnp.ndarray]):
            keys, priorities = keys_and_priorities
            keys, priorities = tree.map_structure(
                # Fetch array and combine device and batch dimensions.
                lambda x: jax_utils.fetch_devicearray(x).reshape((-1,) + x.shape[2:]),
                (keys, priorities),
            )
            replay_client.mutate_priorities(
                table=adders_reverb.DEFAULT_PRIORITY_TABLE,
                updates=dict(zip(keys, priorities)),
            )

        self._async_priority_updater = async_utils.AsyncExecutor(update_priorities)

        self._iterator = iterator

        # Do not record timestamps until after the first learning step is done.
        # This is to avoid including the time it takes for actors to come online and
        # fill the replay buffer.
        self._timestamp = None

    def _compute_targets(
        self,
        target_params: tdmpc_networks.TDMPCParams,
        sequences: adders_reverb.Step,
        online_params: adders_reverb.Step,
        z_prior: types.NestedArray,
        key: networks_lib.PRNGKey,
    ):
        """Computes the targets for the TDMPC loss."""

        def policy(params, obs, key):
            return self._networks.pi(params, obs, self._min_std, key)

        batched_policy = jax.vmap(policy, in_axes=(None, 0, 0))
        batched_critic = jax.vmap(self._networks.q, (None, 0, 0))
        batched_encoder = jax.vmap(self._networks.h, (None, 0))

        z_target = batched_encoder(target_params, sequences.observation)

        # NOTE: Actions for policy evaluation is conditioned on
        # actions selected by the online policy based on prior encoding, See
        # https://github.com/nicklashansen/tdmpc/blob/main/src/algorithm/tdmpc.py#L172-L177
        policy_action = batched_policy(
            online_params, z_prior, jax.random.split(key, z_prior.shape[0])
        )
        policy_action = jax.lax.stop_gradient(policy_action)

        v1_target, v2_target = batched_critic(
            target_params, z_prior[1:], policy_action[1:]
        )

        # Transform to raw reward space.
        v1_target, v2_target = jax.tree_map(
            self._tx_pair.apply_inv, (v1_target, v2_target)
        )
        target_values = jnp.squeeze(jnp.minimum(v1_target, v2_target), axis=-1)

        discounts = self._discount * sequences.discount[:-1]
        td_target = sequences.reward[:-1] + discounts * target_values
        reward_target = sequences.reward[:-1]

        # Transform back to canonical space.
        td_target = jax.lax.stop_gradient(self._tx_pair.apply(td_target))
        reward_target = jax.lax.stop_gradient(self._tx_pair.apply(reward_target))

        return LossTargets(
            td_target=td_target,
            z_target=z_target,
            reward=reward_target,
        )

    def _compute_predictions(
        self,
        params: tdmpc_networks.TDMPCNetworks,
        sequences: adders_reverb.Step,
    ):
        """Computes the predictions from the model."""

        def unroll_dynamics(params, actions, z):
            def _next_core(action, z):
                next_z, reward = self._networks.next(params, z, action)
                return (next_z, reward), next_z

            (online_z_posterior, reward_pred), _ = hk.static_unroll(
                _next_core, actions, z
            )

            z_predictions = jnp.concatenate(
                [jnp.expand_dims(z, axis=0), online_z_posterior], axis=0
            )

            return z_predictions, jnp.squeeze(reward_pred, axis=-1)

        batched_critic = jax.vmap(self._networks.q, (None, 0, 0))
        batched_encoder = jax.vmap(self._networks.h, (None, 0))

        z_prior = batched_encoder(params, sequences.observation)
        # [H+1, B, Z], [H, B]
        z_predictions, reward_pred = unroll_dynamics(
            params, sequences.action[:-1], z_prior[0]
        )

        # ([H, B], [H, B])
        q_values = batched_critic(params, z_predictions[:-1], sequences.action[:-1])
        q_values = jax.tree_map(lambda x: jnp.squeeze(x, axis=-1), q_values)

        return ModelOutputs(
            z_prediction=z_predictions,
            z_prior=z_prior,
            reward_pred=reward_pred,
            q_value=q_values,
        )

    def _policy_loss(
        self,
        params: tdmpc_networks.TDMPCNetworks,
        z_predictions: types.NestedArray,
        key: networks_lib.PRNGKey,
    ):
        """Computes the policy loss."""

        def policy(params, obs, key):
            return self._networks.pi(params, obs, self._min_std, key)

        batched_policy = jax.vmap(policy, in_axes=(None, 0, 0))

        # stop gradient from policy_loss to other model components
        frozen_params = jax.lax.stop_gradient(params)
        z_policy = jax.lax.stop_gradient(z_predictions)

        actions = batched_policy(
            params, z_policy, jax.random.split(key, z_predictions.shape[0])
        )
        # Compute policy loss
        q1, q2 = jax.vmap(self._networks.q, (None, 0, 0))(
            frozen_params, z_policy, actions
        )

        q = jnp.squeeze(jnp.minimum(q1, q2), axis=-1)

        return -q

    def _loss_fn(
        self,
        params: tdmpc_networks.TDMPCParams,
        target_params: tdmpc_networks.TDMPCParams,
        batch: reverb.ReplaySample,
        key: jax.random.PRNGKeyArray,
    ):
        """Computes the loss for the model and policy."""
        sequences: adders_reverb.Step = jax_utils.batch_to_sequence(batch.data)
        # [H+1, B, ...]
        # For horizon of H, we insert sequences of size H+1 for bootstrapping
        horizon = sequences.reward.shape[0] - 1

        predictions = self._compute_predictions(params, sequences)
        q1_t, q2_t = predictions.q_value

        key, target_key = jax.random.split(key)
        targets = self._compute_targets(
            target_params, sequences, params, predictions.z_prior, target_key
        )

        # Compute model loss
        consistency_loss = jnp.mean(
            _squared_error(predictions.z_prediction[1:], targets.z_target[1:]), axis=-1
        )
        reward_loss = _squared_error(predictions.reward_pred, targets.reward)
        value_loss = _squared_error(q1_t, targets.td_target) + _squared_error(
            q2_t, targets.td_target
        )
        priorities = _l1_loss(q1_t, targets.td_target) + _l1_loss(
            q2_t, targets.td_target
        )

        chex.assert_equal_shape([reward_loss, value_loss, priorities, consistency_loss])

        rhos = jnp.expand_dims(jnp.power(self._rho, jnp.arange(horizon + 1)), axis=-1)

        # Discount the model losses
        consistency_loss = jnp.sum(rhos[:-1] * consistency_loss, axis=0)
        reward_loss = jnp.sum(rhos[:-1] * reward_loss, axis=0)
        value_loss = jnp.sum(rhos[:-1] * value_loss, axis=0)
        priorities = jnp.sum(rhos[:-1] * priorities, axis=0)

        model_loss = (
            self._loss_scale.consistency * jnp.clip(consistency_loss, 0, 1e4)
            + self._loss_scale.reward * jnp.clip(reward_loss, 0, 1e4)
            + self._loss_scale.value * jnp.clip(value_loss, 0, 1e4)
        )
        model_loss = optax.scale_gradient(model_loss, 1.0 / horizon)

        # Compute policy loss
        policy_loss = self._policy_loss(params, predictions.z_prediction, key)
        chex.assert_rank([rhos, policy_loss], 2)
        chex.assert_equal_rank([rhos, policy_loss])
        policy_loss = jnp.mean(jnp.sum(rhos * policy_loss, axis=0), axis=0)

        # Compute weights for the model loss
        # NOTE: In the original paper, weights are only used for the model loss.
        # The policy loss is not weighted.
        probabilities = batch.info.probability
        importance_sampling_weights = (1 / probabilities).astype(jnp.float32)
        importance_sampling_weights **= self._importance_sampling_exponent
        importance_sampling_weights /= jnp.max(importance_sampling_weights)

        weighted_model_loss = importance_sampling_weights * model_loss
        weighted_model_loss = jnp.mean(weighted_model_loss, axis=0)

        total_loss = weighted_model_loss + policy_loss

        metrics = {
            "total_loss": jnp.mean(total_loss),
            "policy_loss": jnp.mean(policy_loss),
            "model/model_loss": jnp.mean(model_loss),
            "model/weighted_model_loss": jnp.mean(weighted_model_loss),
            "model/consistentcy_loss": jnp.mean(consistency_loss),
            "model/reward_loss": jnp.mean(reward_loss),
            "model/critic_loss": jnp.mean(value_loss),
        }

        return total_loss, (priorities, metrics)

    def _sgd_step(self, state: TrainingState, batch: TDMPCReplaySample):
        key, random_key = jax.random.split(state.key)
        grad_fn = jax.value_and_grad(self._loss_fn, has_aux=True)
        (_, (priorities, metrics)), gradients = grad_fn(
            state.params, state.target_params, batch, key
        )

        metrics["grad_norm"] = optax.global_norm(gradients)

        updates, opt_state = self._optimizer.update(
            gradients, state.opt_state, state.params
        )

        params = optax.apply_updates(state.params, updates)
        target_params = optax.incremental_update(
            params, state.target_params, self._target_update_rate
        )

        steps = state.steps + 1
        new_state = TrainingState(
            params=params,
            target_params=target_params,
            opt_state=opt_state,
            key=random_key,
            steps=steps,
        )

        return (new_state, metrics, priorities)

    def step(self):
        prefetching_splits = next(self._iterator)

        keys = prefetching_splits.host
        samples = prefetching_splits.device

        (self._state, metrics, priorities) = self._update_step(self._state, samples)

        if self._replay_client:
            self._async_priority_updater.put((keys, priorities))

        # Compute elapsed time.
        timestamp = time.time()
        elapsed_time = timestamp - self._timestamp if self._timestamp else 0
        self._timestamp = timestamp

        counts = self._counter.increment(steps=1, walltime=elapsed_time)

        if self._logger is not None:
            self._logger.write({**counts, **metrics})

    def get_variables(self, names: List[str]) -> List[networks_lib.Params]:
        variables = {"policy": self._state.params}
        return [variables[name] for name in names]

    def save(self) -> TrainingState:
        return self._state

    def restore(self, state: TrainingState):
        self._state = state
