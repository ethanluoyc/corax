# type: ignore
"""Rewarder for computing rewards using Optimal Transport."""
from typing import Any, Callable, Optional, Protocol, Sequence

import chex
import jax
import jax.numpy as jnp
import numpy as onp
import ott
from ott.geometry import pointcloud
import ott.geometry.costs
from ott.problems.linear import linear_problem
from ott.solvers.linear import sinkhorn

from corax import types
from corax.jax import networks as networks_lib
from corax.jax import running_statistics
from corax.jax import variable_utils

EncoderFn = Callable[[networks_lib.Params, networks_lib.Observation], types.NestedArray]
PreprocessorState = Any


class Preprocessor(Protocol):
    """Interface for preprocessor.

    The preprocessor is used for extracting informative state
    representation from the observations.

    Implementation of the Preprocessor protocol is expected
    to be stateless. This allows the preprocessor to be used
    in a JAX jitted function.
    """

    def init(self):
        ...

    def update(self, state: PreprocessorState, atoms) -> PreprocessorState:
        ...

    def preprocess(self, params: networks_lib.Params, state: PreprocessorState, atoms):
        ...


class EncoderPreprocessor(Preprocessor):
    """Parametric encoder preprocessor."""

    def __init__(self, encoder_fn: EncoderFn):
        self._encoder_fn = encoder_fn

    def init(self):
        return ()

    def update(self, state, atoms):
        del state, atoms
        return ()

    def preprocess(self, params, state, atoms):
        del state
        return self._encoder_fn(params, atoms)  # pylint: disable=not-callable


class MeanStdPreprocessor(Preprocessor):
    """Running mean/std preprocessor."""

    def __init__(self, spec: types.NestedArray, partial_update: bool = False):
        self._observation_spec = spec
        self._partial_update = partial_update

    def init(self):
        return running_statistics.init_state(self._observation_spec)

    def update(self, state, atoms):
        assert atoms.ndim == 2
        if self._partial_update:
            state = running_statistics.init_state(self._observation_spec)
            state = running_statistics.update(state, atoms)
            return state
        else:
            state = running_statistics.update(state, atoms)
            return state

    def preprocess(self, params, state, atoms):
        del params
        return running_statistics.normalize(atoms, state)


class NoOpPreprocessor(Preprocessor):
    """Identity preprocessor."""

    def init(self):
        return ()

    def update(self, state, atoms):
        del state, atoms
        return ()

    def preprocess(self, params, state, atoms):
        del params, state
        return atoms


class AggregateFn(Protocol):
    """Function for combining pseudo-rewards from multiple expert demonstrations."""

    def __call__(self, rewards: chex.Array, **kwargs) -> chex.Array:
        ...


def aggregate_top_k(rewards, k=1):
    """Aggregate rewards from multiple expert demonstrations by mean of top-K demos."""
    scores = jnp.sum(rewards, axis=-1)
    _, indices = jax.lax.top_k(scores, k=k)
    return jnp.mean(rewards[indices], axis=0)


def aggregate_mean(rewards):
    """Aggregate rewards from multiple expert demonstrations by taking the mean"""
    return jnp.mean(rewards, axis=0)


class SquashingFn(Protocol):
    """Function for squashing pseudo-rewards computed from the OT computation."""

    def __call__(self, rewards: chex.Array, **kwargs) -> chex.Array:
        ...


def squashing_linear(rewards, alpha: float = 10.0):
    """Compute squashed rewards with alpha * rewards."""
    return alpha * rewards


def squashing_exponential(rewards, alpha: float = 5.0, beta: float = 5.0):
    """Compute squashed rewards with alpha * exp(beta * rewards)."""
    return alpha * jnp.exp(beta * rewards)


class OTRewarder:
    """OT rewarder.

    The rewarder measures similarities with the expert demonstration.
    """

    def __init__(
        self,
        demonstrations: Sequence[Sequence[types.Transition]],
        episode_length: int,
        preprocessor: Optional[Preprocessor] = None,
        aggregate_fn: AggregateFn = aggregate_top_k,
        squashing_fn: SquashingFn = squashing_linear,
        max_iterations: float = 100,
        threshold: float = 1e-9,
        epsilon: float = 1e-2,
        preprocessor_update_period: int = 1,
        use_actions_for_distance: bool = False,
        variable_client: Optional[variable_utils.VariableClient] = None,
    ):
        if use_actions_for_distance and preprocessor is not None:
            raise NotImplementedError("Use actions with preprocessor not yet supported")

        self._episode_length = episode_length
        self._aggregate_fn = aggregate_fn
        self._squashing_fn = squashing_fn
        self._max_iterations = max_iterations
        self._threshold = threshold
        self._epsilon = epsilon
        self._preprocessor_update_period = preprocessor_update_period
        self._variable_client = variable_client
        self._params = variable_client.params if variable_client is not None else None
        self._num_episodes_seen = 0
        self._use_actions_for_distance = use_actions_for_distance

        # Prepare expert atoms
        self._expert_atoms = []
        self._expert_weights = []
        # Vectorize atoms, pad the atoms and compute weights
        for demo in demonstrations:
            atoms, weights, _, _ = _pack_trajectory(
                demo, self._episode_length, self._use_actions_for_distance
            )
            self._expert_atoms.append(atoms)
            self._expert_weights.append(weights)

        self._expert_atoms = onp.stack(self._expert_atoms)
        self._expert_weights = onp.stack(self._expert_weights)

        self._preprocessor = preprocessor or NoOpPreprocessor()
        self._preprocessor_state = self._preprocessor.init()

        self._batched_ot_solve = jax.jit(
            jax.vmap(self._solve_ot, in_axes=(None, None, 0, 0, None, None))
        )
        self._compute_rewards = jax.jit(self._compute_otil_rewards)
        self._update_preprocessor = jax.jit(self._preprocessor.update)

    def _solve_ot(
        self, params, state, expert_atoms, expert_weights, agent_atoms, agent_weights
    ):
        agent_atoms = self._preprocessor.preprocess(params, state, agent_atoms)
        expert_atoms = self._preprocessor.preprocess(params, state, expert_atoms)
        cost_fn = ott.geometry.costs.Cosine()
        geom = pointcloud.PointCloud(
            agent_atoms, expert_atoms, cost_fn=cost_fn, epsilon=self._epsilon
        )
        solver = sinkhorn.Sinkhorn(
            threshold=self._threshold, max_iterations=self._max_iterations
        )
        problem = linear_problem.LinearProblem(geom, a=agent_weights, b=expert_weights)
        sinkhorn_output = solver(problem)
        coupling_matrix = geom.transport_from_potentials(
            sinkhorn_output.f, sinkhorn_output.g
        )
        cost_matrix = cost_fn.all_pairs(agent_atoms, expert_atoms)
        ot_costs = jnp.einsum("ij,ij->i", coupling_matrix, cost_matrix)
        return ot_costs

    def _compute_otil_rewards(
        self,
        params,
        preprocessor_state,
        all_expert_atoms,
        all_expert_weights,
        agent_atoms,
        agent_weights,
        agent_mask,
    ):
        ot_costs = self._batched_ot_solve(
            params,
            preprocessor_state,
            all_expert_atoms,
            all_expert_weights,
            agent_atoms,
            agent_weights,
        )
        # Compute pseudo rewards based on alignments
        pseudo_rewards = -ot_costs
        rewards = self._squashing_fn(pseudo_rewards)
        # Maskout rewards for padded atoms with zero
        rewards = jnp.where(agent_mask, rewards, 0.0)
        rewards = self._aggregate_fn(rewards)
        return rewards

    def compute_offline_rewards(self, agent_steps):
        """Compute rewards based on optimal transport."""
        # Vectorize atoms, pad the atoms and compute weights
        agent_atoms, agent_weights, num_agent_atoms, agent_mask = _pack_trajectory(
            agent_steps, self._episode_length, self._use_actions_for_distance
        )

        # Update preprocessor state if necessary
        if self._num_episodes_seen % self._preprocessor_update_period == 0:
            # Maybe update parameters
            if self._variable_client is not None:
                self._variable_client.update_and_wait()
                self._params = self._variable_client.params
            # Take only the first num_agent_atoms for updating the preprocessor
            # due to padding.
            self._preprocessor_state = self._update_preprocessor(
                self._preprocessor_state, agent_atoms[:num_agent_atoms]
            )

        rewards = self._compute_otil_rewards(
            self._params,
            self._preprocessor_state,
            self._expert_atoms,
            self._expert_weights,
            agent_atoms,
            agent_weights,
            agent_mask,
        )
        # "Unpad" the computed rewards
        rewards = rewards[:num_agent_atoms]

        self._num_episodes_seen += 1

        return jax.device_get(rewards)


def _pack_trajectory(
    trajectory: Sequence[types.Transition],
    max_sequence_length: int,
    use_actions: bool = False,
):
    """Pack a list of observations in a trajectory into array."""
    num_atoms = len(trajectory)
    if use_actions:
        observations = [
            onp.concatenate([atom.observation, atom.action], axis=-1)
            for atom in trajectory
        ]
    else:
        observations = [atom.observation for atom in trajectory]

    atoms = onp.stack(observations, axis=0)
    atoms = _pad(atoms, max_sequence_length)
    # Compute weights for used in OT
    # NOTE: The weights are padded with zeros so that the
    # padded atoms has zero probability.
    # This ensures that the computed OT solution is identical
    # with non-padded atoms.
    weights = onp.ones((num_atoms,)) / max_sequence_length
    weights = _pad(weights, max_sequence_length)
    # Compute a mask for the valid steps in the sequence
    mask = _pad(onp.ones(num_atoms, dtype=bool), max_sequence_length)
    return atoms, weights, num_atoms, mask


def _pad(x, max_sequence_length: int):
    paddings = [(0, max_sequence_length - x.shape[0])]
    paddings.extend([(0, 0) for _ in range(x.ndim - 1)])
    return onp.pad(x, paddings, mode="constant", constant_values=0.0)
