# type: ignore
"""Rewarder for computing rewards used in ROT."""
import threading
from typing import Any, Callable, Optional, Protocol, Sequence

import jax
import jax.numpy as jnp
import numpy as onp
import ott
from ott.geometry import pointcloud
import ott.geometry.costs
from ott.solvers.linear import sinkhorn
import tree

from corax import types
from corax.jax import networks as networks_lib
from corax.jax import running_statistics
from corax.jax import utils
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

    def init(self, expert_trajectories):
        ...

    def update(self, state: PreprocessorState, trajectory) -> PreprocessorState:
        ...

    def preprocess(
        self, params: networks_lib.Params, state: PreprocessorState, trajectory
    ):
        ...


class EncoderPreprocessor(Preprocessor):
    """Parametric encoder preprocessor."""

    def __init__(self, encoder_fn: EncoderFn, use_actions: bool = False):
        self._encoder_fn = encoder_fn
        self._use_actions = use_actions

    def init(self, expert_trajectories):
        del expert_trajectories
        return ()

    def update(self, state, trajectory):
        del state, trajectory
        return ()

    def preprocess(self, params, state, trajectory):
        observations, actions = trajectory
        encoded = self._encoder_fn(params, observations)  # pylint: disable=not-callable
        if self._use_actions:
            atoms = utils.batch_concat((encoded, actions))
        else:
            atoms = encoded
        return atoms


class MeanStdPreprocessor(Preprocessor):
    """Running mean/std preprocessor."""

    def __init__(
        self,
        spec: types.NestedArray,
        action_spec: types.NestedArray = None,
        partial_update: bool = False,
        use_actions: bool = False,
    ):
        if use_actions and action_spec is None:
            raise ValueError(
                "action_spec needs to be specified when use_actions is True."
            )
        self._observation_spec = spec
        self._action_spec = action_spec
        self._partial_update = partial_update
        self._use_actions = use_actions

    def init(self, expert_trajectories):
        del expert_trajectories
        if self._use_actions:
            return running_statistics.init_state(
                (self._observation_spec, self._action_spec)
            )
        else:
            return running_statistics.init_state(self._observation_spec)

    def update(self, state, trajectory):
        observations, actions = trajectory
        if self._use_actions:
            atoms = (observations, actions)
        else:
            atoms = observations
        if self._partial_update:
            # Partial update uses only statistics from the latest atoms
            state = running_statistics.init_state(self._observation_spec)
            new_state = running_statistics.update(state, atoms)
        else:
            new_state = running_statistics.update(state, atoms)
        new_state = new_state.replace(
            std=tree.map_structure(
                lambda s: (s < 1e-5).astype(jnp.float32) + s, new_state.std
            )
        )
        return new_state

    def preprocess(self, params, state, trajectory):
        del params
        observations, actions = trajectory
        if not self._use_actions:
            atoms = observations
        else:
            atoms = (observations, actions)
        normalized = running_statistics.normalize(atoms, state)
        return utils.batch_concat(normalized)


def _concat_trajectories(trajectories):
    observations = tree.map_structure(
        lambda *xs: onp.concatenate(xs, axis=0), *[traj[0] for traj in trajectories]
    )
    actions = tree.map_structure(
        lambda *xs: onp.concatenate(xs, axis=0), *[traj[1] for traj in trajectories]
    )
    return observations, actions


class FixedStatsPreprocessor(Preprocessor):
    """Running mean/std preprocessor."""

    def __init__(self, use_actions: bool = False):
        self._use_actions = use_actions

    def init(self, expert_trajectories):
        observations, actions = _concat_trajectories(expert_trajectories)
        if self._use_actions:
            atoms = (observations, actions)
        else:
            atoms = observations

        mean = tree.map_structure(lambda x: onp.mean(x, axis=0), atoms)
        std = tree.map_structure(lambda x: onp.std(x, axis=0, dtype="float64"), atoms)
        std = tree.map_structure(lambda s: (s < 1e-5) + s, std)
        return jax.device_put((mean, std))

    def update(self, state, trajectory):
        del trajectory
        return state

    def preprocess(self, params, state, trajectory):
        del params
        observations, actions = trajectory
        mean, std = state
        if not self._use_actions:
            atoms = observations
        else:
            atoms = (observations, actions)
        return utils.batch_concat(
            tree.map_structure(lambda a, m, s: (a - m) / s, atoms, mean, std)
        )


class AggregateFn(Protocol):
    """Function for combining pseudo-rewards from multiple expert demonstrations."""

    def __call__(self, rewards: jax.Array, **kwargs) -> jax.Array:
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

    def __call__(self, rewards: jax.Array, **kwargs) -> jax.Array:
        ...


def squashing_linear(rewards, alpha: float = 10.0):
    """Compute squashed rewards with alpha * rewards."""
    return alpha * rewards


def squashing_exponential(rewards, alpha: float = 5.0, beta: float = 5.0):
    """Compute squashed rewards with alpha * exp(beta * rewards)."""
    return alpha * jnp.exp(beta * rewards)


class ROTRewarder:
    """ROT rewarder.

    The rewarder measures similarities with the expert demonstration.
    """

    def __init__(
        self,
        demonstrations: Sequence[Sequence[types.Transition]],
        episode_length: int,
        preprocessor: Optional[Preprocessor] = None,
        aggregate_fn: AggregateFn = aggregate_top_k,
        max_iterations: float = 100,
        threshold: float = 1e-9,
        epsilon: float = 1e-2,
        preprocessor_update_period: int = 1,
        auto_reward_scale_factor: float = 10.0,
        cost_fn: str = "cosine",
        variable_client: Optional[variable_utils.VariableClient] = None,
    ):
        if cost_fn not in ["cosine", "euclidean"]:
            raise ValueError("cost_fn must be either cosine or euclidean")
        expert_trajectories = []

        for demo in demonstrations:
            expert_trajectory = self._pack_trajectory(demo)
            expert_trajectories.append(expert_trajectory)

        self._expert_trajectories = expert_trajectories
        self._cost_fn = cost_fn
        self._episode_length = episode_length
        self._aggregate_fn = jax.jit(aggregate_fn)
        self._preprocessor_update_period = preprocessor_update_period
        self._variable_client = variable_client
        self._params = variable_client.params if variable_client is not None else None
        self._num_episodes_seen = 0
        self._reward_scale = 1.0
        self._auto_reward_scale_factor = auto_reward_scale_factor

        self._preprocessor = preprocessor
        self._preprocessor_state = self._preprocessor.init(self._expert_trajectories)
        self._mutex = threading.Lock()

        def solve_ot(expert_atoms: jax.Array, agent_atoms: jax.Array):
            # agent_atom_mean = jnp.mean(agent_atoms, axis=0)
            # agent_atom_std = jnp.std(agent_atoms, axis=0)
            # expert_atoms = (expert_atoms - agent_atom_mean) / (agent_atom_std + 1e-5)
            # agent_atoms = (agent_atoms - agent_atom_mean) / (agent_atom_std + 1e-5)
            if self._cost_fn == "cosine":
                cost_fn = ott.geometry.costs.Cosine()
                geom = pointcloud.PointCloud(
                    agent_atoms,
                    expert_atoms,
                    cost_fn=cost_fn,
                    epsilon=epsilon,
                    scale_cost=True,
                )
            elif self._cost_fn == "euclidean":
                cost_fn = ott.geometry.costs.Euclidean()
                geom = pointcloud.PointCloud(
                    agent_atoms,
                    expert_atoms,
                    cost_fn=cost_fn,
                    epsilon=epsilon,
                    scale_cost=True,
                )

            problem = ott.problems.linear.linear_problem.LinearProblem(geom)
            solver = sinkhorn.Sinkhorn(
                threshold=threshold, max_iterations=max_iterations
            )
            solver_output = solver(problem)
            coupling_matrix = geom.transport_from_potentials(
                solver_output.f, solver_output.g
            )
            cost_matrix = cost_fn.all_pairs(agent_atoms, expert_atoms)
            # Eqn (2) of https://arxiv.org/pdf/2006.04678.pdf, Eqn (5) of OTIL
            ot_costs = jnp.einsum("ij,ij->i", coupling_matrix, cost_matrix)
            return ot_costs

        self._solve_ot = jax.jit(solve_ot)
        self._update_preprocessor = jax.jit(self._preprocessor.update)
        self._preprocess = jax.jit(self._preprocessor.preprocess)

    def _pack_trajectory(self, steps):
        observations = [step.observation for step in steps]
        actions = [step.action for step in steps]
        observations = tree.map_structure(lambda *xs: onp.stack(xs, 0), *observations)
        actions = tree.map_structure(lambda *xs: onp.stack(xs, 0), *actions)
        return observations, actions

    def _compute_offline_rewards(self, agent_steps, update: bool):
        """Compute rewards based on optimal transport."""

        agent_trajectory = self._pack_trajectory(agent_steps)

        # Update preprocessor state if necessary
        if update and self._num_episodes_seen % self._preprocessor_update_period == 0:
            # Maybe update parameters
            if self._variable_client is not None:
                self._variable_client.update_and_wait()
                self._params = self._variable_client.params
            self._preprocessor_state = self._update_preprocessor(
                self._preprocessor_state, agent_trajectory
            )

        agent_atoms = self._preprocess(
            self._params, self._preprocessor_state, agent_trajectory
        )

        rewards = []
        for expert_trajectory in self._expert_trajectories:
            # NOTE(yl): It may be possible to only recompute
            # expert atoms per preprocessor update, which saves some
            # computation, especially for an encoder preprocessors.
            # However, care must be taken to ensure there are
            # no concurrency issues when we move to distributed rewarder.
            expert_atoms = self._preprocess(
                self._params, self._preprocessor_state, expert_trajectory
            )
            ot_costs = self._solve_ot(expert_atoms, agent_atoms)
            pseudo_rewards = -ot_costs
            rewards.append(pseudo_rewards)

        rewards = jnp.stack(rewards)
        rewards = self._aggregate_fn(rewards)

        if update:
            if self._num_episodes_seen == 0:
                self._reward_scale = self._auto_reward_scale_factor / onp.sum(
                    onp.abs(rewards)
                )
            self._num_episodes_seen += 1

        rewards = rewards * self._reward_scale
        return rewards[:-1]

    def compute_offline_rewards(self, agent_steps, update: bool):
        """Compute rewards based on optimal transport."""
        with self._mutex:
            return self._compute_offline_rewards(agent_steps, update)
