# type: ignore
"""A training loop for IL agent."""
import operator
import time
from typing import List, Optional, Protocol, Sequence

import dm_env
import jax.numpy as jnp
import numpy as np
import tree
from dm_env import specs

from corax import adders
from corax import core
from corax import types
from corax.utils import counting
from corax.utils import loggers
from corax.utils import observers as observers_lib


class EpisodeRewarder(Protocol):
    def compute_offline_rewards(self, agent_steps, update: bool):
        """Compute pseudo-rewards for an episode."""


class ImitationEnvironmentLoop(core.Worker):
    def __init__(
        self,
        environment: dm_env.Environment,
        actor: core.Actor,
        adder: Optional[adders.Adder] = None,
        rewarder: Optional[EpisodeRewarder] = None,
        counter: Optional[counting.Counter] = None,
        logger: Optional[loggers.Logger] = None,
        should_update_actor: bool = True,
        should_update_rewarder: bool = True,
        label: str = "environment_loop",
        observers: Sequence[observers_lib.EnvLoopObserver] = (),
    ):
        # Internalize agent and environment.
        self._environment = environment
        self._actor = actor
        self._adder = adder
        self._rewarder = rewarder
        self._counter = counter or counting.Counter()
        self._logger = logger or loggers.make_default_logger(
            label, steps_key=self._counter.get_steps_key()
        )
        self._should_update_actor = should_update_actor
        self._should_update_rewarder = should_update_rewarder
        self._observers = observers

    def run_episode(self) -> loggers.LoggingData:
        """Run one episode.

        Each episode is a loop which interacts first with the environment to get an
        observation and then give that observation to the agent in order to retrieve
        an action.

        Returns:
          An instance of `loggers.LoggingData`.
        """
        # Reset any counts and start the environment.
        episode_start_time = time.time()
        select_action_durations: List[float] = []
        env_step_durations: List[float] = []
        episode_steps: int = 0

        # For evaluation, this keeps track of the total undiscounted reward
        # accumulated during the episode.
        episode_return = tree.map_structure(
            _generate_zeros_from_spec, self._environment.reward_spec()
        )
        env_reset_start = time.time()
        timestep = self._environment.reset()
        env_reset_duration = time.time() - env_reset_start
        # Make the first observation.
        timesteps = []
        steps = []
        self._actor.observe_first(timestep)
        timesteps.append(timestep)

        for observer in self._observers:
            # Initialize the observer with the current state of the env after reset
            # and the initial timestep.
            observer.observe_first(self._environment, timestep)

        # Run an episode.
        while not timestep.last():
            # Book-keeping.
            episode_steps += 1

            # Generate an action from the agent's policy.
            select_action_start = time.time()
            action = self._actor.select_action(timestep.observation)
            select_action_durations.append(time.time() - select_action_start)

            # Step the environment with the agent's selected action.
            env_step_start = time.time()
            timestep = self._environment.step(action)
            env_step_durations.append(time.time() - env_step_start)

            # Have the agent and observers observe the timestep.
            # TODO(https://github.com/ethanluoyc/cdrl/issues/26).
            # Investigate a different approach to align observation, action.
            self._actor.observe(action, next_timestep=timestep)
            steps.append(
                types.Transition(timesteps[-1].observation, action, (), (), ())
            )
            timesteps.append(timestep)

            for observer in self._observers:
                # One environment step was completed. Observe the current state of the
                # environment, the current timestep and the action.
                observer.observe(self._environment, timestep, action)

            # Give the actor the opportunity to update itself.
            if self._should_update_actor:
                self._actor.update()

            # Equivalent to: episode_return += timestep.reward
            # We capture the return value because if timestep.reward is a JAX
            # DeviceArray, episode_return will not be mutated in-place. (In all other
            # cases, the returned episode_return will be the same object as the
            # argument episode_return.)
            episode_return = tree.map_structure(
                operator.iadd, episode_return, timestep.reward
            )

        steps.append(
            types.Transition(
                observation=timesteps[-1].observation,
                action=np.zeros_like(steps[0].action),
                reward=(),
                discount=(),
                next_observation=(),
            )
        )

        # Compute episode imitation return
        rewarder_start_time = time.time()

        def _convert_timestep(timestep):
            # TODO(yl): LP has a bug when passing non-JAX TF arrays.
            # Convert to JAX arrays until that is fixed.
            return tree.map_structure(
                lambda x: jnp.asarray(x) if x is not None else None, timestep
            )

        psuedo_rewards = self._rewarder.compute_offline_rewards(
            [_convert_timestep(step) for step in steps],
            self._should_update_rewarder,
        )
        assert len(timesteps) - 1 == len(psuedo_rewards)

        episode_return_imitation = np.sum(psuedo_rewards)
        rewarder_duration_sec = time.time() - rewarder_start_time

        # Add the updated reward timesteps to adder
        if self._adder is not None:
            first_timestep = timesteps[0]
            self._adder.add_first(first_timestep._replace(reward=psuedo_rewards[0]))
            actions = [step.action for step in steps[:-1]]
            assert len(actions) == len(timesteps) - 1 == len(psuedo_rewards)
            for action, next_ts, relabeled_reward in zip(
                actions, timesteps[1:], psuedo_rewards
            ):
                self._adder.add(action, next_ts._replace(reward=relabeled_reward))
            self._adder.reset()

        # Record counts.
        counts = self._counter.increment(episodes=1, steps=episode_steps)

        # Collect the results and combine with counts.
        steps_per_second = episode_steps / (time.time() - episode_start_time)
        result = {
            "episode_length": episode_steps,
            "episode_return": episode_return,
            "episode_return_imitation": episode_return_imitation,
            "steps_per_second": steps_per_second,
            "env_reset_duration_sec": env_reset_duration,
            "select_action_duration_sec": np.mean(select_action_durations),
            "env_step_duration_sec": np.mean(env_step_durations),
            "rewarder_duration_sec": rewarder_duration_sec,
        }
        result.update(counts)
        for observer in self._observers:
            result.update(observer.get_metrics())
        return result

    def run(
        self,
        num_episodes: Optional[int] = None,
        num_steps: Optional[int] = None,
    ) -> int:
        """Perform the run loop.

        Run the environment loop either for `num_episodes` episodes or for at
        least `num_steps` steps (the last episode is always run until completion,
        so the total number of steps may be slightly more than `num_steps`).
        At least one of these two arguments has to be None.

        Upon termination of an episode a new episode will be started. If the number
        of episodes and the number of steps are not given then this will interact
        with the environment infinitely.

        Args:
          num_episodes: number of episodes to run the loop for.
          num_steps: minimal number of steps to run the loop for.

        Returns:
          Actual number of steps the loop executed.

        Raises:
          ValueError: If both 'num_episodes' and 'num_steps' are not None.
        """

        if not (num_episodes is None or num_steps is None):
            raise ValueError('Either "num_episodes" or "num_steps" should be None.')

        def should_terminate(episode_count: int, step_count: int) -> bool:
            return (num_episodes is not None and episode_count >= num_episodes) or (
                num_steps is not None and step_count >= num_steps
            )

        episode_count: int = 0
        step_count: int = 0
        while not should_terminate(episode_count, step_count):
            episode_start = time.time()
            result = self.run_episode()
            result = {**result, **{"episode_duration": time.time() - episode_start}}
            episode_count += 1
            step_count += int(result["episode_length"])
            # Log the given episode results.
            self._logger.write(result)

        return step_count


def _generate_zeros_from_spec(spec: specs.Array) -> np.ndarray:
    return np.zeros(spec.shape, spec.dtype)
