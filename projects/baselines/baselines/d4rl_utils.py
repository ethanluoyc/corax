import re
from typing import Callable, Tuple

import dm_env
import dm_env_wrappers
import numpy as np
from absl import logging
from d4rl import infos as d4rl_info

import corax
from corax.utils import counting
from corax.utils import loggers
from corax.utils import observers
from corax.wrappers import gym_wrapper


def load_d4rl_environment(name: str, seed: int):
    import d4rl  # noqa: F401
    import gym

    environment = gym.make(name)
    environment.seed(seed)
    environment = gym_wrapper.GymWrapper(environment)
    environment = dm_env_wrappers.SinglePrecisionWrapper(environment)

    return environment


def parse_d4rl_dataset_name(dataset_name: str) -> Tuple[str, str, str]:
    match = re.match(
        r"(?P<env>[a-z0-9]+)-(?P<dataset>[a-z\-]+)-(?P<version>v\d)", dataset_name
    )
    if not match:
        raise ValueError(f"Invalid D4RL dataset name: {dataset_name}")

    return match.group("env"), match.group("dataset"), match.group("version")


def get_tfds_name(d4rl_name: str) -> str:
    """Return the corresponding TFDS name for a given D4RL dataset name."""

    env, dataset, version = parse_d4rl_dataset_name(d4rl_name)
    if env in ["halfcheetah", "hopper", "walker2d", "ant"]:
        if version == "v0" and dataset == "medium-replay":
            return f"d4rl_mujoco_{env}/{version}-mixed"
        else:
            return f"d4rl_mujoco_{env}/{version}-{dataset}"
    elif env in ["antmaze"]:
        return f"d4rl_antmaze/{dataset}-{version}"
    elif env in ["pen", "door", "hammer", "relocate"]:
        return f"d4rl_adroit_{env}/{version}-{dataset}"
    else:
        raise ValueError(f"Unknown D4RL environment: {env}")


class D4RLScoreObserver(observers.EnvLoopObserver):
    def __init__(self, dataset_name: str) -> None:
        assert dataset_name in d4rl_info.REF_MAX_SCORE
        self._episode_return = 0
        self._dataset_name = dataset_name

    def observe_first(self, env, timestep) -> None:
        """Observes the initial state."""
        del env, timestep
        self._episode_return = 0

    def observe(self, env, timestep, action: np.ndarray) -> None:
        """Records one environment step."""
        del env, action
        self._episode_return += timestep.reward

    def get_metrics(self):
        """Returns metrics collected for the current episode."""
        score = self._episode_return
        ref_min_score = d4rl_info.REF_MIN_SCORE[self._dataset_name]
        ref_max_score = d4rl_info.REF_MAX_SCORE[self._dataset_name]
        normalized_score = (score - ref_min_score) / (ref_max_score - ref_min_score)
        return {"d4rl_normalized_score": normalized_score}


class D4RLEvaluator:
    def __init__(
        self,
        environment_factory: Callable[[], dm_env.Environment],
        actor: corax.Actor,
        counter: counting.Counter,
        logger: loggers.Logger,
    ):
        self._environment_factory = environment_factory
        self._actor = actor
        self._counter = counter
        self._logger = logger

    def run(self, num_episodes: int):
        environment = self._environment_factory()
        self._actor.update(wait=True)
        avg_reward = 0.0
        steps = 0
        for _ in range(num_episodes):
            timestep = environment.reset()
            self._actor.observe_first(timestep)
            while not timestep.last():
                action = self._actor.select_action(timestep.observation)
                timestep = environment.step(action)
                self._actor.observe(action, timestep)
                steps += 1
                avg_reward += timestep.reward

        counts = self._counter.increment(steps=steps, episodes=num_episodes)

        avg_reward /= num_episodes
        d4rl_score = environment.get_normalized_score(avg_reward)  # type: ignore
        results = {
            "average_episode_return": avg_reward,
            "average_normalized_score": d4rl_score,
        }
        results.update(counts)

        logging.info("---------------------------------------")
        logging.info(
            "Evaluation over %d episodes: %.3f (unormalized %.3f)",
            num_episodes,
            d4rl_score,
            avg_reward,
        )
        logging.info("---------------------------------------")
        self._logger.write(results)
        return results
