import re
from typing import Tuple

import dm_env_wrappers
import numpy as np
from d4rl import infos as d4rl_info

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
