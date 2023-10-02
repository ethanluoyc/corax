import re
from typing import Tuple

import dm_env_wrappers
import numpy as np
import rlds
import tensorflow as tf
from d4rl import infos as d4rl_info

from corax import types
from corax.utils import observers
from corax.wrappers import gym_wrapper


def make_environment(name: str, seed: int):
    import d4rl  # noqa: F401
    import gym

    environment = gym.make(name)
    environment.seed(seed)
    environment = gym_wrapper.GymWrapper(environment)
    environment = dm_env_wrappers.CanonicalSpecWrapper(environment, clip=True)
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


def _batched_step_to_transition(step: rlds.Step) -> types.Transition:
    return types.Transition(
        observation=tf.nest.map_structure(lambda x: x[0], step[rlds.OBSERVATION]),
        action=tf.nest.map_structure(lambda x: x[0], step[rlds.ACTION]),
        reward=tf.nest.map_structure(lambda x: x[0], step[rlds.REWARD]),
        discount=1.0 - tf.cast(step[rlds.IS_TERMINAL][1], dtype=tf.float32),  # type: ignore
        # If next step is terminal, then the observation may be arbitrary.
        next_observation=tf.nest.map_structure(lambda x: x[1], step[rlds.OBSERVATION]),
    )


def _batch_steps(episode: rlds.Episode) -> tf.data.Dataset:
    return rlds.transformations.batch(
        episode[rlds.STEPS], size=2, shift=1, drop_remainder=True
    )


def transform_transitions_dataset(
    episode_dataset,
    reward_scale: float = 1.0,
    reward_bias: float = 0.0,
):
    batched_steps = episode_dataset.flat_map(_batch_steps)
    transitions = rlds.transformations.map_steps(
        batched_steps, _batched_step_to_transition
    )
    return transitions.map(
        lambda transition: transition._replace(
            reward=(transition.reward * reward_scale + reward_bias)
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )


def add_episode_return(episodes_dataset):
    def add_fn(episode):
        episode_length = episode["steps"].cardinality()
        steps = episode["steps"].batch(episode_length).get_single_element()
        episode_return = tf.reduce_sum(steps["reward"])
        return {**episode, "episode_return": episode_return}

    return episodes_dataset.map(add_fn, num_parallel_calls=tf.data.AUTOTUNE)


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
