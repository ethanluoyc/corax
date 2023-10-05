# type: ignore
from typing import Optional

import rlds
import tensorflow as tf

from corax import types


def is_truncated_last_step(step: rlds.Step) -> tf.Tensor:
    return tf.logical_and(step[rlds.IS_LAST], tf.logical_not(step[rlds.IS_TERMINAL]))


def skip_truncated_last_step(episode: rlds.Episode) -> rlds.Episode:
    filtered_steps = episode[rlds.STEPS].filter(
        lambda step: tf.logical_not(is_truncated_last_step(step))
    )
    return {**episode, rlds.STEPS: filtered_steps}


def episodes_to_transitions_dataset(
    episode_dataset: tf.data.Dataset,
    cycle_length: Optional[int] = None,
    block_length: Optional[int] = None,
    num_parallel_calls: Optional[int] = None,
    deterministic: Optional[bool] = None,
):
    def _batched_step_to_transition(step: rlds.Step) -> types.Transition:
        return types.Transition(
            observation=tf.nest.map_structure(lambda x: x[0], step[rlds.OBSERVATION]),
            action=tf.nest.map_structure(lambda x: x[0], step[rlds.ACTION]),
            reward=tf.nest.map_structure(lambda x: x[0], step[rlds.REWARD]),
            discount=1.0 - tf.cast(step[rlds.IS_TERMINAL][1], dtype=tf.float32),  # type: ignore
            next_observation=tf.nest.map_structure(
                lambda x: x[1], step[rlds.OBSERVATION]
            ),
        )

    def _batch_steps(episode: rlds.Episode) -> tf.data.Dataset:
        return rlds.transformations.batch(
            episode[rlds.STEPS], size=2, shift=1, drop_remainder=True
        )

    batched_steps = episode_dataset.interleave(
        _batch_steps,
        cycle_length=cycle_length,
        block_length=block_length,
        num_parallel_calls=num_parallel_calls,
        deterministic=deterministic,
    )
    transitions = rlds.transformations.map_steps(
        batched_steps, _batched_step_to_transition
    )
    return transitions


def compute_episode_return(steps_dataset: tf.data.Dataset):
    episode_length = tf.cast(
        rlds.transformations.episode_length(steps_dataset), tf.int64
    )
    steps = steps_dataset.batch(episode_length).get_single_element()
    episode_return = tf.reduce_sum(steps["reward"])
    return episode_return


def rescale_reward(step: rlds.Step, reward_scale, reward_bias):
    return {**step, rlds.REWARD: step[rlds.REWARD] * reward_scale + reward_bias}
