import numpy as np
import rlds
import tensorflow as tf
import tree


def stack_observations_np(observations, stack_size):
    def _repeat_first(sequence):
        reps = [1] * (sequence[0].ndim + 1)
        reps[0] = stack_size - 1
        repeated_first_step = np.tile(sequence[0], reps)
        return np.concatenate(
            [repeated_first_step, sequence], 0
        )  # [:-(stack_size - 1)]

    def _stack_observation(observation):
        stack = [np.roll(observation, i, axis=0) for i in range(stack_size)]
        stack.reverse()  # chronological order
        return np.concatenate(stack, axis=-1)

    observations = _repeat_first(observations)
    return _stack_observation(observations)[(stack_size - 1) :]


def get_n_step_transitions(episode_steps, nsteps, discount):
    """Get n_step transitions from an episode of steps"""
    episode_length = episode_steps["reward"].shape[0] - 1
    transitions = []

    for t in range(0, episode_length - nsteps + 1):
        observations = episode_steps["observation"][t : t + nsteps + 1]
        actions = episode_steps["action"][t : t + nsteps + 1]
        rewards = episode_steps["reward"][t : t + nsteps + 1]
        discounts = episode_steps["discount"][t : t + nsteps + 1]

        total_reward = rewards[0]
        total_discount = discounts[0]

        for n in range(1, len(discounts) - 1):
            total_discount *= discount
            total_reward += rewards[n] * total_discount
            total_discount *= discounts[n]

        transitions.append(
            {
                "observation": observations[0],
                "next_observation": observations[-1],
                "reward": total_reward,
                "discount": total_discount,
                "action": actions[0],
            }
        )
    # This can be faster
    return tree.map_structure(lambda *xs: np.stack(xs), *transitions)


def stack_observation_tfds(observations: tf.data.Dataset, stack_size: int):
    first_obs = observations.take(1)
    observations = first_obs.repeat(stack_size - 1).concatenate(observations)
    data = observations.window(stack_size, shift=1, drop_remainder=True).flat_map(
        lambda x: x.batch(stack_size, drop_remainder=True)
    )
    return data.map(
        lambda x: tf.concat(tf.unstack(x, axis=0), axis=-1),
        num_parallel_calls=tf.data.AUTOTUNE,
    )


def process_data(steps_dataset, stack_size):
    obs = steps_dataset.map(
        lambda step: step["observation"], num_parallel_calls=tf.data.AUTOTUNE
    )
    stacked_obs = stack_observation_tfds(obs, stack_size)
    zipped_ds = tf.data.Dataset.zip((stacked_obs, steps_dataset))

    def combine(stacked_obs, steps):
        steps["observation"] = stacked_obs
        return steps

    return zipped_ds.map(combine, num_parallel_calls=tf.data.AUTOTUNE)


@tf.function
def _sequence_to_transitions(sequence, discount):
    total_reward = sequence["reward"][0]
    total_discount = sequence["discount"][0]
    n_steps = len(sequence["reward"]) - 1

    for t in range(1, n_steps):
        total_discount *= discount
        total_reward += total_discount * sequence["reward"][t]
        total_discount *= sequence["discount"][t]

    return {
        "observation": sequence["observation"][0],
        "next_observation": sequence["observation"][-1],
        "action": sequence["action"][0],
        "reward": total_reward,
        "discount": total_discount,
    }


def tfds_get_n_step_transitions(steps_dataset, nsteps, discount):
    """Get n_step transitions from an episode of steps"""
    batched_steps = rlds.transformations.batch(
        steps_dataset, size=nsteps + 1, shift=1, drop_remainder=True
    )
    return batched_steps.map(
        lambda seq: _sequence_to_transitions(seq, discount),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )
