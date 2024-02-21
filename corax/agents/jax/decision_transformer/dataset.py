import functools

import numpy as np
import rlds
import tensorflow as tf

from corax.jax import running_statistics


def get_observation_mean_std(episode_dataset):
    mean, std = rlds.transformations.mean_and_std(
        episode_dataset, get_step_fields=rlds.transformations.sar_fields_mask
    )

    observation_mean = np.asarray(mean["observation"], dtype=np.float32)
    observation_std = np.asarray(std["observation"], dtype=np.float32)

    return running_statistics.NestedMeanStd(mean=observation_mean, std=observation_std)  # type: ignore


def discounted_return(rewards, discounts):
    def discounted_return_fn(acc, reward_discount):
        reward, discount = reward_discount
        return acc * discount + reward

    return tf.scan(
        fn=discounted_return_fn,
        elems=(rewards, discounts),
        reverse=True,
        initializer=0.0,
    )


def transform_decision_transformer_input(
    episode_dataset,
    *,
    sequence_length: int,
    scale: float,
    observation_mean_std,
):
    def add_return_to_go(episode):
        episode_length = episode["steps"].cardinality()
        steps = episode["steps"].batch(episode_length).get_single_element()
        returns = discounted_return(steps["reward"], steps["discount"])
        timesteps = tf.range(episode_length)
        # Scale RTG
        steps["return_to_go"] = returns / scale  # type: ignore
        steps["timestep"] = timesteps
        # TODO(yl): Remove this
        if "info" in steps:
            steps.pop("infos")
        steps = tf.nest.map_structure(lambda x: x[:-1], steps)
        episode["steps"] = tf.data.Dataset.from_tensor_slices(steps)
        return episode

    def _pad_along_axis(x, padded_size, axis=0, value=0):
        pad_width = padded_size - tf.shape(x)[axis]  # type: ignore
        if pad_width <= 0:
            return x
        padding = [(0, 0)] * len(x.shape.as_list())
        padding[axis] = (pad_width, 0)
        padded = tf.pad(x, padding, mode="CONSTANT", constant_values=value)
        return padded

    def pad_steps(steps, max_len):
        normalized_observation = (
            steps["observation"] - observation_mean_std.mean
        ) / observation_mean_std.std
        padded_obs = _pad_along_axis(normalized_observation, max_len, 0, 0)
        padded_act = _pad_along_axis(steps["action"], max_len, 0, -10)
        padded_rtg = _pad_along_axis(steps["return_to_go"], max_len, 0, 0)
        padded_discounts = _pad_along_axis(steps["discount"], max_len, 0, 2)
        padded_timesteps = _pad_along_axis(steps["timestep"], max_len, 0, 0)
        mask = _pad_along_axis(
            tf.ones(tf.shape(steps["reward"])[0], dtype=bool),  # type: ignore
            max_len,
            0,
            False,
        )
        return {
            "observation": padded_obs,
            "action": padded_act,
            "return_to_go": tf.expand_dims(padded_rtg, axis=-1),
            "discount": padded_discounts,
            "timestep": padded_timesteps,
            "mask": mask,
        }

    dataset = episode_dataset.map(add_return_to_go).cache()
    dataset = dataset.interleave(
        lambda episode: rlds.transformations.batch(
            episode["steps"], size=sequence_length, shift=1, drop_remainder=False
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=False,
    )

    dataset = dataset.map(functools.partial(pad_steps, max_len=sequence_length))
    return dataset
