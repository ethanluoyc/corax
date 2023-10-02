"""Data augmentations used in DrQ."""
from typing import Callable

import jax
import jax.numpy as jnp

from corax import types
from corax.jax import types as jax_types

DataAugmentation = Callable[[jax_types.PRNGKey, types.NestedArray], types.NestedArray]


# From https://github.com/ikostrikov/jax-rl/blob/main/jax_rl/agents/drq/augmentations.py
def random_crop(key: jax_types.PRNGKey, img, padding):
    crop_from = jax.random.randint(key, (2,), 0, 2 * padding + 1)
    crop_from = jnp.concatenate([crop_from, jnp.zeros((1,), dtype=jnp.int32)])
    padded_img = jnp.pad(
        img, ((padding, padding), (padding, padding), (0, 0)), mode="edge"
    )
    return jax.lax.dynamic_slice(padded_img, crop_from, img.shape)


def batched_random_crop(key, imgs, padding=4):
    keys = jax.random.split(key, imgs.shape[0])
    return jax.vmap(random_crop, (0, 0, None))(keys, imgs, padding)


def random_shift_aug(key, img, padding):
    """Perform the random shift augmentation used in DrQ-V2"""
    # Pad the image
    padded_img = jnp.pad(
        img, ((padding, padding), (padding, padding), (0, 0)), mode="edge"
    )
    # Shift and Resample
    shifts = jax.random.randint(key, (2,), 0, 2 * padding + 1)
    return jax.image.scale_and_translate(
        padded_img,
        img.shape,
        spatial_dims=(0, 1),
        scale=jnp.ones(2, dtype=jnp.float32),
        translation=-shifts,
        method="bilinear",
    ).astype(img.dtype)


def batched_random_shift_aug(key, img, padding=4):
    keys = jax.random.split(key, img.shape[0])
    return jax.vmap(random_shift_aug, (0, 0, None))(keys, img, padding)
