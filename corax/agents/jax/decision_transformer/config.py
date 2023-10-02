import dataclasses
from typing import Any, Callable

import jax.numpy as jnp
import tree


def exclude_bias_and_normalizers(params):
    def predicate(path, value: jnp.ndarray) -> jnp.ndarray:
        del value
        return path[-1] == "b" or "norm" in path[-2] or path[-1] == "embeddings"  # type: ignore

    return tree.map_structure_with_path(predicate, params)


@dataclasses.dataclass
class DecisionTransformerConfig:
    # DT parameters
    context_length: int
    target_return: float
    return_scale: float
    mode: str = "normal"

    # Optimizer parameters
    learning_rate: float = 1e-4
    warmup_steps: int = 10000
    grad_norm_clipping: float = 0.25
    weight_decay: float = 1e-4
    weight_decay_mask: Callable[[Any], Any] = exclude_bias_and_normalizers

    def __post_init__(self):
        assert self.mode in [
            "normal",
            "delayed",
        ], "mode must be either 'normal' or 'delayed'"
