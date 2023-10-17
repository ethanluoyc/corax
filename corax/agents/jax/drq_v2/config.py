"""Configuration for DrQV2."""
import dataclasses
from typing import Optional

from corax.adders import reverb as adders_reverb
from corax.agents.jax.drq_v2 import augmentations


@dataclasses.dataclass
class DrQV2Config:
    """Configuration parameters for DrQ."""

    augmentation: augmentations.DataAugmentation = (
        augmentations.batched_random_shift_aug
    )

    min_replay_size: int = 2_000
    max_replay_size: int = 1_000_000
    replay_table_name: str = adders_reverb.DEFAULT_PRIORITY_TABLE

    discount: float = 0.99
    batch_size: int = 256
    n_step: int = 3

    critic_q_soft_update_rate: float = 0.01
    learning_rate: float = 1e-4
    noise_clip: float = 0.3
    sigma: float = 0.2
    bc_alpha: Optional[float] = None

    samples_per_insert: float = 128.0
    samples_per_insert_tolerance_rate: float = 0.1

    num_sgd_steps_per_step: int = 1
    variable_update_period: int = 1
