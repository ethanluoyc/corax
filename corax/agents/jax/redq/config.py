import dataclasses
from typing import Optional

from corax.adders import reverb as adders_reverb


@dataclasses.dataclass
class REDQConfig:
    """Configuration options for REDQ."""

    # Optimizer options
    batch_size: int = 256
    actor_learning_rate: float = 3e-4
    critic_learning_rate: float = 3e-4
    temperature_learning_rate: float = 3e-4

    init_temperature: float = 1.0
    backup_entropy: bool = True
    discount: float = 0.99
    n_step: int = 1
    target_entropy: Optional[float] = None
    # Target smoothing coefficient.
    tau: float = 0.005

    # Replay options
    min_replay_size: int = 10000
    max_replay_size: int = 1000000
    replay_table_name: str = adders_reverb.DEFAULT_PRIORITY_TABLE

    samples_per_insert: float = 256
    samples_per_insert_tolerance_rate: float = 0.1

    offline_fraction: float = 0.5

    # How many gradient updates to perform per step.
    num_sgd_steps_per_step: int = 1

    # Reward scaling options
    # reward may be scaled as reward_scale * reward + reward_bias
    reward_scale: float = 1.0
    reward_bias: float = 0.0
