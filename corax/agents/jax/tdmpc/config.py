"""Configuration for TD-MPC agents"""
import dataclasses

import optax
import rlax


@dataclasses.dataclass
class TDMPCConfig:
    std_schedule: optax.Schedule
    horizon_schedule: optax.Schedule

    optimizer: optax.GradientTransformation
    discount: float = 0.99

    # Training configuration
    critic_update_rate: float = 0.01  # tau
    value_tx_pair: rlax.TxPair = rlax.IDENTITY_PAIR
    variable_update_period: int = 1

    # Loss scales for model learning
    consistency_loss_scale: float = 2.0
    reward_loss_scale: float = 0.5
    value_loss_scale: float = 0.1

    rho: float = 0.5  # model prediction loss discount

    # Planning options
    horizon: int = 5  # Planning horizon
    num_trajectories: int = 512  # Number of trajectories to sample per planning step
    # Fraction of trajectories to sample from policy
    policy_trajectory_fraction: float = 0.05
    num_elites: int = 64  # Number of elite trajectories for CEM
    num_iterations: int = 6  # Number of CEM iterations
    min_std: float = 0.05  # Minimum noise to add to sample trajectories

    # parameter used for computing the normalized empirical estimate of
    # the value of the trajectories.
    temperature: float = 0.5
    momentum: float = 0.1  # Momentum for updating trajectories distribution parameters

    # Replay configuration
    samples_per_insert: float = 512.0
    samples_per_insert_tolerance_rate: float = 0.1
    # min_replay_size currently has dual meaning:
    # 1. The number of samples to insert into Reverb prior to learning begins
    # 2. The number of random exploration steps
    min_replay_size: int = 5000
    max_replay_size: int = int(1e6)
    batch_size: int = 512

    # Prioritized Experience Replay configuration
    # See https://arxiv.org/pdf/1511.05952.pdf
    importance_sampling_exponent: float = 0.6  # alpha
    priority_exponent: float = 0.4  # beta
