import dataclasses


@dataclasses.dataclass
class Config:
    env_name: str = "gymnasium:HalfCheetah-v4"
    max_num_actor_steps: int = int(1e6)
    num_eval_episodes: int = 10
    eval_every: int = 10000
    hidden_dims = (256, 256)
    seed: int = 0

    policy_lr: float = 3e-4
    critic_lr: float = 3e-4
    temperature_lr: float = 3e-4

    batch_size: int = 256
    max_replay_size: int = int(1e6)
    min_replay_size: int = 5000
    utd_ratio: int = 1
    discount: float = 0.99

    log_to_wandb: bool = False


def get_config():
    return Config()
