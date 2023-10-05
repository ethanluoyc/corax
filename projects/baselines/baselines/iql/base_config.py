import dataclasses
from typing import Optional, Sequence


@dataclasses.dataclass
class Config:
    env_name: str = "halfcheetah-medium-v2"
    num_episodes: Optional[int] = None
    batch_size: int = 256

    max_num_learner_steps: int = int(1e6)
    num_eval_episodes: int = 10
    eval_every: int = 5000
    seed: int = 0

    hidden_dims: Sequence[int] = (256, 256)

    use_cosine_decay: bool = True
    policy_lr: float = 3e-4
    critic_lr: float = 3e-4
    value_lr: float = 3e-4

    discount: float = 0.99
    tau: float = 5e-3
    expectile: float = 0.7
    temperature: float = 3.0

    log_to_wandb: bool = False
    num_sgd_steps_per_step: int = 1


def get_base_config():
    return Config()
