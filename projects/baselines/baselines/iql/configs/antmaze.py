import itertools

from baselines.iql import base_config
from baselines.iql.configs import sweeps


def get_config():
    config = base_config.get_base_config()
    config.env_name = "antmaze-umaze-v2"

    config.eval_every = 50000
    config.num_eval_episodes = 100

    config.expectile = 0.9  # The actual tau for expectiles.
    config.temperature = 10.0

    config.tau = 0.005  # For soft target updates.

    return config


_NUM_SEEDS = 10


def sweep_datasets():
    parameters = []
    for seed, env_name in itertools.product(
        range(_NUM_SEEDS),
        sweeps.sweep_d4rl_antmaze(),
    ):
        parameters.append({"config.seed": seed, "config.env_name": env_name})
    return parameters
