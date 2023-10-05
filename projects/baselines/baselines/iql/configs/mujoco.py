import itertools

from baselines.iql import base_config
from baselines.iql.configs import sweeps

_NUM_SEEDS = 10


def get_config():
    return base_config.get_base_config()


def sweep_datasets():
    parameters = []
    for seed, env_name in itertools.product(
        range(_NUM_SEEDS),
        sweeps.sweep_d4rl_locomotion(),
    ):
        parameters.append({"config.seed": seed, "config.env_name": env_name})
    return parameters
