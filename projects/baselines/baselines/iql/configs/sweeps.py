import itertools


def sweep_d4rl_locomotion(version="v2"):
    sweep = []
    envs = ["halfcheetah", "hopper", "walker2d"]
    datasets = ["medium", "medium-replay", "medium-expert"]
    for env, dataset in itertools.product(envs, datasets):
        sweep.append(f"{env}-{dataset}-{version}")
    return sweep


def sweep_d4rl_antmaze(version="v2"):
    envs = ["antmaze"]
    datasets = [
        "umaze",
        "umaze-diverse",
        "medium-play",
        "medium-diverse",
        "large-play",
        "large-diverse",
    ]
    sweep = []
    for env, dataset in itertools.product(envs, datasets):
        sweep.append(f"{env}-{dataset}-{version}")
    return sweep
