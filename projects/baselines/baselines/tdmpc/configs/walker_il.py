from baselines.tdmpc.configs import default


def get_config():
    config = default.get_config()
    config.task = "walker-walk"
    config.action_repeat = 2

    config.lr = 3e-4
    config.rewarder = dict(
        use_encoder=False,
        num_demos=10,
        update_period=10,
    )

    # config.reward_coef = 0.5
    # config.value_coef = 0.1
    # config.consistency_coef = 2

    return config
