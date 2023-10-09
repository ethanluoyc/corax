from ml_collections import config_dict


def get_config():
    config = config_dict.ConfigDict()
    config.dataset_name = "antmaze-medium-diverse-v2"
    config.seed = 0
    config.log_to_wandb = False

    config.policy_hidden_sizes = (256, 256)
    config.critic_hidden_sizes = (256, 256, 256, 256)

    config.policy_lr = 1e-4
    config.qf_lr = 3e-4
    config.batch_size = 256

    config.discount = 0.99

    config.reward_scale = 10.0
    config.reward_bias = -5

    config.initial_num_steps = 5000

    config.enable_calql = True
    config.cql_config = dict(
        cql_lagrange_threshold=0.8,
        cql_num_samples=10,
        max_target_backup=True,
        tau=5e-3,
    )

    # Offline training config
    config.offline_num_steps = int(1e6)
    config.offline_eval_every = int(5e4)
    config.offline_num_eval_episodes = 20

    # Offline training config
    config.mixing_ratio = 0.5
    config.online_utd_ratio = 1
    config.online_num_steps = int(1e6)
    config.online_eval_every = 2000
    config.online_num_eval_episodes = 20
    return config
