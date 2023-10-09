import ml_collections


def get_config():
    config = ml_collections.ConfigDict()
    config.actor_lr = 3e-4
    config.critic_lr = 3e-4

    config.hidden_dims = (256, 256)
    config.num_qs = 10
    config.num_min_qs = 2

    config.discount = 0.99

    config.tau = 0.005

    config.temp_lr = 3e-4

    config.init_temperature = 1.0
    config.backup_entropy = True
    config.critic_layer_norm = True

    config.log_to_wandb = False
    config.env_name = "halfcheetah-expert-v2"
    config.offline_ratio = 0.5
    config.seed = 42
    config.eval_episodes = 10
    config.eval_interval = 10000
    config.batch_size = 256
    config.max_steps = int(250000)
    config.utd_ratio = 20
    config.start_training = int(1e4)

    return config
