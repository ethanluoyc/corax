import ml_collections


def get_config():
    config = ml_collections.ConfigDict()
    config.dataset = "medium"
    config.env = "hopper"
    config.mode = "normal"
    config.K = 20
    config.pct_traj = 1.0
    config.batch_size = 64
    config.network_config = dict(
        hidden_size=128,
        num_layers=3,
        num_heads=1,
        dropout_rate=0.1,
    )
    config.seed = 0
    config.learning_rate = 1e-4
    config.weight_decay = 1e-4
    config.warmup_steps = 10000
    config.num_eval_episodes = 10
    config.num_steps = int(1e5)
    config.eval_every = int(1e4)
    config.log_to_wandb = False
    config.max_ep_len = 1000
    config.scale = 1000.0
    return config
