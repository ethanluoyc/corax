from absl import app
from absl import flags
import jax
from ml_collections import config_flags
import tensorflow as tf

from baselines import d4rl_utils
from baselines import experiment_utils
import corax
from corax import environment_loop
from corax.agents.jax import decision_transformer
from corax.agents.jax.decision_transformer import dataset as dataset_lib
from corax.datasets import tfds
from corax.jax import utils as jax_utils

_CONFIG = config_flags.DEFINE_config_file("config", None)
flags.mark_flag_as_required("config")


def main(_):
    config = _CONFIG.value

    env_name = config["env"]
    max_ep_len = config.max_ep_len
    scale = config.scale  # normalization for rewards/returns
    seed = config.seed

    if env_name == "hopper":
        eval_env_name = "Hopper-v3"
        target_return = 3600  # evaluation conditioning targets
    elif env_name == "halfcheetah":
        eval_env_name = "HalfCheetah-v3"
        target_return = 12000
    elif env_name == "walker2d":
        eval_env_name = "Walker2d-v3"
        target_return = 5000
    else:
        raise NotImplementedError

    dataset_name = f"{env_name}-{config['dataset']}-v2"
    env = d4rl_utils.load_d4rl_environment(eval_env_name, seed)

    K = config.K
    batch_size = config.batch_size
    max_num_learner_steps = config.num_steps
    episode_dataset = tfds.load_tfds_dataset(d4rl_utils.get_tfds_name(dataset_name))
    observation_mean_std = dataset_lib.get_observation_mean_std(episode_dataset)

    def make_dataset_iterator(key):
        del key
        dataset = (
            dataset_lib.transform_decision_transformer_input(
                episode_dataset,
                sequence_length=K,
                scale=scale,
                observation_mean_std=observation_mean_std,
            )
            .shuffle(int(1e6))
            .repeat()
            .batch(batch_size)
            .as_numpy_iterator()
        )
        return jax_utils.device_put(dataset, jax.local_devices()[0])

    network_factory = lambda spec: decision_transformer.make_gym_networks(
        spec=spec, episode_length=max_ep_len, **config.network_config
    )

    dt_config = decision_transformer.DecisionTransformerConfig(
        context_length=K,
        target_return=target_return,
        return_scale=scale,
        mode=config.mode,
        learning_rate=config.learning_rate,
        warmup_steps=config.warmup_steps,
        grad_norm_clipping=0.25,
        weight_decay=config.weight_decay,
    )

    builder = decision_transformer.DecisionTransformerBuilder(
        dt_config,
        observation_mean_std,
        max_num_learner_steps=max_num_learner_steps,
        actor_device="gpu",
    )

    logger_factory = experiment_utils.LoggerFactory(log_to_wandb=config.log_to_wandb)
    random_key = jax.random.PRNGKey(config.seed)
    environment = env

    spec = corax.make_environment_spec(environment)
    dataset = make_dataset_iterator(random_key)

    networks = network_factory(spec)
    learner = builder.make_learner(
        random_key,
        networks,
        dataset,
        logger_factory,  # type: ignore
        spec,
    )
    learner.step()
    policy = builder.make_policy(networks, spec, True)
    actor = builder.make_actor(random_key, policy, spec, learner)

    eval_loop = environment_loop.EnvironmentLoop(environment, actor)  # type: ignore

    steps = 0
    while steps < config.num_steps:
        learner_steps = min(config.eval_every, max_num_learner_steps - steps)
        for _ in range(learner_steps):
            learner.step()
        if eval_loop:
            eval_loop.run(num_episodes=config.num_eval_episodes)
            steps += learner_steps


if __name__ == "__main__":
    tf.config.set_visible_devices([], "GPU")
    app.run(main)
