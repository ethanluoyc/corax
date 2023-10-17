from absl import app
from absl import flags
import jax
from ml_collections import config_flags

from baselines import d4rl_utils
from baselines import experiment_utils
from corax.agents.jax import redq
from corax.datasets import tfds
from corax.jax import experiments

_CONFIG = config_flags.DEFINE_config_file(
    "config", None, "File path to the training configuration."
)
flags.mark_flag_as_required("config")

_WORKDIR = flags.DEFINE_string("workdir", None, "")


def _get_demonstration_dataset_factory(name, seed):
    def make_demonstrations(batch_size):
        tfds_name = d4rl_utils.get_tfds_name(name)
        transitions = tfds.get_tfds_dataset(tfds_name)
        with jax.default_device(jax.devices("cpu")[0]):  # type: ignore
            # NOTE(yl): The yield from is necessary to
            # for some reason returning the iterator directly causes
            # sampling to be much slower.
            yield from tfds.JaxInMemoryRandomSampleIterator(
                transitions, jax.random.PRNGKey(seed), batch_size
            )

    return make_demonstrations


def main(_):
    config = _CONFIG.value
    env_name = config.env_name

    seed = config.seed

    network_factory = lambda environment_spec: redq.make_networks(
        environment_spec,
        hidden_sizes=config.hidden_dims,
        num_qs=config.num_qs,
        num_min_qs=config.num_min_qs,
        critic_layer_norm=config.critic_layer_norm,
    )

    redq_config = redq.REDQConfig(
        actor_learning_rate=config.actor_lr,
        critic_learning_rate=config.critic_lr,
        temperature_learning_rate=config.temp_lr,
        init_temperature=config.init_temperature,
        backup_entropy=config.backup_entropy,
        discount=config.discount,
        n_step=1,
        target_entropy=None,  # Automatic entropy tuning.
        # Target smoothing coefficient.
        tau=config.tau,
        max_replay_size=config.max_steps,
        batch_size=config.batch_size,
        min_replay_size=config.start_training,
        # Convert from UTD to SPI
        # In the pure online setting, SPI = batch_size is equivalent to UTD = 1
        # For RLPD, SPI = online_batch_size * UTD = (1 - offline_ratio) * batch_size * UTD
        samples_per_insert=(
            config.utd_ratio * config.batch_size * (1 - config.offline_ratio)
        ),
        # Effectively equivalent to UTD
        num_sgd_steps_per_step=config.utd_ratio,
        offline_fraction=config.offline_ratio,
        reward_bias=-1 if "antmaze" in env_name else 0,
    )

    builder = redq.REDQBuilder(
        redq_config,
        make_demonstrations=_get_demonstration_dataset_factory(env_name, seed=seed),
    )

    environment_factory = lambda seed: d4rl_utils.load_d4rl_environment(env_name, seed)

    logger_factory = experiment_utils.LoggerFactory(
        workdir=_WORKDIR.value,
        log_to_wandb=config.log_to_wandb,
        wandb_kwargs={
            "project": "rlpd",
            "config": config,
        },
        evaluator_time_delta=0.01,
        add_uid=True,
    )

    experiment = experiments.ExperimentConfig(
        builder=builder,
        environment_factory=environment_factory,
        network_factory=network_factory,
        seed=seed,
        max_num_actor_steps=config.max_steps,
        logger_factory=logger_factory,  # type: ignore
        checkpointing=None,
    )
    experiments.run_experiment(
        experiment=experiment,
        eval_every=config.eval_interval,
        num_eval_episodes=config.eval_episodes,
    )


if __name__ == "__main__":
    app.run(main)
