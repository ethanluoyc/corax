# type: ignore
"""Example running TD-MPC on dm_control.
See configs/ for configurations for other environments.

"""
import copy
import functools
import os

import dm_env_wrappers as wrappers
import optax
import rlax
import rlds
import tensorflow as tf
import tensorflow_datasets as tfds
import tree
from absl import app
from absl import flags
from absl import logging
from ml_collections import config_flags

from baselines import experiment_utils
from baselines import experiments
from baselines.tdmpc import rewarder as rewarder_lib
from corax import types
from corax.agents.jax import tdmpc
from corax.jax import variable_utils

os.environ["MUJOCO_GL"] = "egl"

_CONFIG = config_flags.DEFINE_config_file("config", None)
_WORKDIR = flags.DEFINE_string("workdir", None, "Where to store artifacts")
flags.mark_flag_as_required("config")


def make_logger_factory(config):
    wandb_kwargs = dict(
        name=config.wandb_name,
        entity=config.wandb_entity,
        project=config.wandb_project,
        config=config.to_dict(),
        tags=[config.task],
    )

    logger_factory = experiment_utils.LoggerFactory(
        log_to_wandb=config.get("use_wandb", False),
        workdir=_WORKDIR.value,
        learner_time_delta=10.0,
        wandb_kwargs=wandb_kwargs,
    )

    return logger_factory


def make_environment_factory(config):
    def environment_factory(seed):
        # pylint: disable=import-outside-toplevel
        from dm_control import suite

        domain, task = config.task.replace("-", "_").split("_", 1)
        domain = dict(cup="ball_in_cup").get(domain, domain)
        assert (domain, task) in suite.ALL_TASKS
        env = suite.load(
            domain, task, task_kwargs={"random": seed}, visualize_reward=False
        )
        env = wrappers.ConcatObservationWrapper(env)
        env = wrappers.ActionRepeatWrapper(env, config.action_repeat)
        env = wrappers.CanonicalSpecWrapper(env, clip=True)
        env = wrappers.SinglePrecisionWrapper(env)
        return env

    return environment_factory


def _make_schedule(config):
    return getattr(optax, config.name)(**config.kwargs)


# Create function that computes episode return from rlds dataset
EPISODE_RETURN = "episode_return"


@tf.function
def _add_episode_return(episode, output_key=EPISODE_RETURN):
    """Computes the return of the episode up to the 'placed' tag."""
    episode = copy.copy(episode)
    # Truncate the episode after the placed tag.
    episode[output_key] = rlds.transformations.sum_dataset(
        episode[rlds.STEPS], lambda step: step[rlds.REWARD]
    )
    return episode


def add_episode_return(episode_dataset):
    return episode_dataset.map(
        _add_episode_return, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )


def filter_by_min_return(episode_dataset, min_return):
    return episode_dataset.filter(lambda episode: episode[EPISODE_RETURN] > min_return)


def concat_observations(episode_dataset):
    def _concat(step):
        flat_observations = tree.flatten(step[rlds.OBSERVATION])
        step[rlds.OBSERVATION] = tf.concat(flat_observations, axis=-1)
        return step

    return rlds.transformations.map_nested_steps(episode_dataset, _concat)


def transform_rlds_episode_to_otil_demo(episode_dataset: tf.data.Dataset):
    demos = []
    for episode in episode_dataset:
        steps_dataset = episode["steps"]
        episode_length = int(rlds.transformations.episode_length(steps_dataset))
        steps = steps_dataset.batch(episode_length).get_single_element()
        steps = tree.map_structure(lambda x: x.numpy(), steps)

        trajectory = []

        def get_nth_item(n, nest):
            return tree.map_structure(lambda x: x[n], nest)

        # Ignore terminal observation
        for t in range(episode_length):
            trajectory.append(
                types.Transition(
                    observation=get_nth_item(t, steps["observation"]),
                    action=get_nth_item(t, steps["action"]),
                    reward=(),
                    discount=(),
                    next_observation=(),
                )
            )
        demos.append(trajectory)
    return demos


def make_experiment_config(config):
    environment_factory = make_environment_factory(config)
    logger_factory = make_logger_factory(config)
    networks_factory = functools.partial(
        tdmpc.make_networks,
        latent_size=config.latent_dim,
        encoder_hidden_size=config.enc_dim,
        mlp_hidden_size=config.mlp_dim,
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(config.grad_clip_norm),
        optax.adam(config.lr),
    )

    def rewarder_factory(variable_source, spec):
        dataset = tfds.load(
            f"rlu_control_suite/{config.task.replace('-', '_')}", split="train"
        )
        min_return = 900
        transforms = [
            add_episode_return,
            functools.partial(filter_by_min_return, min_return=min_return),
            concat_observations,
        ]
        for transform in transforms:
            dataset = transform(dataset)

        rewarder_config = config.rewarder

        dataset = dataset.take(rewarder_config.num_demos)
        demonstrations = transform_rlds_episode_to_otil_demo(dataset)

        if rewarder_config.use_encoder:
            preprocessor = rewarder_lib.MeanStdPreprocessor(
                spec.observations, spec.actions, use_actions=False
            )
            networks = networks_factory(spec)

            def encoder_fn(params, obs):
                return networks.h(params, obs)

            preprocessor = rewarder_lib.EncoderPreprocessor(
                encoder_fn, use_actions=False
            )

            variable_client = variable_utils.VariableClient(variable_source, "policy")
            preprocessor_update_period = rewarder_config.update_period
        else:
            preprocessor = rewarder_lib.MeanStdPreprocessor(
                spec.observations, spec.actions, use_actions=False
            )
            preprocessor_update_period = 1
            variable_client = None

        rewarder = rewarder_lib.ROTRewarder(
            demonstrations=demonstrations,
            episode_length=config.episode_length,
            preprocessor=preprocessor,
            preprocessor_update_period=preprocessor_update_period,
            variable_client=variable_client,
        )

        return rewarder

    std_schedule = _make_schedule(config.std_schedule)
    horizon_schedule = _make_schedule(config.horizon_schedule)
    builder = tdmpc.TDMPCBuilder(
        tdmpc.TDMPCConfig(
            std_schedule=std_schedule,
            horizon_schedule=horizon_schedule,
            optimizer=optimizer,
            batch_size=config.batch_size,
            # One update per actor step.
            samples_per_insert=config.batch_size,
            samples_per_insert_tolerance_rate=0.1,
            max_replay_size=config.max_buffer_size,
            variable_update_period=config.variable_update_period,
            importance_sampling_exponent=config.per_alpha,
            priority_exponent=config.per_beta,
            discount=config.discount,
            num_trajectories=config.num_samples,
            min_std=config.min_std,
            temperature=config.temperature,
            momentum=config.momentum,
            num_elites=config.num_elites,
            num_iterations=config.iterations,
            critic_update_rate=config.tau,
            min_replay_size=config.seed_steps,
            policy_trajectory_fraction=config.mixture_coef,
            horizon=config.horizon,
            value_tx_pair=rlax.SIGNED_LOGP1_PAIR,
            consistency_loss_scale=config.consistency_coef,
            reward_loss_scale=config.reward_coef,
            value_loss_scale=config.value_coef,
            rho=config.rho,
        )
    )

    online_experiment = experiments.ExperimentConfig(
        builder=builder,
        network_factory=networks_factory,
        environment_factory=environment_factory,
        max_num_actor_steps=config.train_steps,
        seed=config.seed,
        logger_factory=logger_factory,
        checkpointing=None,
    )
    return experiments.ImitationExperimentConfig(online_experiment, rewarder_factory)


def main(_):
    tf.config.set_visible_devices([], "GPU")
    config = _CONFIG.value
    logging.info("Config:\n%s", config)
    experiment_config = make_experiment_config(config)
    experiments.run_imitation_experiment(
        experiment_config,
        eval_every=config.eval_freq,
        num_eval_episodes=config.eval_episodes,
    )


if __name__ == "__main__":
    app.run(main)
