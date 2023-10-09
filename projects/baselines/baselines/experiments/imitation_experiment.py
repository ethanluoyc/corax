# type: ignore
"""A training loop for IL agent."""

import dataclasses
from typing import Callable

import jax
import reverb

import corax
from baselines import experiments
from baselines.experiments import config
from baselines.experiments import savers
from baselines.experiments.imitation import imitation_loop
from baselines.experiments.run_experiment import _disable_insert_blocking
from baselines.experiments.run_experiment import _LearningActor
from corax import specs as env_specs
from corax.jax import utils
from corax.utils import counting

Config = experiments.ExperimentConfig
RewarderFactory = Callable[
    [corax.VariableSource, env_specs.EnvironmentSpec], imitation_loop.EpisodeRewarder
]


@dataclasses.dataclass(frozen=True)
class ImitationExperimentConfig:
    config: experiments.ExperimentConfig
    rewarder_factory: RewarderFactory


def run_imitation_experiment(
    imitation_experiment: ImitationExperimentConfig,
    eval_every: int,
    num_eval_episodes: int,
):
    rewarder_factory = imitation_experiment.rewarder_factory
    experiment = imitation_experiment.config
    # Create an environment, grab the spec, and use it to create networks.
    key = jax.random.PRNGKey(experiment.seed)
    # key_train_env, key_eval, key_actor, key_eval_actor = jax.random.split(key, 4)
    environment_key, key = jax.random.split(key)
    environment = experiment.environment_factory(utils.sample_uint32(environment_key))
    environment_spec = experiment.environment_spec or corax.make_environment_spec(
        environment
    )
    networks = experiment.network_factory(environment_spec)
    policy = config.make_policy(
        experiment=experiment,
        networks=networks,
        environment_spec=environment_spec,
        evaluation=False,
    )

    replay_tables = experiment.builder.make_replay_tables(environment_spec, policy)

    # Disable blocking of inserts by tables' rate limiters, as this function
    # executes learning (sampling from the table) and data generation
    # (inserting into the table) sequentially from the same thread
    # which could result in blocked insert making the algorithm hang.
    replay_tables, rate_limiters_max_diff = _disable_insert_blocking(replay_tables)

    replay_server = reverb.Server(replay_tables, port=None)
    replay_client = reverb.Client(f"localhost:{replay_server.port}")

    # Parent counter allows to share step counts between train and eval loops and
    # the learner, so that it is possible to plot for example evaluator's return
    # value as a function of the number of training episodes.
    parent_counter = counting.Counter(time_delta=0.0)

    # Create actor, and learner for generating, storing, and consuming
    # data respectively.
    dataset = experiment.builder.make_dataset_iterator(replay_client)
    # We always use prefetch, as it provides an iterator with additional
    # 'ready' method.
    dataset = utils.prefetch(dataset, buffer_size=1)
    learner_key, key = jax.random.split(key)
    learner_counter = counting.Counter(parent_counter, prefix="learner", time_delta=0.0)

    learner = experiment.builder.make_learner(
        random_key=learner_key,
        networks=networks,
        dataset=dataset,
        logger_fn=experiment.logger_factory,
        environment_spec=environment_spec,
        replay_client=replay_client,
        counter=learner_counter,
    )

    rewarder = rewarder_factory(learner, environment_spec)
    actor_key, key = jax.random.split(key)
    # adder is not used in the actor but instead passed to the imitation loop.
    actor = experiment.builder.make_actor(
        actor_key, policy, environment_spec, variable_source=learner, adder=None
    )

    checkpointer = None
    if experiment.checkpointing is not None:
        checkpointing = experiment.checkpointing
        checkpointer = savers.Checkpointer(
            objects_to_save={"learner": learner, "counter": parent_counter},
            time_delta_minutes=checkpointing.time_delta_minutes,
            directory=checkpointing.directory,
            add_uid=checkpointing.add_uid,
            max_to_keep=checkpointing.max_to_keep,
            keep_checkpoint_every_n_hours=checkpointing.keep_checkpoint_every_n_hours,
            checkpoint_ttl_seconds=checkpointing.checkpoint_ttl_seconds,
        )

    # Replace the actor with a LearningActor. This makes sure that every time
    # that `update` is called on the actor it checks to see whether there is
    # any new data to learn from and if so it runs a learner step. The rate
    # at which new data is released is controlled by the replay table's
    # rate_limiter which is created by the builder.make_replay_tables call above.
    actor = _LearningActor(
        actor, learner, dataset, replay_tables, rate_limiters_max_diff, checkpointer
    )

    parent_counter = counting.Counter(time_delta=0.0)
    train_counter = counting.Counter(parent_counter, "actor", time_delta=0.0)
    train_logger = experiment.logger_factory("actor", train_counter.get_steps_key(), 0)

    adder = experiment.builder.make_adder(replay_client, environment_spec, policy)
    train_loop = imitation_loop.ImitationEnvironmentLoop(
        environment,
        actor,
        adder=adder,
        rewarder=rewarder,
        counter=train_counter,
        logger=train_logger,
        observers=experiment.observers,
    )

    eval_loop = None
    if num_eval_episodes > 0:
        # Create the evaluation actor and loop.
        eval_policy = config.make_policy(
            experiment=experiment,
            networks=networks,
            environment_spec=environment_spec,
            evaluation=True,
        )
        eval_actor_key, key = jax.random.split(key)
        eval_actor = experiment.builder.make_actor(
            random_key=eval_actor_key,
            policy=eval_policy,
            environment_spec=environment_spec,
            variable_source=learner,
        )
        eval_environment_key, key = jax.random.split(key)
        if experiment.eval_environment_factory is None:
            eval_env = experiment.environment_factory(
                utils.sample_uint32(eval_environment_key)
            )
        else:
            eval_env = experiment.eval_environment_factory(
                utils.sample_uint32(eval_environment_key)
            )

        eval_counter = counting.Counter(parent_counter, "evaluator", time_delta=0.0)
        eval_logger = experiment.logger_factory(
            "evaluator", eval_counter.get_steps_key(), 0
        )
        eval_loop = imitation_loop.ImitationEnvironmentLoop(
            eval_env,
            eval_actor,
            adder=None,
            rewarder=rewarder,
            counter=eval_counter,
            logger=eval_logger,
            should_update_rewarder=False,
            observers=experiment.observers,
        )

    steps = 0
    while steps < experiment.max_num_actor_steps:
        if eval_loop:
            eval_loop.run(num_episodes=num_eval_episodes)
        train_loop.run(num_steps=eval_every)
        steps += eval_every
    if eval_loop:
        eval_loop.run(num_episodes=num_eval_episodes)
        eval_env.close()  # type: ignore
    environment.close()
