import time
from typing import Optional, Sequence

from absl import logging
import jax
import numpy as np
import optax
import rlds
import tensorflow as tf

from baselines import d4rl_utils
from baselines import experiment_utils
from baselines import rlds_utils
from baselines.iql import base_config
import corax
from corax.agents.jax import actor_core
from corax.agents.jax import actors as actors_lib
from corax.agents.jax import iql
from corax.agents.jax import learners as learners_lib
from corax.datasets import tfds
from corax.jax import utils as jax_utils
from corax.jax import variable_utils
from corax.utils import counting
from corax.utils import loggers
from corax.utils import observers as observers_lib


def make_d4rl_transition_dataset(
    d4rl_name: str, batch_size: int, seed: int, num_episodes: Optional[int] = None
):
    dataset = tfds.load_tfds_dataset(
        d4rl_utils.get_tfds_name(d4rl_name), num_episodes=num_episodes, download=False
    )
    start_time = time.time()
    if "antmaze" in d4rl_name:
        reward_scale = 1.0
        reward_bias = -1.0
    else:
        episode_returns = dataset.map(
            lambda episode: rlds_utils.compute_episode_return(episode[rlds.STEPS]),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        episode_returns = (
            episode_returns.batch(int(dataset.cardinality())).get_single_element().numpy()  # type: ignore
        )
        max_episode_return = np.max(episode_returns)
        min_episode_return = np.min(episode_returns)

        reward_scale = 1000.0 / (max_episode_return - min_episode_return)
        reward_bias = 0.0

    dataset = rlds.transformations.map_nested_steps(
        dataset, lambda step: rlds_utils.rescale_reward(step, reward_scale, reward_bias)
    )
    dataset = dataset.map(
        rlds_utils.skip_truncated_last_step, num_parallel_calls=tf.data.AUTOTUNE
    )

    transitions = rlds_utils.episodes_to_transitions_dataset(
        dataset,
        cycle_length=16,
        block_length=16,
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=True,
    )

    transitions = transitions.cache()

    iterator = tfds.JaxInMemoryRandomSampleIterator(
        transitions, jax.random.PRNGKey(seed), batch_size
    )

    elapsed_time = time.time() - start_time
    logging.info("Created dataset in %.2f seconds", elapsed_time)
    return iterator


def run_offline_experiment(
    environment_factory,
    dataset_factory,
    network_factory,
    learner_core_factory,
    policy_factory,
    *,
    seed: int,
    logger_factory: loggers.LoggerFactory,
    max_num_learner_steps: int,
    num_eval_episodes: int,
    eval_every: int,
    num_sgd_steps_per_step: int,
    observers: Sequence[observers_lib.EnvLoopObserver] = (),
):
    parent_counter = counting.Counter(time_delta=0.0)

    learner_counter = counting.Counter(parent_counter, prefix="learner", time_delta=0.0)

    environment = environment_factory(seed)
    environment_spec = corax.make_environment_spec(environment)

    random_key = jax.random.PRNGKey(seed)
    learner_key, dataset_key, evaluator_key = jax.random.split(random_key, 3)

    dataset = dataset_factory(dataset_key)

    networks = network_factory(environment_spec)
    learner_core = learner_core_factory(networks)
    learner = learners_lib.DefaultJaxLearner(
        learner_core,
        learner_key,
        dataset,
        counter=learner_counter,
        logger=logger_factory("learner", learner_counter.get_steps_key()),
        num_sgd_steps_per_step=num_sgd_steps_per_step,
    )

    eval_policy = policy_factory(networks, environment_spec, True)
    eval_counter = counting.Counter(parent_counter, prefix="evaluator", time_delta=0.0)
    eval_actor = actors_lib.GenericActor(
        eval_policy,
        evaluator_key,
        variable_utils.VariableClient(learner, "policy", device="cpu"),
        per_episode_update=True,
        backend="cpu",
    )

    eval_loop = corax.EnvironmentLoop(
        environment,
        eval_actor,
        logger=logger_factory("evaluator", eval_counter.get_steps_key()),
        observers=observers,
        counter=eval_counter,
    )

    learner_counter.increment(steps=0)
    eval_loop.run(num_episodes=num_eval_episodes)
    steps = 0
    while steps < max_num_learner_steps:
        learner_steps = min(eval_every, max_num_learner_steps - steps)
        for _ in range(learner_steps):
            learner.step()
        if eval_loop:
            eval_loop.run(num_episodes=num_eval_episodes)
            steps += learner_steps


def train_and_evaluate(config: base_config.Config, workdir: Optional[str] = None):
    del workdir

    if config.num_sgd_steps_per_step > 1:
        assert config.max_num_learner_steps % config.num_sgd_steps_per_step == 0
        assert config.eval_every % config.num_sgd_steps_per_step == 0

    logger_factory = experiment_utils.LoggerFactory(
        log_to_wandb=config.log_to_wandb,
        wandb_kwargs={"project": "corax", "config": config},
        learner_time_delta=1.0,
        async_learner_logger=True,
        evaluator_time_delta=0.0,
    )

    def environment_factory(seed: int):
        return d4rl_utils.load_d4rl_environment(config.env_name, seed=seed)

    def dataset_factory(key):
        return make_d4rl_transition_dataset(
            config.env_name,
            config.batch_size * config.num_sgd_steps_per_step,
            seed=jax_utils.sample_uint32(key),
            num_episodes=config.num_episodes,
        )

    def network_factory(spec):
        return iql.make_networks(spec, hidden_dims=config.hidden_dims)

    def learner_core_factory(networks):
        if config.use_cosine_decay:
            schedule_fn = optax.cosine_decay_schedule(
                -config.policy_lr, config.max_num_learner_steps
            )
            policy_optimizer = optax.chain(
                optax.scale_by_adam(),
                optax.scale_by_schedule(schedule_fn),
            )
        else:
            policy_optimizer = optax.adam(learning_rate=config.policy_lr)

        return iql.IQLLearnerCore(
            networks,
            policy_optimizer=policy_optimizer,
            critic_optimizer=optax.adam(config.critic_lr),
            value_optimizer=optax.adam(config.value_lr),
            discount=config.discount,
            tau=config.tau,
            expectile=config.expectile,
            temperature=config.temperature,
        )

    def policy_factory(networks, spec, evaluation):
        return actor_core.batched_feed_forward_to_actor_core(
            iql.apply_policy_and_sample(networks, spec.actions, eval_mode=evaluation)
        )

    run_offline_experiment(
        environment_factory,
        dataset_factory,
        network_factory,
        learner_core_factory,
        policy_factory,
        logger_factory=logger_factory,
        seed=config.seed,
        max_num_learner_steps=(
            config.max_num_learner_steps // config.num_sgd_steps_per_step
        ),
        eval_every=config.eval_every // config.num_sgd_steps_per_step,
        num_eval_episodes=config.num_eval_episodes,
        num_sgd_steps_per_step=config.num_sgd_steps_per_step,
        observers=[d4rl_utils.D4RLScoreObserver(config.env_name)],
    )
