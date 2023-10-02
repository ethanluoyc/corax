# type: ignore
from typing import Iterator

import absl.app
import absl.flags
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import optax
import reverb
import tensorflow as tf
import tree
from ml_collections import config_flags

import corax
from baselines import d4rl_evaluation
from baselines import d4rl_utils
from baselines import experiment_utils
from corax import datasets as acme_datasets
from corax import specs
from corax import types
from corax.adders import reverb as adders_reverb
from corax.agents.jax import actor_core
from corax.agents.jax import actors
from corax.agents.jax import calql
from corax.agents.jax.calql.adder import SparseReward
from corax.datasets import tfds
from corax.jax import utils as jax_utils
from corax.jax import variable_utils
from corax.utils import counting
from corax.utils import loggers


def _get_config():
    config = ml_collections.ConfigDict()
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


_CONFIG = config_flags.DEFINE_config_dict("config", _get_config(), lock_config=False)


@tf.function
def compute_return_to_go(rewards, discounts, gamma):
    rewards = tf.convert_to_tensor(rewards)
    discounts = tf.convert_to_tensor(discounts)

    def discounted_return_fn(acc, reward_discount):
        reward, discount = reward_discount
        return acc * discount * gamma + reward

    return tf.scan(
        fn=discounted_return_fn,
        elems=(rewards, discounts),
        reverse=True,
        initializer=tf.constant(0.0, dtype=rewards.dtype),
    )


@tf.function
def preprocess_episode(episode, negative_reward, reward_scale, reward_bias, gamma):
    steps = episode["steps"].batch(episode["steps"].cardinality()).get_single_element()

    observations = steps["observation"][:-1]
    next_observations = steps["observation"][1:]
    rewards = tf.cast(steps["reward"][:-1], tf.float64)
    actions = steps["action"][:-1]
    discounts = tf.cast(steps["discount"][:-1], tf.float64)

    rewards = rewards * reward_scale + reward_bias
    reward_negative = negative_reward * reward_scale + reward_bias
    gamma = tf.convert_to_tensor(gamma, dtype=rewards.dtype)

    if tf.reduce_all(rewards == reward_negative):
        return_to_go = tf.ones_like(rewards) * (reward_negative / (1 - gamma))
    else:
        return_to_go = compute_return_to_go(rewards, discounts, gamma)

    return types.Transition(
        observation=observations,
        action=actions,
        discount=discounts,
        reward=tf.cast(rewards, tf.float32),
        next_observation=next_observations,
        extras={"mc_return": return_to_go},
    )


def get_transitions_dataset(
    d4rl_name, negative_reward, reward_scale, reward_bias, gamma
):
    tfds_name = d4rl_utils.get_tfds_name(d4rl_name)
    dataset = tfds.load_tfds_dataset(tfds_name)
    tf_transitions = []
    for episode in dataset:
        converted_transitions = preprocess_episode(
            episode, negative_reward, reward_scale, reward_bias, gamma
        )
        tf_transitions.append(converted_transitions)

    transitions = tf.data.Dataset.from_tensor_slices(
        tree.map_structure(lambda *x: tf.concat(x, axis=0), *tf_transitions)
    )
    return transitions


def main(argv):
    del argv
    config = _CONFIG.value

    logger_factory = experiment_utils.LoggerFactory(
        workdir=None, log_to_wandb=config.log_to_wandb, evaluator_time_delta=0.0
    )

    reward_config = SparseReward(
        reward_scale=config.reward_scale,
        reward_bias=config.reward_bias,
        positive_reward=1,
        negative_reward=0,
    )

    transitions = get_transitions_dataset(
        config.dataset_name,
        reward_config.negative_reward,
        reward_config.reward_scale,
        reward_config.reward_bias,
        config.discount,
    )

    env = d4rl_utils.make_environment(config.dataset_name, config.seed)
    env_spec = corax.make_environment_spec(env)

    networks = calql.make_networks(
        env_spec,
        policy_hidden_sizes=config.policy_hidden_sizes,
        critic_hidden_sizes=config.critic_hidden_sizes,
    )

    eval_policy = actor_core.batched_feed_forward_to_actor_core(
        calql.apply_policy_and_sample(networks, eval_mode=True)
    )
    train_policy = actor_core.batched_feed_forward_to_actor_core(
        calql.apply_policy_and_sample(networks, eval_mode=False)
    )

    def make_offline_iterator(batch_size):
        return tfds.JaxInMemoryRandomSampleIterator(
            transitions, jax.random.PRNGKey(config.seed), batch_size
        )

    def make_offline_learner(
        networks: calql.CQLNetworks,
        random_key: jax.random.PRNGKeyArray,
        dataset: Iterator[types.Transition],
        env_spec: specs.EnvironmentSpec,
        logger: loggers.Logger,
        counter: counting.Counter,
    ):
        target_entropy = -np.prod(env_spec.actions.shape).item()
        return calql.CalQLLearner(
            config.batch_size,
            networks,
            random_key,
            dataset,
            policy_optimizer=optax.adam(config.policy_lr),
            critic_optimizer=optax.adam(config.qf_lr),
            reward_scale=1.0,  # Reward scaled in dataset iterator
            target_entropy=target_entropy,
            discount=config.discount,
            num_bc_iters=0,
            use_calql=config.enable_calql,
            **config.cql_config,
            logger=logger,
            counter=counter,
        )

    def make_actor(policy, random_key, variable_source, adder=None):
        return actors.GenericActor(
            policy,
            random_key,
            variable_utils.VariableClient(
                variable_source, "policy", update_period=1, device="cpu"
            ),
            adder=adder,
            per_episode_update=True,
        )

    def make_online_iterator(replay_client: reverb.Client):
        # mix offline and online buffer
        assert config.mixing_ratio >= 0.0
        online_batch_size = int(config.mixing_ratio * config.batch_size)
        offline_batch_size = config.batch_size - online_batch_size
        offline_iterator = make_offline_iterator(offline_batch_size)

        online_dataset = acme_datasets.make_reverb_dataset(
            table="priority_table",
            server_address=replay_client.server_address,
            num_parallel_calls=4,
            batch_size=online_batch_size,
            prefetch_size=1,
        ).as_numpy_iterator()

        while True:
            offline_batch = next(offline_iterator)
            offline_transitions = jax.device_put(offline_batch)
            online_transitions = jax.device_put(next(online_dataset).data)

            yield tree.map_structure(
                lambda x, y: jnp.concatenate([x, y]),
                offline_transitions,
                online_transitions,
            )

    def make_evaluator(random_key, learner):
        return d4rl_evaluation.D4RLEvaluator(
            lambda: env,
            make_actor(eval_policy, random_key, learner),
            logger=evaluator_logger,
            counter=evaluator_counter,
        )

    def make_replay_tables(env_spec):
        return [
            reverb.Table(
                name="priority_table",
                sampler=reverb.selectors.Uniform(),
                remover=reverb.selectors.Fifo(),
                # Do not limit the size of the table.
                max_size=config.online_num_steps,
                rate_limiter=reverb.rate_limiters.MinSize(1),
                signature=adders_reverb.NStepTransitionAdder.signature(
                    env_spec,
                    extras_spec={
                        "mc_return": tf.TensorSpec(shape=(), dtype=tf.float32)
                    },  # type: ignore
                ),
            )
        ]

    key = jax.random.PRNGKey(config.seed)

    parent_counter = counting.Counter(time_delta=0.0)
    learner_counter = counting.Counter(parent_counter, "learner", time_delta=0.0)
    learner_logger = logger_factory("learner", learner_counter.get_steps_key(), 0)

    learner_key, key = jax.random.split(key)

    offline_iterator = make_offline_iterator(config.batch_size)
    offline_iterator = jax_utils.prefetch(offline_iterator)

    offline_learner = make_offline_learner(
        networks,
        learner_key,
        offline_iterator,
        env_spec,
        learner_logger,
        learner_counter,
    )

    actor_key, key = jax.random.split(key)

    evaluator_counter = counting.Counter(parent_counter, "evaluator", time_delta=0.0)
    evaluator_logger = logger_factory("evaluator", evaluator_counter.get_steps_key(), 0)

    offline_evaluator = make_evaluator(actor_key, offline_learner)

    # Offline training
    max_num_offline_steps = config.offline_num_steps
    steps = 0
    eval_every = config.offline_eval_every
    while steps < max_num_offline_steps:
        learner_steps = min(eval_every, max_num_offline_steps - steps)
        for _ in range(learner_steps):
            offline_learner.step()
        offline_evaluator.run(config.offline_num_eval_episodes)
        steps += learner_steps

    reverb_tables = make_replay_tables(env_spec)
    replay_server = reverb.Server(reverb_tables, port=None)
    replay_client = replay_server.localhost_client()

    train_env = d4rl_utils.make_environment(config.dataset_name, config.seed)
    online_counter = counting.Counter(parent_counter, "actor", time_delta=0.0)
    online_logger = logger_factory("actor", online_counter.get_steps_key(), 0)

    # Online fine-tuning
    online_dataset = make_online_iterator(replay_client)

    learner_key, key = jax.random.split(key)
    online_learner = make_offline_learner(
        networks,
        learner_key,
        online_dataset,
        env_spec,
        learner_logger,
        learner_counter,
    )
    online_learner.restore(offline_learner.save())

    del offline_learner, offline_iterator, offline_evaluator

    adder = calql.CalQLAdder(
        adders_reverb.NStepTransitionAdder(
            replay_client, n_step=1, discount=config.discount
        ),
        config.discount,
        reward_config=reward_config,
    )

    actor_key, eval_key = jax.random.split(key)
    online_actor = make_actor(train_policy, actor_key, online_learner, adder=adder)
    online_evaluator = make_evaluator(eval_key, online_learner)

    initial_num_steps = config.initial_num_steps
    eval_every = config.online_eval_every
    eval_episodes = config.online_num_eval_episodes
    num_steps = 0
    episode_length = 0
    episode_return = 0
    timestep = train_env.reset()
    online_actor.observe_first(timestep)
    online_actor.update()
    online_evaluator.run(num_episodes=eval_episodes)
    while True:
        action = online_actor.select_action(timestep.observation)
        next_timestep = train_env.step(action)
        num_steps += 1
        episode_return += next_timestep.reward
        episode_length += 1
        if num_steps >= int(config.online_num_steps):
            break
        if num_steps >= initial_num_steps:
            for _ in range(config.online_utd_ratio):
                online_learner.step()

        if num_steps % eval_every == 0:
            online_evaluator.run(num_episodes=eval_episodes)

        online_actor.observe(action, next_timestep)
        online_actor.update()

        if next_timestep.last():
            counts = online_counter.increment(episodes=1, steps=episode_length)
            online_logger.write({**counts, "episode_return": episode_return})
            episode_return = 0
            episode_length = 0
            timestep = train_env.reset()
            online_actor.observe_first(timestep)
        else:
            timestep = next_timestep


if __name__ == "__main__":
    absl.app.run(main)
