import dataclasses
from typing import Optional, Sequence

import jax
import numpy as np
import optax
import tensorflow as tf
from absl import app
from ml_collections import config_flags

import corax
from baselines import d4rl_utils
from baselines import experiment_utils
from corax import environment_loop
from corax.agents.jax import actor_core
from corax.agents.jax import actors as actors_lib
from corax.agents.jax import iql
from corax.agents.jax import learners as learners_lib
from corax.datasets import tfds
from corax.jax import variable_utils
from corax.utils import counting


@dataclasses.dataclass
class Config:
    env_name: str = "halfcheetah-medium-v2"
    num_episodes: Optional[int] = None
    batch_size: int = 256

    max_num_learner_steps: int = int(1e6)
    num_eval_episodes: int = 10
    eval_every: int = 5000
    seed: int = 0

    hidden_dims: Sequence[int] = (256, 256)

    use_cosine_decay: bool = True
    policy_lr: float = 3e-4
    critic_lr: float = 3e-4
    value_lr: float = 3e-4

    discount: float = 0.99
    tau: float = 5e-3
    expectile: float = 0.7
    temperature: float = 3.0

    log_to_wandb: bool = False


_CONFIG = config_flags.DEFINE_config_dataclass("config", Config())


def make_dataset(d4rl_name, batch_size, seed, num_episodes=None):
    dataset = tfds.load_tfds_dataset(
        d4rl_utils.get_tfds_name(d4rl_name), num_episodes=num_episodes
    )
    if "antmaze" in d4rl_name:
        reward_scale = 1.0
        reward_bias = -1.0
    else:
        num_episodes = dataset.cardinality()
        dataset = dataset.apply(d4rl_utils.add_episode_return).cache()  # type: ignore
        episode_returns = dataset.map(
            lambda episode: episode["episode_return"],
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        episode_returns = (
            episode_returns.batch(num_episodes).get_single_element().numpy()  # type: ignore
        )
        max_episode_return = np.max(episode_returns)
        min_episode_return = np.min(episode_returns)
        reward_scale = 1000.0 / (max_episode_return - min_episode_return)
        reward_bias = 0.0

    transitions = d4rl_utils.transform_transitions_dataset(
        dataset, reward_scale, reward_bias
    )
    iterator = tfds.JaxInMemoryRandomSampleIterator(
        transitions, jax.random.PRNGKey(seed), batch_size
    )
    return iterator


def train_evaluate(
    environment,
    dataset,
    random_key,
    learner_core,
    eval_policy,
    logger_factory,
    *,
    max_num_learner_steps: int,
    num_eval_episodes: int,
    eval_every: int,
    observers=(),
):
    parent_counter = counting.Counter(time_delta=0.0)

    learner_counter = counting.Counter(parent_counter, prefix="learner", time_delta=0.0)

    learner_key, evaluator_key = jax.random.split(random_key)

    learner = learners_lib.DefaultJaxLearner(
        learner_core,
        learner_key,
        dataset,
        counter=learner_counter,
        logger=logger_factory("learner", learner_counter.get_steps_key()),
    )

    eval_counter = counting.Counter(parent_counter, prefix="evaluator", time_delta=0.0)
    eval_actor = actors_lib.GenericActor(
        eval_policy,
        evaluator_key,
        variable_utils.VariableClient(learner, "policy"),
    )

    eval_loop = environment_loop.EnvironmentLoop(
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


def main(_):
    config: Config = _CONFIG.value

    logger_factory = experiment_utils.LoggerFactory(
        log_to_wandb=config.log_to_wandb,
        wandb_kwargs={"project": "corax", "config": config},
        learner_time_delta=5.0,
        evaluator_time_delta=0.0,
    )

    env = d4rl_utils.make_environment(config.env_name, seed=config.seed)
    dataset = make_dataset(
        config.env_name,
        config.batch_size,
        seed=config.seed,
        num_episodes=config.num_episodes,
    )
    spec = corax.make_environment_spec(env)

    networks = iql.make_networks(spec, hidden_dims=config.hidden_dims)

    if config.use_cosine_decay:
        schedule_fn = optax.cosine_decay_schedule(
            -config.policy_lr, config.max_num_learner_steps
        )
        policy_optimizer = optax.chain(
            optax.scale_by_adam(), optax.scale_by_schedule(schedule_fn)
        )
    else:
        policy_optimizer = optax.adam(learning_rate=config.policy_lr)

    learner_core = iql.IQLLearnerCore(
        networks,
        policy_optimizer=policy_optimizer,
        critic_optimizer=optax.adam(config.critic_lr),
        value_optimizer=optax.adam(config.value_lr),
        discount=config.discount,
        tau=config.tau,
        expectile=config.expectile,
        temperature=config.temperature,
    )

    eval_policy = actor_core.batched_feed_forward_to_actor_core(
        iql.apply_policy_and_sample(networks, spec.actions, eval_mode=True)
    )

    train_evaluate(
        env,
        dataset,
        jax.random.PRNGKey(config.seed),
        learner_core,
        eval_policy,
        logger_factory,
        max_num_learner_steps=config.max_num_learner_steps,
        eval_every=config.eval_every,
        num_eval_episodes=config.num_eval_episodes,
        observers=[d4rl_utils.D4RLScoreObserver(config.env_name)],
    )


if __name__ == "__main__":
    app.run(main)
