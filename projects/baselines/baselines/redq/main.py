import dataclasses
from typing import Any, Callable, Iterator, Optional

import dm_env_wrappers
import jax
import optax
import reverb
from absl import app
from absl import flags
from ml_collections import config_flags

import corax
from baselines import experiment_utils
from baselines.redq.config import Config
from corax import adders as adders_lib
from corax import environment_loop
from corax import specs
from corax import types
from corax import wrappers
from corax.adders import reverb as adders
from corax.agents.jax import actor_core
from corax.agents.jax import actors as actors_lib
from corax.agents.jax import learners as learners_lib
from corax.agents.jax import local_layout
from corax.agents.jax import redq
from corax.datasets import reverb as reverb_datasets
from corax.jax import utils as jax_utils
from corax.jax import variable_utils
from corax.utils import counting
from corax.utils import reverb_utils

# TODO(yl): Move this to corax package


@dataclasses.dataclass
class ReverbReplay:
    server: reverb.Server
    adder: adders_lib.Adder
    data_iterator: Iterator[Any]
    client: Optional[reverb.Client] = None
    can_sample: Callable[[], bool] = lambda: True


def make_local_reverb_transition_replay(
    environment_spec: specs.EnvironmentSpec,
    extra_spec: types.NestedSpec = (),
    batch_size: int = 256,
    discount: float = 0.99,
    max_replay_size: int = int(1e6),
    min_replay_size: int = 1000,
    n_step: int = 1,
    prefetch_size: int = 1,
    samples_per_insert: float = 256,
    samples_per_insert_tolerance_rate: float = 0.1,
    replay_table_name: str = adders.DEFAULT_PRIORITY_TABLE,
):
    samples_per_insert_tolerance = (
        samples_per_insert_tolerance_rate * samples_per_insert
    )
    error_buffer = min_replay_size * samples_per_insert_tolerance
    limiter = reverb.rate_limiters.SampleToInsertRatio(
        min_size_to_sample=min_replay_size,
        samples_per_insert=samples_per_insert,
        error_buffer=max(error_buffer, 2 * samples_per_insert),
    )

    # Create a replay server to add data to. This uses no limiter behavior in
    # order to allow the Agent interface to handle it.
    replay_tables = [
        reverb.Table(
            name=replay_table_name,
            sampler=reverb.selectors.Uniform(),
            remover=reverb.selectors.Fifo(),
            max_size=max_replay_size,
            rate_limiter=limiter,
            signature=adders.NStepTransitionAdder.signature(
                environment_spec, extra_spec
            ),
        )
    ]

    replay_tables, sample_sizes = reverb_utils.disable_insert_blocking(replay_tables)

    server = reverb.Server(replay_tables, port=None)

    # The adder is used to insert observations into replay.
    client = server.localhost_client()
    adder = adders.NStepTransitionAdder(client, n_step=n_step, discount=discount)

    dataset = reverb_datasets.make_reverb_dataset(
        table=replay_table_name,
        server_address=client.server_address,
        batch_size=batch_size,
        prefetch_size=prefetch_size,
    )

    dataset = dataset.map(lambda x: x.data).as_numpy_iterator()

    def can_sample():
        for table, sample_size in zip(replay_tables, sample_sizes):
            if not table.can_sample(sample_size):
                return False
        return True

    return ReverbReplay(server, adder, dataset, client, can_sample=can_sample)  # type: ignore


def train(
    environment_factory,
    network_factory,
    replay_factory,
    learner_core_factory,
    policy_factory,
    logger_factory,
    seed: int,
    *,
    max_num_actor_steps: int,
    num_eval_episodes: int,
    eval_every: int,
):
    environment = environment_factory(seed)
    spec = corax.make_environment_spec(environment)
    replay = replay_factory(spec)
    networks = network_factory(spec)

    dataset = jax_utils.device_put(replay.data_iterator, jax.local_devices()[0])  # type: ignore
    dataset = jax_utils.prefetch(dataset, buffer_size=1)

    actor_key, learner_key, eval_key = jax.random.split(jax.random.PRNGKey(seed), 3)

    parent_counter = counting.Counter(time_delta=0.0)

    learner_core = learner_core_factory(spec, networks)

    learner_counter = counting.Counter(parent_counter, prefix="learner", time_delta=0.0)
    learner = learners_lib.DefaultJaxLearner(
        learner_core,
        learner_key,
        dataset,
        logger=logger_factory("learner", learner_counter.get_steps_key()),
        counter=learner_counter,
    )

    train_policy = policy_factory(networks, evaluation=False)
    eval_policy = policy_factory(networks, evaluation=False)

    train_actor = actors_lib.GenericActor(
        train_policy,
        random_key=actor_key,
        variable_client=variable_utils.VariableClient(learner, "policy", device="cpu"),
        adder=replay.adder,
    )

    agent = local_layout.LocalLayout(train_actor, learner, dataset, replay.can_sample)

    actor_counter = counting.Counter(parent_counter, prefix="actor", time_delta=0.0)
    train_loop = environment_loop.EnvironmentLoop(
        environment,
        agent,
        logger=logger_factory("actor", actor_counter.get_steps_key()),
        counter=actor_counter,
        should_update=True,
    )

    # Create the evaluation actor and loop.
    eval_counter = counting.Counter(parent_counter, prefix="evaluator", time_delta=0.0)
    eval_logger = logger_factory("evaluator", eval_counter.get_steps_key())

    eval_actor = actors_lib.GenericActor(
        eval_policy,
        random_key=eval_key,
        variable_client=variable_utils.VariableClient(learner, "policy", device="cpu"),
    )

    eval_loop = environment_loop.EnvironmentLoop(
        environment,
        eval_actor,
        counter=eval_counter,
        logger=eval_logger,
        should_update=True,
    )

    steps = 0
    while steps < max_num_actor_steps:
        eval_loop.run(num_episodes=num_eval_episodes)
        num_steps = min(eval_every, max_num_actor_steps - steps)
        steps += train_loop.run(num_steps=num_steps)

    eval_loop.run(num_episodes=num_eval_episodes)


def make_environment(suite, task, seed):
    if suite == "gymnasium":
        import gymnasium

        environment = gymnasium.make(task)
        environment.reset(seed=seed)
        environment = wrappers.GymnasiumWrapper(environment)
        environment = dm_env_wrappers.SinglePrecisionWrapper(environment)
        environment = dm_env_wrappers.CanonicalSpecWrapper(environment, clip=True)

    elif suite == "dmc":
        from dm_control import suite as dm_suite

        domain_name, task_name = task.split(":")
        environment = dm_suite.load(
            domain_name, task_name, task_kwargs={"random": seed}
        )
        environment = dm_env_wrappers.SinglePrecisionWrapper(environment)
        environment = dm_env_wrappers.ConcatObservationWrapper(environment)
        environment = dm_env_wrappers.CanonicalSpecWrapper(environment, clip=True)
    else:
        raise ValueError(f"Unknown environment suite: {suite}")
    return environment


_CONFIG = config_flags.DEFINE_config_file("config", None)
flags.mark_flag_as_required("config")


def main(_):
    config: Config = _CONFIG.value

    logger_factory = experiment_utils.LoggerFactory(
        log_to_wandb=config.log_to_wandb,
        wandb_kwargs={"project": "corax", "config": config},
        learner_time_delta=5.0,
        evaluator_time_delta=0.0,
    )

    def environment_factory(seed: int):
        suite, task = config.env_name.split(":", 1)
        return make_environment(suite, task, seed)

    def network_factory(spec):
        return redq.make_networks(spec, hidden_sizes=config.hidden_dims)

    def replay_factory(spec):
        return make_local_reverb_transition_replay(
            spec,
            batch_size=config.batch_size,
            discount=config.discount,
            max_replay_size=config.max_replay_size,
            min_replay_size=config.min_replay_size,
            samples_per_insert=config.batch_size * config.utd_ratio,
        )

    def learner_core_factory(spec, networks):
        return redq.REDQLearnerCore(
            networks,
            policy_optimizer=optax.adam(config.policy_lr),
            critic_optimizer=optax.adam(config.critic_lr),
            temperature_optimizer=optax.adam(config.temperature_lr),
            discount=config.discount,
            target_entropy=redq.target_entropy_from_spec(spec.actions),
            utd_ratio=config.utd_ratio,
        )

    def policy_factory(networks, evaluation):
        return actor_core.batched_feed_forward_to_actor_core(
            redq.apply_policy_and_sample(networks, evaluation=evaluation)
        )

    train(
        environment_factory,
        network_factory,
        replay_factory,
        learner_core_factory,
        policy_factory,
        logger_factory,
        seed=config.seed,
        max_num_actor_steps=config.max_num_actor_steps,
        num_eval_episodes=config.num_eval_episodes,
        eval_every=config.eval_every,
    )


if __name__ == "__main__":
    app.run(main)
