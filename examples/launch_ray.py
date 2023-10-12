# type: ignore
import os

import dm_env_wrappers
import jax
import optax
import reverb
from absl import app
from dm_control import suite as control_suite

import corax
from corax import raypad as rp
from corax import specs
from corax.adders import reverb as reverb_adders
from corax.agents.jax import actor_core
from corax.agents.jax import actors
from corax.agents.jax import td3
from corax.datasets import reverb as reverb_datasets
from corax.jax import utils as jax_utils
from corax.jax import variable_utils
from corax.utils import counting
from corax.utils import loggers

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

BATCH_SIZE = 256
SPI = 256.0


def make_dataset_iterator(replay_client: reverb.Client, batch_size=BATCH_SIZE):
    dataset = reverb_datasets.make_reverb_dataset(
        replay_client.server_address, batch_size=batch_size, prefetch_size=4
    )
    dataset = dataset.map(lambda x: x.data)
    return jax_utils.device_put(dataset.as_numpy_iterator(), jax.local_devices()[0])


class Learner:
    def __init__(self, spec, network_factory, replay_client, counter) -> None:
        networks = network_factory(spec)
        self._core = td3.TD3LearnerCore(
            networks, 0.99, optax.adam(3e-4), optax.adam(3e-4), optax.adam(3e-4)
        )
        print(jax.local_devices())
        self._state = self._core.init(jax.random.PRNGKey(0))
        self._update_step = jax.jit(self._core.step)
        self._dataset = make_dataset_iterator(replay_client)
        self._logger = loggers.TerminalLogger("learner", print_fn=print, time_delta=1.0)
        self._counter = counting.Counter(counter, "learner")

    def get_variables(self, names):
        return [self._state.policy_params]

    def step(self):
        batch = next(self._dataset)
        self._state, metrics = self._update_step(self._state, batch)
        counts = self._counter.increment(steps=1)
        self._logger.write({**counts, **metrics})

    def run(self):
        while True:
            self.step()


class Actor:
    def __init__(
        self,
        environment_factory,
        network_factory,
        learner,
        replay_client,
        counter,
    ) -> None:
        self._learner = learner
        self._environment = environment_factory()
        spec = specs.make_environment_spec(self._environment)
        self._networks = network_factory(spec)
        self._policy = td3.get_default_behavior_policy(
            self._networks, spec.actions, 0.3
        )
        adder = reverb_adders.NStepTransitionAdder(
            replay_client, n_step=5, discount=0.99
        )
        self._actor = actors.GenericActor(
            actor_core.batched_feed_forward_to_actor_core(self._policy),
            jax.random.PRNGKey(0),
            variable_client=variable_utils.VariableClient(
                learner, "policy", device="cpu", update_period=1
            ),
            adder=adder,
        )
        self._counter = counting.Counter(counter, "actor")

    def run(self):
        print("Actor running")
        loop = corax.EnvironmentLoop(
            self._environment,
            self._actor,
            logger=loggers.TerminalLogger("actor", print_fn=print, time_delta=1.0),
            counter=self._counter,
        )
        loop.run()


def main(_):
    def environment_builder():
        env = control_suite.load("walker", "walk")
        env = dm_env_wrappers.SinglePrecisionWrapper(env)
        env = dm_env_wrappers.ConcatObservationWrapper(env)
        return env

    def network_factory(spec):
        return td3.make_networks(spec, (256, 256))

    program = rp.Program("program")
    environment_spec = specs.make_environment_spec(environment_builder())

    def make_replay_tables():
        samples_per_insert = SPI
        samples_per_insert_tolerance_rate = 0.1
        samples_per_insert_tolerance = (
            samples_per_insert_tolerance_rate * samples_per_insert
        )
        min_replay_size = 1000
        max_replay_size = int(1e6)
        error_buffer = min_replay_size * samples_per_insert_tolerance
        limiter = reverb.rate_limiters.SampleToInsertRatio(
            min_size_to_sample=min_replay_size,
            samples_per_insert=samples_per_insert,
            error_buffer=error_buffer,
        )
        return [
            reverb.Table(
                name=reverb_adders.DEFAULT_PRIORITY_TABLE,
                sampler=reverb.selectors.Uniform(),
                remover=reverb.selectors.Fifo(),
                max_size=max_replay_size,
                rate_limiter=limiter,
                signature=reverb_adders.NStepTransitionAdder.signature(
                    environment_spec
                ),
            )
        ]

    def build_counter():
        print("Created counter")
        return counting.Counter()

    counter = program.add_node(rp.RayNode(build_counter), "counter")
    program.add_node(rp.RayNode(rp.StepsLimiter, counter, int(5e3)), "counter")
    replay = program.add_node(rp.ReverbNode(make_replay_tables), "replay")
    learner = program.add_node(
        rp.RayNode(Learner, environment_spec, network_factory, replay, counter),
        "learner",
    )
    with program.group("actor"):
        for _ in range(1):
            program.add_node(
                rp.RayNode(
                    Actor,
                    environment_builder,
                    network_factory,
                    learner,
                    replay,
                    counter,
                ),
            )
    resources = {"learner": {"num_gpus": 1}}
    rp.launch(program, resources)


if __name__ == "__main__":
    app.run(main)
