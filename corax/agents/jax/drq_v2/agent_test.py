import unittest

from absl.testing import absltest
from dm_env import specs as env_specs
import dm_env_wrappers
import numpy as np

from corax import specs
from corax.agents.jax import drq_v2
from corax.jax import experiments
from corax.testing import fakes
from corax.utils import loggers


def _make_empty_experiment_logger(*args, **kwargs):
    del args, kwargs
    return loggers.TerminalLogger(time_delta=10.0)


def _fake_drq_control_environment():
    environment = fakes.Environment(
        specs.EnvironmentSpec(
            observations=env_specs.BoundedArray((84, 84, 9), np.uint8, 0, 255),
            actions=env_specs.BoundedArray((3,), np.float32, -1, 1),
            rewards=env_specs.Array((), np.float32),
            discounts=env_specs.BoundedArray((), np.float32, 0, 1),
        )
    )
    return dm_env_wrappers.CanonicalSpecWrapper(environment, clip=True)


class DrQV2Test(absltest.TestCase):
    @unittest.skip("slow")
    def test_agent(self):
        environment_factory = lambda _: _fake_drq_control_environment()
        builder = drq_v2.DrQV2Builder(
            config=drq_v2.DrQV2Config(
                min_replay_size=10, batch_size=10, samples_per_insert=10
            )
        )
        config = experiments.ExperimentConfig(
            builder,
            max_num_actor_steps=20,
            seed=0,
            network_factory=drq_v2.make_networks,  # type: ignore
            environment_factory=environment_factory,
            logger_factory=_make_empty_experiment_logger,
            checkpointing=None,
        )
        experiments.run_experiment(config)


if __name__ == "__main__":
    absltest.main()
