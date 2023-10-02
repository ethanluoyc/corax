import numpy as np
from absl.testing import absltest

from baselines.imitation import imitation_loop
from corax import specs
from corax.testing import fakes


class _DummyRewarder(imitation_loop.EpisodeRewarder):
    def compute_offline_rewards(self, agent_steps, update: bool):
        del update
        return np.zeros((len(agent_steps) - 1))


class ImitationLoopTest(absltest.TestCase):
    def test_imitation_loop(self):
        environment = fakes.DiscreteEnvironment()
        actor = fakes.Actor(specs.make_environment_spec(environment))
        rewarder = _DummyRewarder()
        loop = imitation_loop.ImitationEnvironmentLoop(
            environment, actor, adder=None, rewarder=rewarder
        )
        loop.run_episode()


if __name__ == "__main__":
    absltest.main()
