from absl.testing import absltest

from baselines.iql import base_config
from baselines.iql import train


class TrainTest(absltest.TestCase):
    def test_train_and_evaluate(self):
        config = base_config.get_base_config()
        config.num_episodes = 2
        config.env_name = "halfcheetah-medium-replay-v2"
        config.num_eval_episodes = 1
        config.batch_size = 1
        config.hidden_dims = (8, 8)
        config.max_num_learner_steps = 10
        train.train_and_evaluate(config, workdir=None)


if __name__ == "__main__":
    absltest.main()
