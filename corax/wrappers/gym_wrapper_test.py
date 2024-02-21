# type: ignore
"""Tests for gymnasium_wrapper."""

from absl.testing import absltest
from dm_env import specs
import gym
import numpy as np

from corax.wrappers import gym_wrapper


class GymWrapperTest(absltest.TestCase):
    def test_gym_cartpole(self):
        env = gym_wrapper.GymWrapper(gym.make("CartPole-v0"))

        # Test converted observation spec.
        observation_spec: specs.BoundedArray = env.observation_spec()
        self.assertEqual(type(observation_spec), specs.BoundedArray)
        self.assertEqual(observation_spec.shape, (4,))
        self.assertEqual(observation_spec.minimum.shape, (4,))
        self.assertEqual(observation_spec.maximum.shape, (4,))
        self.assertEqual(observation_spec.dtype, np.dtype("float32"))

        # Test converted action spec.
        action_spec: specs.BoundedArray = env.action_spec()
        self.assertEqual(type(action_spec), specs.DiscreteArray)
        self.assertEqual(action_spec.shape, ())
        self.assertEqual(action_spec.minimum, 0)
        self.assertEqual(action_spec.maximum, 1)
        self.assertEqual(action_spec.num_values, 2)
        self.assertEqual(action_spec.dtype, np.dtype("int64"))

        # Test step.
        timestep = env.reset()
        self.assertTrue(timestep.first())
        timestep = env.step(1)
        self.assertEqual(timestep.reward, 1.0)
        self.assertTrue(np.isscalar(timestep.reward))
        self.assertEqual(timestep.observation.shape, (4,))
        env.close()

    def test_early_truncation(self):
        # Pendulum has no early termination condition. Recent versions of gym force
        # to use v1. We try both in case an earlier version is installed.
        try:
            gym_env = gym.make("Pendulum-v1")
        except:  # noqa
            gym_env = gym.make("Pendulum-v0")
        env = gym_wrapper.GymWrapper(gym_env)
        ts = env.reset()
        while not ts.last():
            ts = env.step(env.action_spec().generate_value())
        self.assertEqual(ts.discount, 1.0)
        self.assertTrue(np.isscalar(ts.reward))
        env.close()

    def test_multi_discrete(self):
        space = gym.spaces.MultiDiscrete([2, 3])
        spec = gym_wrapper._convert_to_spec(space)

        spec.validate([0, 0])
        spec.validate([1, 2])

        self.assertRaises(ValueError, spec.validate, [2, 2])
        self.assertRaises(ValueError, spec.validate, [1, 3])


if __name__ == "__main__":
    absltest.main()
