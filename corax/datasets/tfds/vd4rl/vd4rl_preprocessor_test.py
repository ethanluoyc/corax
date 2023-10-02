import numpy as np
import tensorflow as tf
from absl.testing import absltest
from absl.testing import parameterized

from corax.datasets.tfds.vd4rl import vd4rl_preprocessor

_TRANSITION_TEST_CASES = [
    {
        "testcase_name": "test_normal",
        "steps": {
            "observation": np.arange(4, dtype=np.float32),
            "reward": np.array([1, 2, 3, 4], dtype=np.float32),
            "discount": np.ones(4, dtype=np.float32),
            "action": np.arange(4, dtype=np.float32),
        },
        "expected_transitions": {
            "observation": np.array([0, 1]),
            "next_observation": np.array([2, 3]),
            "discount": np.array([0.99, 0.99]),
            "action": np.array([0, 1]),
            "reward": np.array([1 + 2 * 0.99, 2 + 3 * 0.99]),
        },
        "n_steps": 2,
        "discount": 0.99,
    },
    {
        "testcase_name": "test_normal_2",
        "steps": {
            "observation": np.array([0, 1, 2, 3, 4, 5], dtype=np.float32),
            "reward": np.array([1, 2, 3, 4, 5, 6], dtype=np.float32),
            "discount": np.array([1, 1, 1, 1, 1, 1], dtype=np.float32),
            "action": np.array([0, 1, 2, 3, 4, 5], dtype=np.float32),
        },
        "expected_transitions": {
            "observation": np.array([0, 1, 2]),
            "next_observation": np.array([3, 4, 5]),
            "discount": np.array([0.99 * 0.99, 0.99 * 0.99, 0.99 * 0.99]),
            "action": np.array([0, 1, 2]),
            "reward": np.array(
                [
                    1 + 2 * 0.99 + 3 * 0.99 * 0.99,
                    2 + 3 * 0.99 + 4 * 0.99 * 0.99,
                    3 + 4 * 0.99 + 5 * 0.99 * 0.99,
                ]
            ),
        },
        "n_steps": 3,
        "discount": 0.99,
    },
    {
        "testcase_name": "test_terminal",
        "steps": {
            "observation": np.array([0, 1, 2, 3, 4, 5], dtype=np.float32),
            "reward": np.array([1, 2, 3, 4, 5, 6], dtype=np.float32),
            "discount": np.array([1, 1, 1, 1, 0, 0], dtype=np.float32),
            "action": np.array([0, 1, 2, 3, 4, 5], dtype=np.float32),
        },
        "expected_transitions": {
            "observation": np.array([0, 1, 2]),
            "next_observation": np.array([3, 4, 5]),
            "discount": np.array([0.99 * 0.99, 0.99 * 0.99, 0]),
            "action": np.array([0, 1, 2]),
            "reward": np.array(
                [
                    1 + 2 * 0.99 + 3 * 0.99 * 0.99,
                    2 + 3 * 0.99 + 4 * 0.99 * 0.99,
                    # 3 + 4 * 0.99 + 5 * 0.99 * 0.99,
                    3 + 4 * 0.99 + 5 * 0.99 * 0.99,
                ]
            ),
        },
        "n_steps": 3,
        "discount": 0.99,
    },
    {
        "testcase_name": "test_terminal_2",
        "steps": {
            "observation": np.array([1, 2, 3, 4, 5], dtype=np.float32),
            "reward": np.array([2, 3, 5, 7, 0], dtype=np.float32),
            "discount": np.array([1, 1, 1, 0, 0], dtype=np.float32),
            "action": np.array([0, 0, 0, 0, 0], dtype=np.float32),
        },
        "expected_transitions": {
            "observation": np.array([1, 2]),
            "next_observation": np.array([4, 5]),
            "discount": np.array([0.5 * 0.5, 0]),
            "action": np.array([0, 0]),
            "reward": np.array(
                [
                    2 + 0.5 * 3 + 0.25 * 5,
                    3 + 0.5 * 5 + 0.25 * 7,
                ]
            ),
        },
        "n_steps": 3,
        "discount": 0.5,
    },
    {
        "testcase_name": "test_one_step",
        "steps": {
            "observation": np.array([1, 2, 3], dtype=np.float32),
            "reward": np.array([2, 3, -1], dtype=np.float32),
            "discount": np.array([1, 0, 0], dtype=np.float32),
            "action": np.array([0, 1, 0], dtype=np.float32),
        },
        "expected_transitions": {
            "observation": np.array([1, 2]),
            "next_observation": np.array([2, 3]),
            "discount": np.array([1, 0]),
            "action": np.array([0, 1]),
            "reward": np.array([2, 3]),
        },
        "n_steps": 1,
        "discount": 0.5,
    },
]


def _assert_transitions_equal(np_transitions, expected_transitions):
    np.testing.assert_allclose(
        np_transitions["observation"], expected_transitions["observation"]
    )
    np.testing.assert_allclose(
        np_transitions["next_observation"], expected_transitions["next_observation"]
    )
    np.testing.assert_allclose(np_transitions["reward"], expected_transitions["reward"])
    np.testing.assert_allclose(
        np_transitions["discount"], expected_transitions["discount"]
    )
    np.testing.assert_allclose(np_transitions["action"], expected_transitions["action"])


class VD4RLDatasetTest(parameterized.TestCase):
    def test_observation_stacking_np(self):
        test_obs = np.arange(4).reshape((4, 1))
        stacked_obs = vd4rl_preprocessor.stack_observations_np(test_obs, 2)
        expected_stacked_obs = np.array(
            [
                [0, 0],
                [0, 1],
                [1, 2],
                [2, 3],
            ]
        )
        np.testing.assert_equal(stacked_obs, expected_stacked_obs)

    def test_observation_stacking_tfds(self):
        test_obs = np.arange(4).reshape((4, 1))
        stacked_obs = next(
            vd4rl_preprocessor.stack_observation_tfds(
                tf.data.Dataset.from_tensor_slices(test_obs), 2
            )
            .batch(100)
            .as_numpy_iterator()
        )
        expected_stacked_obs = np.array(
            [
                [0, 0],
                [0, 1],
                [1, 2],
                [2, 3],
            ]
        )
        np.testing.assert_equal(stacked_obs, expected_stacked_obs)

    @parameterized.named_parameters(*_TRANSITION_TEST_CASES)
    def test_create_transitions_np(
        self, n_steps, steps, discount, expected_transitions
    ):
        np_transitions = vd4rl_preprocessor.get_n_step_transitions(
            steps, n_steps, discount
        )
        _assert_transitions_equal(np_transitions, expected_transitions)

    @parameterized.named_parameters(*_TRANSITION_TEST_CASES)
    def test_create_transitions_tfds(
        self, n_steps, steps, discount, expected_transitions
    ):
        steps_dataset = tf.data.Dataset.from_tensor_slices(steps)
        transition_ds = vd4rl_preprocessor.tfds_get_n_step_transitions(
            steps_dataset, n_steps, discount
        )
        tf_transitions = next(transition_ds.batch(100).as_numpy_iterator())
        _assert_transitions_equal(tf_transitions, expected_transitions)


if __name__ == "__main__":
    tf.config.set_visible_devices([], "GPU")
    absltest.main()
