"""rot dataset."""

import dataclasses
import os
import pickle

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow_datasets.rlds import rlds_base

_DESCRIPTION = """
Datasets for the [ROT paper](https://arxiv.org/abs/2206.15469).

NOTE: last observations are not recorded in the dataset.
"""

_CITATION = """
@inproceedings{
haldar2022watch,
title={Watch and Match: Supercharging Imitation with Regularized Optimal Transport},
author={Siddhant Haldar and Vaibhav Mathur and Denis Yarats and Lerrel Pinto},
booktitle={6th Annual Conference on Robot Learning},
year={2022},
url={https://openreview.net/forum?id=ZUtgUA0Fuwd}
}
"""
_HOMEPAGE = """
https://rot-robot.github.io/
"""

_FILE_URL = "https://osf.io/zgh2f/download"


def _build_rot_dataset(file_path: os.PathLike):
    with tf.io.gfile.GFile(file_path, "rb") as f:
        demo = pickle.load(f)
        expert_pixels, expert_states, expert_actions, expert_reward = demo
        # [N, T, C, H, W] -> [N, T, H, W, C]
        expert_pixels = np.transpose(expert_pixels, [0, 1, 3, 4, 2])

    num_demonstrations = expert_pixels.shape[0]
    episode_length = expert_pixels.shape[1]

    episodes = []
    for i in range(num_demonstrations):
        episodes.append(
            {
                "steps": {
                    "observation": {
                        "pixels": expert_pixels[i],
                        "states": expert_states[i],
                    },
                    "action": expert_actions[i],
                    "reward": expert_reward[i],
                    "is_first": [True] + [False] * (episode_length - 1),
                    "is_last": [False] * (episode_length - 1) + [True],
                    "is_terminal": [False] * episode_length,
                }
            }
        )
    return episodes


@dataclasses.dataclass
class RotDmcConfig(tfds.core.BuilderConfig):
    suite: str = "dmc"
    observation_size: int = 0
    action_size: int = 0


class RotDmc(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for rot dataset."""

    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {
        "1.0.0": "Initial release.",
    }

    # pylint: disable=unexpected-keyword-arg
    BUILDER_CONFIGS = [
        RotDmcConfig(
            name="cheetah_run",
            suite="dmc",
            observation_size=17,
            action_size=6,
        ),
        RotDmcConfig(
            name="cartpole_swingup",
            suite="dmc",
            observation_size=5,
            action_size=1,
        ),
        RotDmcConfig(
            name="acrobot_swingup",
            suite="dmc",
            observation_size=6,
            action_size=1,
        ),
        RotDmcConfig(
            name="finger_spin",
            suite="dmc",
            observation_size=9,
            action_size=2,
        ),
        RotDmcConfig(
            name="hopper_hop",
            suite="dmc",
            observation_size=15,
            action_size=4,
        ),
        RotDmcConfig(
            name="hopper_stand",
            suite="dmc",
            observation_size=15,
            action_size=4,
        ),
        RotDmcConfig(
            name="quadruped_run",
            suite="dmc",
            observation_size=78,
            action_size=12,
        ),
        RotDmcConfig(
            name="walker_run",
            suite="dmc",
            observation_size=24,
            action_size=6,
        ),
        RotDmcConfig(
            name="walker_stand",
            suite="dmc",
            observation_size=24,
            action_size=6,
        ),
        RotDmcConfig(
            name="walker_walk",
            suite="dmc",
            observation_size=24,
            action_size=6,
        ),
    ]

    # pylint: disable=unexpected-keyword-arg

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        builder_config = self.builder_config
        observation_info = {
            "states": tfds.features.Tensor(
                shape=(builder_config.observation_size,),
                dtype=tf.float32,
                encoding=tfds.features.Encoding.ZLIB,
            ),
            "pixels": tfds.features.Tensor(
                shape=(84, 84, 9),
                dtype=tf.uint8,
                encoding=tfds.features.Encoding.ZLIB,
            ),
        }
        action_info = tfds.features.Tensor(
            shape=(builder_config.action_size,),
            dtype=tf.float32,
            encoding=tfds.features.Encoding.ZLIB,
        )
        ds_config = rlds_base.DatasetConfig(
            name=builder_config.name,
            observation_info=observation_info,
            action_info=action_info,
            reward_info=tfds.features.Tensor(shape=(), dtype=tf.float64),
            citation=_CITATION,
            homepage=_HOMEPAGE,
            overall_description=_DESCRIPTION,
            supervised_keys=None,  # pytype: disable=wrong-arg-types  # gen-stub-imports
        )
        domain, _ = self.builder_config.name.split("_")
        camera_id = dict(quadruped=2).get(domain, 0)
        metadata = {
            "camera_id": camera_id,
            "frame_stack": 3,
            "action_repeat": 2,
        }
        return rlds_base.build_info(ds_config, self, metadata)

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        # TODO(rot): Downloads the data and defines the splits
        demo_path = dl_manager.download_and_extract({"file_path": _FILE_URL})[
            "file_path"
        ]
        name = self.builder_config.name

        return {
            "train": self._generate_examples(
                demo_path / "expert_demos" / self.builder_config.suite / name
            ),
        }

    def _generate_examples(self, path):
        """Yields examples."""
        # TODO(rot): Yields (key, example) tuples from the dataset
        for f in path.glob("*.pkl"):
            for episode_id, episode in enumerate(_build_rot_dataset(f)):
                yield episode_id, episode
