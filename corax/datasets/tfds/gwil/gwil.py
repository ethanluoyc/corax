"""gwil dataset."""

import dataclasses
import os
import pickle

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow_datasets.rlds import rlds_base

_DESCRIPTION = """
Expert demonstrations used in [Fickinger et al., 2022](https://arxiv.org/pdf/2110.03684.pdf).
"""

_CITATION = """
@inproceedings{fickinger2022gromov,
  title={Cross-Domain Imitation Learning via Optimal Transport},
  author={Fickinger, Arnaud and Cohen, Samuel and Russell, Stuart and Amos, Brandon},
  booktitle={10th International Conference on Learning Representations, ICLR},
  year={2022}
}
"""

_HOMEPAGE = "https://arnaudfickinger.github.io/gwil/"

_DRIVE_PATH = (
    "https://drive.google.com/uc?export=download&id=1xE882IuQkXUuaeXHInYaP9eqvhQm48Et"
)
_DATA_PATHS = {
    "cartpole_swingup": "expert_trajectory_cartpole_swingup.pickle",
    "walker_walk": "expert_trajectory_walker_walk.pickle",
    "cheetah_run": "expert_trajectory_cheetah_run.pickle",
    "pendulum_swingup": "expert_trajectory_pendulum_swingup.pickle",
    "maze_end_pointmass": "expert_trajectory_MazeEnd_PointMass_0.pickle",
}


def build_episode(demo_file: os.PathLike):
    # GWIL dataset contains only a single episode
    time_limit = 1000
    with open(demo_file, "rb") as infile:
        traj = pickle.load(infile)
    # if "Maze" in os.path.normpath(demo_file):
    #   dmc_observations = traj['dmc_obs']
    #   observation_keys = traj['dmc_obs'].keys()
    #   flat_observation = np.concatenate([traj['dmc_obs'][key] for key in observation_keys], axis=-1)
    #   print(traj['dmc_obs'].keys())
    # Data validation
    # Check next_observation and observation is consistent.
    assert (traj["obs"][1:] == traj["nobs"][:-1]).all()
    # NOTE: Pixel observation in the GWIL episode is inconsistent
    # In addition, they are not the same as the pixel observations used in typical pixel agents.
    # Therefore, don't use the pixel observation for training.
    episode_length = len(traj["reward"])
    ends_in_terminal = episode_length < time_limit
    steps_dict = {
        "observation": {
            "states": np.concatenate([traj["obs"], [traj["nobs"][-1]]], axis=0),
            "pixels": np.transpose(
                np.concatenate([traj["pixel_obs"], [traj["pixel_nobs"][-1]]], axis=0),
                [0, 2, 3, 1],
            ),
        },
        "action": np.concatenate(
            [traj["action"], [np.zeros_like(traj["action"][0])]], axis=0
        ),
        "reward": np.concatenate(
            [np.asarray(traj["reward"]), [np.zeros_like(traj["reward"][0])]], axis=0
        ),
        "is_first": np.asarray([True] + [False] * episode_length),
        "is_last": np.asarray([False] * episode_length + [True]),
        "is_terminal": np.asarray([False] * episode_length + [ends_in_terminal]),
    }
    episode = {"episode_return": traj["cumulative_reward"], "steps": steps_dict}
    return episode


@dataclasses.dataclass
class GwilBuilderConfig(tfds.core.BuilderConfig):
    state_size: int = 0
    action_size: int = 0


class Gwil(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for gwil dataset."""

    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {
        "1.0.0": "Initial release.",
    }

    # pylint: disable=unexpected-keyword-arg
    BUILDER_CONFIGS = [
        GwilBuilderConfig(
            name="cartpole_swingup",
            state_size=5,
            action_size=1,
        ),
        GwilBuilderConfig(
            name="pendulum_swingup",
            state_size=3,
            action_size=1,
        ),
        GwilBuilderConfig(
            name="walker_walk",
            state_size=24,
            action_size=6,
        ),
        GwilBuilderConfig(
            name="cheetah_run",
            state_size=17,
            action_size=6,
        ),
        GwilBuilderConfig(
            name="maze_end_pointmass",
            state_size=6,
            action_size=2,
        ),
    ]

    # pylint: enable=unexpected-keyword-arg

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        # pylint: disable=unexpected-keyword-arg
        builder_config = self.builder_config
        observation_info = {
            "states": tfds.features.Tensor(
                shape=(builder_config.state_size,),
                dtype=tf.float64,
                encoding=tfds.features.Encoding.ZLIB,
            ),
            "pixels": tfds.features.Image(
                shape=(84, 84, 3), dtype=tf.uint8, encoding_format="png"
            ),
        }
        action_info = tfds.features.Tensor(
            shape=(builder_config.action_size,), dtype=tf.float32
        )
        ds_config = rlds_base.DatasetConfig(
            name=builder_config.name,
            observation_info=observation_info,
            action_info=action_info,
            reward_info=tfds.features.Tensor(shape=(), dtype=tf.float64),
            episode_metadata_info={
                "episode_return": tfds.features.Tensor(shape=(), dtype=tf.float64)
            },
            citation=_CITATION,
            homepage=_HOMEPAGE,
            overall_description=_DESCRIPTION,
            supervised_keys=None,  # pytype: disable=wrong-arg-types  # gen-stub-imports
        )
        return rlds_base.build_info(ds_config, self)

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        # TODO(gwil): Downloads the data and defines the splits
        path = dl_manager.download_and_extract({"file_path": _DRIVE_PATH})["file_path"]

        filename = _DATA_PATHS[self.builder_config.name]

        # TODO(gwil): Returns the Dict[split names, Iterator[Key, Example]]
        return {
            "train": self._generate_examples(path / "exp" / filename),
        }

    def _generate_examples(self, path):
        """Yields examples."""
        # TODO(gwil): Yields (key, example) tuples from the dataset
        key = os.path.basename(path)
        yield key, build_episode(path)
