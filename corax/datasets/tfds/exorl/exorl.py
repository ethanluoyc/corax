"""exorl dataset."""

import dataclasses

import numpy as np
import tensorflow_datasets as tfds

_DESCRIPTION = """
Exploratory dataset from the

"Don't Change the Algorithm, Change the Data: Exploratory Data for Offline Reinforcement Learning"

paper by Denis Yarats, David Brandfonbrener, Hao Liu, Misha Laskin, Pieter Abbeel, Alessandro Lazaric, and Lerrel Pinto.

The datasets follow the RLDS format to represent steps and episodes.
"""

# TODO(exorl): BibTeX citation
_CITATION = """
@article{yarats2022exorl,
  title={Don't Change the Algorithm, Change the Data: Exploratory Data for Offline Reinforcement Learning},
  author={Denis Yarats, David Brandfonbrener, Hao Liu, Michael Laskin, Pieter Abbeel, Alessandro Lazaric, Lerrel Pinto},
  journal={arXiv preprint arXiv:2201.13425},
  year={2022}
}
"""

_HOMEPAGE = "https://github.com/denisyarats/exorl"
_BASE_URL = "https://dl.fbaipublicfiles.com/exorl/{domain}/{algorithm}.zip"

_DATASET_DESCRIPTION = """
See more details about the tasks and algorithms in https://github.com/denisyarats/exorl
"""


@dataclasses.dataclass
class ExorlBuilderConfig(tfds.core.BuilderConfig):
    algorithm: str = "proto"
    domain: str = "walker"


@dataclasses.dataclass
class DatasetConfig:
    domain: str
    obs_dim: int = 24
    action_dim: int = 6
    physics_dim: int = 18


_DATASET_CONFIGS = {
    "cartpole": DatasetConfig(
        domain="cartpole", obs_dim=5, action_dim=1, physics_dim=4
    ),
    "cheetah": DatasetConfig(
        domain="cheetah", obs_dim=17, action_dim=6, physics_dim=18
    ),
    "jaco": DatasetConfig(domain="jaco", obs_dim=42, action_dim=9, physics_dim=18),
    "quadruped": DatasetConfig(
        domain="quadruped", obs_dim=78, action_dim=12, physics_dim=57
    ),
    "point_mass_maze": DatasetConfig(
        domain="point_mass_maze", obs_dim=4, action_dim=2, physics_dim=4
    ),
    "walker": DatasetConfig(domain="walker", obs_dim=24, action_dim=6, physics_dim=18),
}


class Exorl(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for exorl dataset."""

    VERSION = tfds.core.Version("1.0.1")
    RELEASE_NOTES = {
        "1.0.1": "Fix issue with reward shape in point_mass_maze.",
        "1.0.0": "Initial release.",
    }

    # pylint: disable=unexpected-keyword-arg
    BUILDER_CONFIGS = [
        # proto
        ExorlBuilderConfig(
            name="cartpole_proto",
            algorithm="proto",
            domain="cartpole",
            description=_DATASET_DESCRIPTION,
        ),
        ExorlBuilderConfig(
            name="cheetah_proto",
            algorithm="proto",
            domain="cheetah",
            description=_DATASET_DESCRIPTION,
        ),
        ExorlBuilderConfig(
            name="jaco_proto",
            algorithm="proto",
            domain="jaco",
            description=_DATASET_DESCRIPTION,
        ),
        ExorlBuilderConfig(
            name="quadruped_proto",
            algorithm="proto",
            domain="quadruped",
            description=_DATASET_DESCRIPTION,
        ),
        ExorlBuilderConfig(
            name="point_mass_maze_proto",
            algorithm="proto",
            domain="point_mass_maze",
            description=_DATASET_DESCRIPTION,
        ),
        ExorlBuilderConfig(
            name="walker_proto",
            algorithm="proto",
            domain="walker",
            description=_DATASET_DESCRIPTION,
        ),
    ]

    # pylint: enable=unexpected-keyword-arg

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        dataset_config = _DATASET_CONFIGS[self.builder_config.domain]
        if self.builder_config.domain == "point_mass_maze":
            # Point mass is a multi-task environment
            reward_spec = tfds.features.Tensor(
                shape=(4,),
                encoding=tfds.features.Encoding.ZLIB,
                dtype=np.float32,
            )
        else:
            reward_spec = np.float32

        steps_dict = {
            "observation": tfds.features.Tensor(
                shape=(dataset_config.obs_dim,),
                encoding=tfds.features.Encoding.ZLIB,
                dtype=np.float32,
            ),
            "action": tfds.features.Tensor(
                shape=(dataset_config.action_dim,),
                encoding=tfds.features.Encoding.ZLIB,
                dtype=np.float32,
            ),
            "reward": reward_spec,
            "discount": np.float32,
            "physics": tfds.features.Tensor(
                shape=(dataset_config.physics_dim,),
                encoding=tfds.features.Encoding.ZLIB,
                dtype=np.float64,
            ),
            "is_terminal": np.bool_,
            "is_first": np.bool_,
            "is_last": np.bool_,
        }
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict(
                {
                    # These are the features of your dataset like images, labels ...
                    "episode_id": np.int64,
                    "steps": tfds.features.Dataset(steps_dict),
                }
            ),
            # If there's a common (input, target) tuple from the
            # features, specify them here. They'll be used if
            # `as_supervised=True` in `builder.as_dataset`.
            supervised_keys=None,  # disabled
            homepage=_HOMEPAGE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        dataset_url = _BASE_URL.format(
            domain=self.builder_config.domain, algorithm=self.builder_config.algorithm
        )
        extracted_path = dl_manager.download_and_extract({"file_path": dataset_url})

        return {
            "train": self._generate_examples(extracted_path),
        }

    def _generate_examples(self, path):
        """Yields examples."""
        buffer_path = path["file_path"] / "buffer"
        for i, f in enumerate(sorted(buffer_path.glob("*.npz"))):
            episode = _decode_episode(f)
            yield i, episode


def _decode_episode(filename):
    with filename.open("rb") as f:
        data = np.load(f)
        data = {k: data[k] for k in data.keys()}
        episode = {}
        # filenames has the format
        # episode_{episode_number}_{episode_length}.npz
        episode_id = int(filename.name.split("_")[1])
        episode["observation"] = data["observation"]
        episode["physics"] = data["physics"]
        # Change to SAR alignment following the RLDS convention
        episode["action"] = np.concatenate(
            (data["action"][1:], [np.zeros_like(data["action"][0])])
        )
        episode["reward"] = np.concatenate(
            (data["reward"][1:], [np.zeros_like(data["reward"][0])])
        )
        if episode["reward"].shape[-1] == 1:
            episode["reward"] = np.squeeze(episode["reward"], axis=-1)
        else:
            episode["reward"] = episode["reward"]
        episode["discount"] = np.concatenate(
            (data["discount"][1:], [np.zeros_like(data["discount"][0])])
        )
        episode["discount"] = np.squeeze(episode["discount"], axis=-1)
        episode["is_first"] = [True] + [False] * (len(data["observation"]) - 1)
        episode["is_last"] = [False] * (len(data["observation"]) - 1) + [True]
        episode["is_terminal"] = [False] * (len(data["observation"]))

        return {"episode_id": episode_id, "steps": episode}
