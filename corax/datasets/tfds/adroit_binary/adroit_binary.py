"""adroit_binary dataset."""

import dataclasses
import os

import numpy as np
import tensorflow_datasets as tfds
import tree


@dataclasses.dataclass
class BuilderConfig(tfds.core.BuilderConfig):
    filename: str = ""


@dataclasses.dataclass
class DatasetConfig:
    task: str
    obs_dim: int
    action_dim: int


_DATASET_CONFIGS = {
    "pen": DatasetConfig(task="pen", obs_dim=45, action_dim=24),
    "door": DatasetConfig(task="door", obs_dim=39, action_dim=28),
    "relocate": DatasetConfig(task="relocate", obs_dim=39, action_dim=30),
}


class AdroitBinary(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for adroit_binary dataset."""

    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {
        "1.0.0": "Initial release.",
    }

    # pylint: disable=unexpected-keyword-arg
    BUILDER_CONFIGS = [
        BuilderConfig(name="pen-binary-expert-v0", filename="pen2_sparse.npy"),
        BuilderConfig(name="door-binary-expert-v0", filename="door2_sparse.npy"),
        BuilderConfig(
            name="relocate-binary-expert-v0", filename="relocate2_sparse.npy"
        ),
        BuilderConfig(name="pen-binary-bc-v0", filename="pen_bc_sparse4.npy"),
        BuilderConfig(name="door-binary-bc-v0", filename="door_bc_sparse4.npy"),
        BuilderConfig(name="relocate-binary-bc-v0", filename="relocate_bc_sparse4.npy"),
    ]

    # pylint: enable=unexpected-keyword-arg

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        task = self.builder_config.name.split("-")[0]
        dataset_config = _DATASET_CONFIGS[task]

        action_dtype = np.float64
        if "bc-v0" in self.builder_config.name:
            action_dtype = np.float32

        steps_dict = {
            "observation": tfds.features.Tensor(
                shape=(dataset_config.obs_dim,),
                encoding=tfds.features.Encoding.ZLIB,
                dtype=np.float64,
            ),
            "action": tfds.features.Tensor(
                shape=(dataset_config.action_dim,),
                encoding=tfds.features.Encoding.ZLIB,
                dtype=action_dtype,
            ),
            "reward": np.float64,
            # "discount": np.float32,
            "is_terminal": np.bool_,
            "is_first": np.bool_,
            "is_last": np.bool_,
        }
        if "bc-v0" in self.builder_config.name:
            steps_dict["infos"] = {
                "goal_achieved": tfds.features.Tensor(shape=(), dtype=np.bool_),
                "env_reward": tfds.features.Tensor(shape=(), dtype=np.float64),
            }
        return tfds.core.DatasetInfo(
            builder=self,
            description="TODO",
            features=tfds.features.FeaturesDict(
                {
                    # These are the features of your dataset like images, labels ...
                    # "episode_id": np.int64,
                    "steps": tfds.features.Dataset(steps_dict),
                }
            ),
            # If there's a common (input, target) tuple from the
            # features, specify them here. They'll be used if
            # `as_supervised=True` in `builder.as_dataset`.
            supervised_keys=None,  # disabled
            homepage="""TODO""",
            citation="""TODO""",
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        dataset_url = "https://drive.google.com/uc?export=download&id=1SsVaQKZnY5UkuR78WrInp9XxTdKHbF0x"
        extracted_path = dl_manager.download_and_extract({"file_path": dataset_url})
        file_path = extracted_path["file_path"] / self.builder_config.filename

        # TODO(exorl): Returns the Dict[split names, Iterator[Key, Example]]
        return {
            "train": self._generate_examples(file_path),
        }

    def _generate_examples(self, path):
        """Yields examples."""
        # TODO(exorl): Yields (key, example) tuples from the dataset
        if "bc_sparse" in os.fspath(path):
            for i, episode in enumerate(load_bc_data(path)):
                yield i, {"steps": episode}
        else:
            for i, episode in enumerate(load_expert_data(path)):
                yield i, {"steps": episode}


def load_expert_data(filename):
    dataset = np.load(filename, allow_pickle=True)
    episodes = []
    for episode in dataset:
        observations = tree.map_structure(
            lambda *x: np.stack(x), *episode["observations"]
        )
        actions = tree.map_structure(lambda *x: np.stack(x), *episode["actions"])
        next_observations = tree.map_structure(
            lambda *x: np.stack(x), *episode["next_observations"]
        )
        rewards = tree.map_structure(lambda *x: np.stack(x), *episode["rewards"])
        terminals = tree.map_structure(lambda *x: np.stack(x), *episode["terminals"])
        agent_infos = tree.map_structure(
            lambda *x: np.stack(x), *episode["agent_infos"]
        )
        env_infos = tree.map_structure(lambda *x: np.stack(x), *episode["env_infos"])
        assert len(agent_infos) == 0
        assert len(env_infos) == 0
        # Expert data has one additional reward
        assert terminals.shape[0] == rewards.shape[0] - 1
        # Expert data observation is redundant
        np.testing.assert_allclose(
            observations["observation"], observations["state_observation"]
        )
        np.testing.assert_allclose(
            next_observations["observation"], next_observations["state_observation"]
        )

        # Assert next observation is the same as the previous observation offset by 1
        np.testing.assert_allclose(
            observations["observation"][1:], next_observations["observation"][:-1]
        )
        np.testing.assert_allclose(
            observations["state_observation"][1:],
            next_observations["state_observation"][:-1],
        )
        episodes.append(
            {
                "is_first": np.asarray(
                    [True] + observations["observation"].shape[0] * [False]
                ),
                "is_last": np.asarray(
                    ([False] * observations["observation"].shape[0] + [True])
                ),
                "is_terminal": np.concatenate([[False], terminals], axis=0),
                "observation": np.concatenate(
                    [
                        observations["observation"],
                        next_observations["observation"][-1:],
                    ],
                    axis=0,
                ),
                "action": np.concatenate(
                    [actions, np.zeros_like(actions[-1:])], axis=0
                ),
                "reward": rewards,
            }
        )
    return episodes


def load_bc_data(filename):
    dataset = np.load(filename, allow_pickle=True)
    episodes = []
    for episode in dataset:
        observations = tree.map_structure(
            lambda *x: np.stack(x), *episode["observations"]
        )
        actions = tree.map_structure(lambda *x: np.stack(x), *episode["actions"])
        next_observations = tree.map_structure(
            lambda *x: np.stack(x), *episode["next_observations"]
        )
        rewards = tree.map_structure(lambda *x: np.concatenate(x), *episode["rewards"])
        # terminals in BC data is rank 1
        terminals = tree.map_structure(
            lambda *x: np.concatenate(x), *episode["terminals"]
        )
        assert terminals.shape[0] == rewards.shape[0]
        agent_infos = tree.map_structure(
            lambda *x: np.stack(x), *episode["agent_infos"]
        )
        # env_infos = tree.map_structure(lambda *x: np.stack(x), *episode["env_infos"])
        np.testing.assert_allclose(observations[1:], next_observations[:-1])
        goal_achieved = np.stack(
            [info["goal_achieved"] for info in episode["env_infos"]]
        )
        env_reward = np.stack([info["env_reward"] for info in episode["env_infos"]])
        is_truncated = episode["env_infos"][-1]["TimeLimit.truncated"]
        # Check that terminals is actually referring to done in Gym 21 API.
        assert terminals[-1] == is_truncated
        # assertion passed, update terminal to use truncation
        terminals[-1] = not is_truncated
        assert len(agent_infos) == 0
        episodes.append(
            {
                "is_first": np.asarray([True] + observations.shape[0] * [False]),
                "is_last": np.asarray(([False] * observations.shape[0] + [True])),
                "is_terminal": np.concatenate([[False], terminals], axis=0),
                "reward": np.concatenate(
                    [rewards, np.zeros_like(rewards[-1:])], axis=0
                ),
                "observation": np.concatenate(
                    [
                        observations,
                        next_observations[-1:],
                    ],
                    axis=0,
                ),
                "action": np.concatenate(
                    [actions, np.zeros_like(actions[-1:])], axis=0
                ),
                "infos": {
                    "goal_achieved": np.concatenate(
                        [goal_achieved, np.zeros_like(goal_achieved[-1:])], axis=0
                    ),
                    "env_reward": np.concatenate(
                        [env_reward, np.zeros_like(env_reward[-1:])], axis=0
                    ),
                },
            }
        )
    return episodes
