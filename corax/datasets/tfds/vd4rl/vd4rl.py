"""vd4rl dataset."""
import dataclasses
import json
import os
from typing import Any, NamedTuple

import dm_env
import h5py
import numpy as np
import tensorflow_datasets as tfds
import tree

_DESCRIPTION = """
TODO
"""

_CITATION = """
TODO
"""

_HOMEPAGE = "TODO"

_DATASET_DESCRIPTION = """
TODO
"""


@dataclasses.dataclass
class Vd4rlBuilderConfig(tfds.core.BuilderConfig):
    prefix: str = "main"
    env: str = "walker_walk"
    quality: str = "expert"
    resolution: int = 84


QUALITIES = ["expert", "medium", "medium_replay", "medium_expert", "random"]
ENVS = ["walker_walk", "cheetah_run", "humanoid_walk"]


def _get_all_configs():
    configs = []
    for env_name in ENVS:
        for quality in QUALITIES:
            # pylint: disable=unexpected-keyword-arg
            configs.append(
                Vd4rlBuilderConfig(
                    name=f"main_{env_name}_{quality}_84px",
                    prefix="main",
                    env=env_name,
                    quality=quality,
                    resolution=84,
                )
            )
            # pylint: enable=unexpected-keyword-arg
    return configs


_BUILDER_CONFIGS = _get_all_configs()


class Vd4rl(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for exorl dataset."""

    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {
        "1.0.0": "Initial release.",
    }

    # pylint: disable=unexpected-keyword-arg
    BUILDER_CONFIGS = _BUILDER_CONFIGS

    # pylint: enable=unexpected-keyword-arg

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        resolution = self.builder_config.resolution
        action_dim = {"walker_walk": 6, "humanoid_walk": 21, "cheetah_run": 6}[
            self.builder_config.env
        ]
        steps_dict = {
            "observation": tfds.features.Image(
                shape=(resolution, resolution, 3), encoding_format="png", dtype=np.uint8
            ),
            "action": tfds.features.Tensor(
                shape=(action_dim,),
                encoding=tfds.features.Encoding.ZLIB,
                dtype=np.float32,
            ),
            "reward": np.float32,
            "discount": np.float32,
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
                    # 'episode_id': tf.int64,
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
        manifest_path = os.path.join(
            os.path.abspath(os.path.dirname(__file__)), "manifest.json"
        )
        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)
        prefix = self.builder_config.prefix
        env_name = self.builder_config.env
        quality = self._builder_config.quality
        resolution = self._builder_config.resolution
        key = f"{prefix}/{env_name}/{quality}/{resolution}px"
        filenames = manifest[key]
        files = {
            fname: f"https://huggingface.co/datasets/conglu/vd4rl/resolve/main/vd4rl/{key}/{fname}"
            for fname in filenames
        }
        downloaded_files = dl_manager.download(files)
        filenames = sorted(downloaded_files.keys())
        files = [downloaded_files[k] for k in filenames]
        return {
            "train": self._generate_examples(files),
        }

    def _generate_examples(self, paths):
        """Yields examples."""
        current_episode = None

        counter = 0
        for step_data in _load_trajectories(paths):
            # action is the action that generated the timestep
            if step_data.timestep.first():
                if current_episode is not None:
                    # episodes.append(current_episode)
                    rlds_steps = convert_to_rlds_steps(current_episode)
                    rlds_steps = tree.map_structure(
                        lambda *xs: np.stack(xs), *rlds_steps
                    )
                    yield counter, {"steps": rlds_steps}
                    counter += 1
                    current_episode = None

            if current_episode is None:
                current_episode = [step_data]
            else:
                current_episode.append(step_data)

        if current_episode is not None:
            rlds_steps = convert_to_rlds_steps(current_episode)
            rlds_steps = tree.map_structure(lambda *xs: np.stack(xs), *rlds_steps)
            yield counter, {"steps": rlds_steps}
            current_episode = None


step_type_lookup = {
    0: dm_env.StepType.FIRST,
    1: dm_env.StepType.MID,
    2: dm_env.StepType.LAST,
}


def _get_timestep_from_idx(offline_data: dict, idx: int):
    # Convert to channel last
    observation = np.transpose(offline_data["observation"][idx], [1, 2, 0])
    timestep = dm_env.TimeStep(
        step_type=step_type_lookup[offline_data["step_type"][idx]],
        reward=offline_data["reward"][idx].astype(np.float32),
        observation=observation,
        discount=offline_data["discount"][idx].astype(np.float32),
    )
    action = offline_data["action"][idx]
    return timestep, action


class _StepData(NamedTuple):
    timestep: dm_env.TimeStep
    action: Any = None


def _load_trajectories(filenames):
    for filename in filenames:
        episodes = h5py.File(filename, "r")
        episodes = {k: v[:] for k, v in episodes.items()}
        offline_data = episodes
        offline_data_length = offline_data["reward"].shape[0]
        for v in offline_data.values():
            assert v.shape[0] == offline_data_length
        for idx in range(offline_data_length):
            time_step, action = _get_timestep_from_idx(offline_data, idx)
            yield _StepData(time_step, action)


## https://github.com/deepmind/envlogger/blob/main/envlogger/backends/rlds_utils.py
def to_rlds_step(prev_step, step: None):
    """Builds an RLDS step from two Envlogger steps.
    Steps follow the RLDS convention from https://github.com/google-research/rlds.
    Args:
      prev_step: previous step.
      step: current step. If None, it builds the last step (where the observation
        is the last one, and the action, reward and discount are undefined).
    Returns:
       RLDS Step.
    """
    metadata = {}
    return {
        "action": step.action
        if step
        else tree.map_structure(np.zeros_like, prev_step.action),
        "discount": step.timestep.discount
        if step
        else tree.map_structure(np.zeros_like, prev_step.timestep.discount),
        "is_first": prev_step.timestep.first(),
        "is_last": prev_step.timestep.last(),
        "is_terminal": (
            prev_step.timestep.last() and prev_step.timestep.discount == 0.0
        ),
        "observation": prev_step.timestep.observation,
        "reward": step.timestep.reward
        if step
        else tree.map_structure(np.zeros_like, prev_step.timestep.reward),
        **metadata,
    }


def convert_to_rlds_steps(episode):
    prev_steps = episode[:-1]
    steps = episode[1:]
    rlds_steps = []
    for prev_step, step in zip(prev_steps, steps):
        rlds_steps.append(to_rlds_step(prev_step, step))
    rlds_steps.append(to_rlds_step(episode[-1], None))
    return rlds_steps
