# type: ignore
# Copyright 2018 DeepMind Technologies Limited. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utility classes for saving model checkpoints and snapshots."""

import abc
import datetime
import pickle
import time
from typing import Mapping, Optional, Union

from absl import logging
import tensorflow as tf

from corax import core
from corax.utils import paths

# from corax.utils import signals

PythonState = tf.train.experimental.PythonState
Checkpointable = Union[tf.Module, tf.Variable, PythonState]

_DEFAULT_CHECKPOINT_TTL = int(datetime.timedelta(days=5).total_seconds())
_DEFAULT_SNAPSHOT_TTL = int(datetime.timedelta(days=90).total_seconds())


class TFSaveable(abc.ABC):
    """An interface for objects that expose their checkpointable TF state."""

    @property
    @abc.abstractmethod
    def state(self) -> Mapping[str, Checkpointable]:
        """Returns TensorFlow checkpointable state."""


class Checkpointer:
    """Convenience class for periodically checkpointing.

    This can be used to checkpoint any object with trackable state (e.g.
    tensorflow variables or modules); see tf.train.Checkpoint for
    details. Objects inheriting from tf.train.experimental.PythonState can also
    be checkpointed.

    Typically people use Checkpointer to make sure that they can correctly recover
    from a machine going down during learning. For more permanent storage of self-
    contained "networks" see the Snapshotter object.

    Usage example:

    ```python
    model = snt.Linear(10)
    checkpointer = Checkpointer(objects_to_save={'model': model})

    for _ in range(100):
      # ...
      checkpointer.save()
    ```
    """

    def __init__(
        self,
        objects_to_save: Mapping[str, Union[Checkpointable, core.Saveable]],
        *,
        directory: str = "~/acme/",
        subdirectory: str = "default",
        time_delta_minutes: float = 10.0,
        enable_checkpointing: bool = True,
        add_uid: bool = True,
        max_to_keep: int = 1,
        checkpoint_ttl_seconds: Optional[int] = _DEFAULT_CHECKPOINT_TTL,
        keep_checkpoint_every_n_hours: Optional[int] = None,
    ):
        """Builds the saver object.

        Args:
          objects_to_save: Mapping specifying what to checkpoint.
          directory: Which directory to put the checkpoint in.
          subdirectory: Sub-directory to use (e.g. if multiple checkpoints are being
            saved).
          time_delta_minutes: How often to save the checkpoint, in minutes.
          enable_checkpointing: whether to checkpoint or not.
          add_uid: If True adds a UID to the checkpoint path, see
            `paths.get_unique_id()` for how this UID is generated.
          max_to_keep: The maximum number of checkpoints to keep.
          checkpoint_ttl_seconds: TTL (time to leave) in seconds for checkpoints.
          keep_checkpoint_every_n_hours: keep_checkpoint_every_n_hours passed to
            tf.train.CheckpointManager.
        """

        # Convert `Saveable` objects to TF `Checkpointable` first, if necessary.
        def to_ckptable(x: Union[Checkpointable, core.Saveable]) -> Checkpointable:
            if isinstance(x, core.Saveable):
                return SaveableAdapter(x)
            return x

        objects_to_save = {k: to_ckptable(v) for k, v in objects_to_save.items()}

        self._time_delta_minutes = time_delta_minutes
        self._last_saved = 0.0
        self._enable_checkpointing = enable_checkpointing
        self._checkpoint_manager = None

        if enable_checkpointing:
            # Checkpoint object that handles saving/restoring.
            self._checkpoint = tf.train.Checkpoint(**objects_to_save)
            self._checkpoint_dir = paths.process_path(
                directory,
                "checkpoints",
                subdirectory,
                ttl_seconds=checkpoint_ttl_seconds,
                backups=False,
                add_uid=add_uid,
            )

            # Create a manager to maintain different checkpoints.
            self._checkpoint_manager = tf.train.CheckpointManager(
                self._checkpoint,
                directory=self._checkpoint_dir,
                max_to_keep=max_to_keep,
                keep_checkpoint_every_n_hours=keep_checkpoint_every_n_hours,
            )

            self.restore()

    def save(self, force: bool = False) -> bool:
        """Save the checkpoint if it's the appropriate time, otherwise no-ops.

        Args:
          force: Whether to force a save regardless of time elapsed since last save.

        Returns:
          A boolean indicating if a save event happened.
        """
        if not self._enable_checkpointing:
            return False

        if not force and time.time() - self._last_saved < 60 * self._time_delta_minutes:
            return False

        # Save any checkpoints.
        logging.info("Saving checkpoint: %s", self._checkpoint_manager.directory)
        self._checkpoint_manager.save()
        self._last_saved = time.time()

        return True

    def restore(self):
        # Restore from the most recent checkpoint (if it exists).
        checkpoint_to_restore = self._checkpoint_manager.latest_checkpoint
        logging.info("Attempting to restore checkpoint: %s", checkpoint_to_restore)
        self._checkpoint.restore(checkpoint_to_restore)

    @property
    def directory(self):
        return self._checkpoint_manager.directory


class CheckpointingRunner(core.Worker):
    """Wrap an object and expose a run method which checkpoints periodically.

    This internally creates a Checkpointer around `wrapped` object and exposes
    all of the methods of `wrapped`. Additionally, any `**kwargs` passed to the
    runner are forwarded to the internal Checkpointer.
    """

    def __init__(
        self,
        wrapped: Union[Checkpointable, core.Saveable, TFSaveable],
        key: str = "wrapped",
        *,
        time_delta_minutes: int = 30,
        **kwargs,
    ):
        if isinstance(wrapped, TFSaveable):
            # If the object to be wrapped exposes its TF State, checkpoint that.
            objects_to_save = wrapped.state
        else:
            # Otherwise checkpoint the wrapped object itself.
            objects_to_save = wrapped

        self._wrapped = wrapped
        self._time_delta_minutes = time_delta_minutes
        self._checkpointer = Checkpointer(
            objects_to_save={key: objects_to_save},
            time_delta_minutes=time_delta_minutes,
            **kwargs,
        )

    # Handle preemption signal. Note that this must happen in the main thread.
    def _signal_handler(self):
        logging.info("Caught SIGTERM: forcing a checkpoint save.")
        self._checkpointer.save(force=True)

    def step(self):
        if isinstance(self._wrapped, core.Learner):
            # Learners have a step() method, so alternate between that and ckpt call.
            self._wrapped.step()
            self._checkpointer.save()
        else:
            # Wrapped object doesn't have a run method; set our run method to ckpt.
            self.checkpoint()

    def run(self):
        # TODO(yl): Add back signal handling logic
        """Runs the checkpointer."""
        while True:
            self.step()

    def __dir__(self):
        return dir(self._wrapped) + ["get_directory"]

    # TODO(b/195915583) : Throw when wrapped object has get_directory() method.
    def __getattr__(self, name):
        if name == "get_directory":
            return self.get_directory
        return getattr(self._wrapped, name)

    def checkpoint(self):
        self._checkpointer.save()
        # Do not sleep for a long period of time to avoid LaunchPad program
        # termination hangs (time.sleep is not interruptible).
        for _ in range(self._time_delta_minutes * 60):
            time.sleep(1)

    def get_directory(self):
        return self._checkpointer.directory


class SaveableAdapter(tf.train.experimental.PythonState):
    """Adapter which allows `Saveable` object to be checkpointed by TensorFlow."""

    def __init__(self, object_to_save: core.Saveable):
        self._object_to_save = object_to_save

    def serialize(self):
        state = self._object_to_save.save()
        return pickle.dumps(state)

    def deserialize(self, pickled: bytes):
        state = pickle.loads(pickled)
        self._object_to_save.restore(state)
