"""Runners used for executing local agents."""

import time
from typing import Callable

import dm_env

from corax import core
from corax import types


class LocalLayout(core.Actor):
    """Actor which learns (updates its parameters) when `update` is called.

    This combines a base actor and a learner. Whenever `update` is called
    on the wrapping actor the learner will take a step (e.g. one step of gradient
    descent) as long as there is data available for training
    (provided iterator and replay_tables are used to check for that).
    Selecting actions and making observations are handled by the base actor.
    Intended to be used by the `run_experiment` only.
    """

    def __init__(
        self,
        actor: core.Actor,
        learner: core.Learner,
        iterator: core.PrefetchingIterator,
        can_sample: Callable[[], bool],
    ):
        """Initializes _LearningActor.

        Args:
          actor: Actor to be wrapped.
          learner: Learner on which step() is to be called when there is data.
          iterator: Iterator used by the Learner to fetch training data.
          replay_tables: Collection of tables from which Learner fetches data
            through the iterator.
          sample_sizes: For each table from `replay_tables`, how many elements the
            table should have available for sampling to wait for the `iterator` to
            prefetch a batch of data. Otherwise more experience needs to be
            collected by the actor.
          checkpointer: Checkpointer to save the state on update.
        """
        self._actor = actor
        self._learner = learner
        self._iterator = iterator
        self._learner_steps = 0
        self._can_sample = can_sample

    def select_action(self, observation: types.NestedArray) -> types.NestedArray:
        return self._actor.select_action(observation)

    def observe_first(self, timestep: dm_env.TimeStep):
        self._actor.observe_first(timestep)

    def observe(self, action: types.NestedArray, next_timestep: dm_env.TimeStep):
        self._actor.observe(action, next_timestep)

    def _maybe_train(self):
        trained = False
        while True:
            if self._iterator.ready():
                self._learner.step()
                batches = self._iterator.retrieved_elements() - self._learner_steps
                self._learner_steps += 1
                assert batches == 1, (
                    "Learner step must retrieve exactly one element from the iterator"
                    f" (retrieved {batches}). Otherwise agent can deadlock. Example "
                    "cause is that your chosen agent"
                    "s Builder has a `make_learner` "
                    "factory that prefetches the data but it shouldn"
                    "t."
                )
                trained = True
            else:
                # Wait for the iterator to fetch more data from the table(s) only
                # if there plenty of data to sample from each table.
                if not self._can_sample():
                    return trained
                # Let iterator's prefetching thread get data from the table(s).
                time.sleep(0.001)

    def update(
        self,
    ):  # pytype: disable=signature-mismatch  # overriding-parameter-count-checks
        if self._maybe_train():
            # Update the actor weights only when learner was updated.
            self._actor.update()
