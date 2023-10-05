import time
from typing import Any, Callable, Generic, Iterator, List, Optional, Sequence

import jax

from corax import core
from corax import types
from corax.agents.jax import learner_core as learner_core_lib
from corax.jax import types as jax_types
from corax.jax import utils as jax_utils
from corax.utils import counting
from corax.utils import loggers


class DefaultJaxLearner(
    core.Learner, Generic[jax_types.TrainingState, jax_types.Sample]
):
    """A generic JAX learner that wraps a pure LearnerCore."""

    _state: jax_types.TrainingState

    def __init__(
        self,
        learner_core: learner_core_lib.LearnerCore[
            jax_types.TrainingState, jax_types.Sample
        ],
        key: jax_types.PRNGKey,
        iterator: Iterator[jax_types.Sample],
        logger: Optional[loggers.Logger] = None,
        counter: Optional[counting.Counter] = None,
        num_sgd_steps_per_step: int = 1,
        postprocess_aux: Optional[Callable[[Any], Any]] = None,
    ) -> None:
        self._learner_core = learner_core
        self._iterator = iterator
        self._state = self._learner_core.init(key)

        update_step = jax_utils.process_multiple_batches(
            self._learner_core.step,
            num_sgd_steps_per_step,
            postprocess_aux=postprocess_aux,
        )
        self._update_step = jax.jit(update_step)

        self._logger = logger or loggers.make_default_logger(
            "learner", asynchronous=True, serialize_fn=jax_utils.fetch_devicearray
        )
        self._counter = counter or counting.Counter()

        self._timestamp = None

    def step(self):
        sample = next(self._iterator)
        self._state, metrics = self._update_step(self._state, sample)

        # Compute elapsed time.
        timestamp = time.time()
        elapsed_time = timestamp - self._timestamp if self._timestamp else 0
        self._timestamp = timestamp

        counts = self._counter.increment(steps=1, walltime=elapsed_time)

        if elapsed_time > 0:
            metrics["steps_per_second"] = 1 / elapsed_time
        else:
            metrics["steps_per_second"] = 0.0

        if self._logger:
            self._logger.write({**counts, **metrics})

    def get_variables(self, names: Sequence[str]) -> List[types.NestedArray]:
        return self._learner_core.get_variables(self._state, names)

    def save(self) -> Any:
        return self._state

    def restore(self, state: Any) -> None:
        self._state = state
