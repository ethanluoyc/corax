import abc
from typing import Generic, List, Sequence, Tuple

from corax import types
from corax.jax import types as jax_types


class LearnerCore(abc.ABC, Generic[jax_types.TrainingState, jax_types.Sample]):
    @abc.abstractmethod
    def init(self, key: jax_types.PRNGKey) -> jax_types.TrainingState:
        ...

    @abc.abstractmethod
    def step(
        self, state: jax_types.TrainingState, batch: jax_types.Sample
    ) -> Tuple[jax_types.TrainingState, jax_types.TrainingMetrics]:
        ...

    @abc.abstractmethod
    def get_variables(
        self, state: jax_types.TrainingState, names: Sequence[str]
    ) -> List[types.NestedArray]:
        ...
