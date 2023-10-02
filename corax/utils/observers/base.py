"""Metrics observers."""

import abc
from typing import Dict, Union

import dm_env
import numpy as np

Number = Union[int, float]


class EnvLoopObserver(abc.ABC):
    """An interface for collecting metrics/counters in EnvironmentLoop."""

    @abc.abstractmethod
    def observe_first(self, env: dm_env.Environment, timestep: dm_env.TimeStep) -> None:
        """Observes the initial state."""

    @abc.abstractmethod
    def observe(
        self, env: dm_env.Environment, timestep: dm_env.TimeStep, action: np.ndarray
    ) -> None:
        """Records one environment step."""

    @abc.abstractmethod
    def get_metrics(self) -> Dict[str, Number]:
        """Returns metrics collected for the current episode."""
