"""Utility definitions for experiments."""
import os
from typing import Any, Dict, List, Mapping, Optional

from absl import logging

from corax.jax import utils as jax_utils
from corax.utils import loggers as loggers_lib
from corax.utils.loggers import wandb as wandb_logger


def _get_time_delta(time_delta: Optional[float], default_time_delta: float):
    if time_delta is not None:
        return time_delta
    else:
        return default_time_delta


class LoggerFactory:
    """Factory for creating loggers used in magi RL experiments."""

    def __init__(
        self,
        workdir: Optional[str] = None,
        log_to_terminal: Optional[bool] = None,
        log_to_wandb: bool = False,
        wandb_kwargs: Optional[Mapping[str, Any]] = None,
        time_delta: float = 1.0,
        actor_time_delta: Optional[float] = None,
        learner_time_delta: Optional[float] = None,
        evaluator_time_delta: Optional[float] = None,
        async_learner_logger: bool = False,
        add_uid: bool = True,
    ):
        """Create a logger factory."""

        wandb_kwargs = wandb_kwargs or {}
        self._log_to_wandb = log_to_wandb

        if log_to_terminal is None:
            log_to_terminal = os.environ.get("JOB_ID", None) is None

        self._log_to_terminal = log_to_terminal
        self._run = None
        if log_to_wandb and self._run is None:
            import wandb

            wandb.require("service")
            self._run = wandb.init(**wandb_kwargs)
        if workdir is not None:
            os.makedirs(workdir, exist_ok=True)
        self._workdir = workdir
        self._time_delta = time_delta
        self._actor_time_delta = actor_time_delta
        self._learner_time_delta = learner_time_delta
        self._evaluator_time_delta = evaluator_time_delta
        self._async_learner_logger = async_learner_logger
        self._add_uid = add_uid

    @property
    def run(self):
        return self._run

    def __call__(
        self,
        label: str,
        steps_key: Optional[str] = None,
        instance: Optional[int] = None,
    ):
        """Create an experiment logger."""
        if steps_key is None:
            steps_key = f"{label}_steps"

        if instance is None:
            instance = 0

        # Binding the experiment logger factory ensures that
        # the wandb run associated with the launching process gets
        # serialized and passed to workers correctly.
        if label in ("learner",):
            return self.make_default_logger(
                label=label,
                log_to_terminal=self._log_to_terminal,
                asynchronous=self._async_learner_logger,
                time_delta=_get_time_delta(self._learner_time_delta, self._time_delta),
                serialize_fn=jax_utils.fetch_devicearray,
                workdir=self._workdir,
                log_to_wandb=self._log_to_wandb,
                steps_key=steps_key,
                wandb_run=self._run,
                add_uid=self._add_uid,
            )
        elif label in ("evaluator",):
            return self.make_default_logger(
                label=label,
                log_to_terminal=self._log_to_terminal,
                time_delta=_get_time_delta(
                    self._evaluator_time_delta, self._time_delta
                ),
                steps_key=steps_key,
                workdir=self._workdir,
                log_to_wandb=self._log_to_wandb,
                wandb_run=self._run,
                add_uid=self._add_uid,
            )
        elif label in ("actor",):
            return self.make_default_logger(
                label=label,
                log_to_terminal=self._log_to_terminal,
                save_data=instance == 0,
                time_delta=_get_time_delta(self._actor_time_delta, self._time_delta),
                steps_key=steps_key,
                workdir=self._workdir,
                log_to_wandb=self._log_to_wandb,
                wandb_run=self._run,
                add_uid=self._add_uid,
            )
        else:
            logging.warning("Unknown label %s. Fallback to default.", label)
            return self.make_default_logger(
                label=label,
                log_to_terminal=self._log_to_terminal,
                steps_key=steps_key,
                time_delta=self._time_delta,
                workdir=self._workdir,
                log_to_wandb=self._log_to_wandb,
                wandb_run=self._run,
                add_uid=self._add_uid,
            )

    @staticmethod
    def make_default_logger(
        label: str,
        log_to_terminal: bool = True,
        save_data: bool = True,
        time_delta: float = 1.0,
        asynchronous: bool = False,
        print_fn=None,
        workdir: Optional[str] = None,
        serialize_fn=loggers_lib.to_numpy,
        steps_key: str = "steps",
        log_to_wandb: bool = False,
        wandb_kwargs: Optional[Dict[str, Any]] = None,
        add_uid: bool = False,
        wandb_run=None,
    ) -> loggers_lib.Logger:
        """Make a default Acme logger.
        Args:
          label: Name to give to the logger.
          save_data: Whether to persist data.
          time_delta: Time (in seconds) between logging events.
          asynchronous: Whether the write function should block or not.
          print_fn: How to print to terminal (defaults to print).
          serialize_fn: An optional function to apply to the write inputs before
            passing them to the various loggers.
          steps_key: Ignored.

        Returns:
          A logger object that responds to logger.write(some_dict).
        """
        if not print_fn:
            print_fn = logging.info

        loggers: List[loggers_lib.Logger] = []
        if log_to_terminal:
            terminal_logger = loggers_lib.TerminalLogger(label=label, print_fn=print_fn)
            loggers.append(terminal_logger)
        if save_data and workdir is not None:
            loggers.append(loggers_lib.CSVLogger(workdir, label=label, add_uid=add_uid))
        if save_data and log_to_wandb:
            if wandb_kwargs is None:
                wandb_kwargs = {}
            loggers.append(
                wandb_logger.WandbLogger(
                    label=label, steps_key=steps_key, run=wandb_run, **wandb_kwargs
                )
            )

        # Dispatch to all writers and filter Nones and by time.
        logger = loggers_lib.Dispatcher(loggers, serialize_fn)
        logger = loggers_lib.NoneFilter(logger)
        if asynchronous:
            logger = loggers_lib.AsyncLogger(logger)
        logger = loggers_lib.TimeFilter(logger, time_delta)
        return loggers_lib.AutoCloseLogger(logger)
