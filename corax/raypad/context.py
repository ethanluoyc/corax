# type: ignore
"""Platform-specific configuration on the node."""

import enum
import threading
from typing import Any, Callable, Optional


class LaunchType(enum.Enum):
    """The different launch types supported by Launchpad.

    Launch type can be specified through `lp_launch_type` command line flag or
    by passing `launch_type` parameter to lp.launch() call.
    """

    RAY = "ray"


class LaunchContext(object):
    """Stores platform-specific launch config of a node.

    This is created and set on the node only at launch time.
    """

    def __init__(self):
        self._launch_type = None
        self._launch_config = None
        self._program_stopper = None
        self._is_initialized = False

    @property
    def launch_type(self) -> LaunchType:
        self._check_inititialized()
        return self._launch_type

    @property
    def launch_config(self) -> Any:
        self._check_inititialized()
        return self._launch_config

    @property
    def program_stopper(self) -> Callable[[], None]:
        self._check_inititialized()
        return self._program_stopper

    def _check_inititialized(self):
        if not self._is_initialized:
            raise RuntimeError(
                "Launch context is not yet initialized. It should be initialized by "
                "calling initialize() at launch time."
            )

    def initialize(
        self,
        launch_type: LaunchType,
        launch_config: Any,
        program_stopper: Optional[Callable[[], None]] = None,
    ):
        self._launch_config = launch_config
        self._launch_type = launch_type
        self._program_stopper = program_stopper
        self._is_initialized = True


_LAUNCH_CONTEXT = threading.local()


def get_context():
    context = getattr(_LAUNCH_CONTEXT, "lp_context", None)
    assert context, (
        "Launchpad context was not instantiated. Are you trying to "
        "access it outside of the node's main thread?"
    )
    return context


def set_context(context: LaunchContext):
    _LAUNCH_CONTEXT.lp_context = context
