from typing import Callable, Optional, Sequence

import portpicker
import ray
import ray.util.state
import reverb


def _should_expose_method(func, method_name):
    return callable(func) and (
        not method_name.startswith("_") or method_name == "__call__"
    )


@ray.remote
class PyActor:
    def __init__(self, constructor, *args, **kwargs) -> None:
        self._handlers = {}
        self._instance = constructor(*args, **kwargs)
        for method_name in dir(self._instance):
            func = getattr(self._instance, method_name)
            if _should_expose_method(func, method_name):
                print("Binding method %s" % method_name)
                self.bind_handler(method_name, func)

    def bind_handler(self, name, handler):
        self._handlers[name] = handler

    def is_runnable(self):
        return "run" in self._handlers.keys()

    def call(self, name, *args, **kwargs):
        return self._handlers[name](*args, **kwargs)


class _AsyncClient:
    def __init__(self, client: "PyActorClient") -> None:
        self._client = client

    def _build_handler(self, method: str):
        def call(*args, **kwargs):
            return self._client._actor_handle.call.remote(
                method, *args, **kwargs
            ).future()

        return call

    def __getattr__(self, method):
        """Gets a callable function for the method that returns a future.

        Args:
        method: Name of the method.

        Returns:
        Callable function for the method that returns a future.
        """
        return self._build_handler(method)

    def __call__(self, *args, **kwargs):
        return self._build_handler("__call__")(*args, **kwargs)


class PyActorClient:
    def __init__(self, actor_handle) -> None:
        self._init_args = (actor_handle,)
        self._actor_handle = actor_handle

        self._async_client = _AsyncClient(self)

    def __reduce__(self):
        return self.__class__, self._init_args

    def _build_handler(self, name):
        """Create a handler for a remote method."""

        def handler(*args, **kwargs):
            return ray.get(self._actor_handle.call.remote(name, *args, **kwargs))

        return handler

    def __getattr__(self, name):
        func = self._build_handler(name)
        setattr(self, name, func)
        return func

    def __call__(self, *args, **kwargs):
        return self._build_handler("__call__")(*args, **kwargs)

    @property
    def futures(self):
        return self._async_client


PriorityTablesFactory = Callable[[], Sequence[reverb.Table]]
CheckpointerFactory = Callable[[], reverb.checkpointers.CheckpointerBase]


@ray.remote
class ReverbActor:
    def __init__(
        self,
        priority_tables_fn: PriorityTablesFactory,
        checkpoint_ctor: Optional[CheckpointerFactory] = None,
        checkpoint_time_delta_minutes: Optional[int] = None,
    ):
        """Initialize a ReverbNode.

        Args:
          priority_tables_fn: A function that returns a sequence of tables to host
            on the server.
          checkpoint_ctor: Constructor for the checkpointer to be used. Passing None
            uses Reverb's default checkpointer.
          checkpoint_time_delta_minutes: Time between async (non-blocking)
            checkpointing calls.
        """
        self._priority_tables_fn = priority_tables_fn
        self._checkpoint_ctor = checkpoint_ctor
        self._checkpoint_time_delta_minutes = checkpoint_time_delta_minutes

        priority_tables = self._priority_tables_fn()
        if self._checkpoint_ctor is None:
            checkpointer = None
        else:
            checkpointer = self._checkpoint_ctor()

        self._server = reverb.Server(
            tables=priority_tables,
            port=portpicker.pick_unused_port(),
            checkpointer=checkpointer,
        )

    def stop(self):
        """Stop the server."""
        self._server.stop()

    def get_reverb_address(self):
        return f"{ray.util.get_node_ip_address()}:{self._server.port}"
