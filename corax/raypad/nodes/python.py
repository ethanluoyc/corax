# Copyright 2020 DeepMind Technologies Limited. All rights reserved.
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

"""Nodes that run user-defined Python code.

PyNode runs a user-defined function. PyClassNode constructs a Python object and
calls its run() method (if provided).

"""

import functools
import itertools
from typing import Any, Callable, Generic, TypeVar

import tree
from absl import flags
from absl import logging

from corax.raypad.nodes import base
from corax.raypad.nodes import dereference

FLAGS = flags.FLAGS


T = TypeVar("T")
HandleType = TypeVar("HandleType", bound=base.Handle)
ReturnType = TypeVar("ReturnType")
WorkerType = TypeVar("WorkerType")


class _DummyHandle(base.Handle[Any]):
    def dereference(self) -> None:
        raise NotImplementedError("_DummyHandle cannot be dereferenced.")


class PyNode(base.Node[HandleType], Generic[HandleType, ReturnType]):
    """Runs a user-defined Python function."""

    # Used only for no-serialization launch
    NEXT_PY_NODE_ID = itertools.count()

    def __init__(self, function: Callable[..., ReturnType], *args, **kwargs):
        super().__init__()
        self._func_args = args
        self._func_kwargs = kwargs
        self._function = function
        self._partial_function = self._construct_function
        self.py_node_id = next(PyNode.NEXT_PY_NODE_ID)

        # Find input handles and put them in self._input_handles.
        tree.traverse(
            functools.partial(base.extract_handles, handles=self._input_handles),
            (self._func_args, self._func_kwargs),
        )

    def _construct_function(self):
        args, kwargs = tree.map_structure(
            dereference.maybe_dereference, (self._func_args, self._func_kwargs)  # type: ignore
        )
        return functools.partial(self._function, *args, **kwargs)()

    def create_handle(self) -> HandleType:
        """Doesn't expose an interface for others to interact with it."""
        return _DummyHandle()  # type: ignore

    @property
    def function(self) -> Callable[..., ReturnType]:
        return self._partial_function

    @staticmethod
    def to_executables(nodes, label, launch_context):
        """Creates Executables."""
        raise NotImplementedError()

    def bind_addresses(self, **kwargs):
        raise NotImplementedError


class PyClassNode(PyNode[HandleType, type(None)], Generic[HandleType, WorkerType]):
    """Instantiates a Python object and runs its run() method (if provided).

    If disable_run() is called before launch, instance.run() method won't be
    called. This is useful in TAP-based integration tests, where users might need
    to step each worker synchronously.
    """

    def __init__(self, constructor: Callable[..., WorkerType], *args, **kwargs):
        """Initializes a new instance of the `PyClassNode` class.

        Args:
          constructor: A function that when called returns a Python object with a
            run method.
          *args: Arguments passed to the constructor.
          **kwargs: Key word arguments passed to the constructor.
        """
        super().__init__(self.run)
        self._constructor = constructor
        self._args = args
        self._kwargs = kwargs
        self._should_run = True
        self._collect_input_handles()
        self._instance = None

    def _collect_input_handles(self):
        self._input_handles.clear()
        try:
            # Find input handles and put them in self._input_handles.
            fn = functools.partial(base.extract_handles, handles=self._input_handles)
            _ = [fn(x) for x in tree.flatten((self._args, self._kwargs))]
        except TypeError as e:
            raise ValueError(
                f"Failed to construct the {self.__class__.__name__} with\n"
                f"- constructor: {self._constructor}\n"
                f"- args: {self._args}\n- kwargs: {self._kwargs}"
            ) from e

    def _construct_instance(self) -> WorkerType:
        if self._instance is None:
            args, kwargs = tree.map_structure(
                dereference.maybe_dereference, (self._args, self._kwargs)  # type: ignore
            )
            self._instance = self._constructor(*args, **kwargs)
        return self._instance

    def disable_run(self) -> None:
        """Prevents the node from calling `run` on the Python object.

        Note that the Python object is still constructed even if `disable_run` has
        been called.
        """
        self._should_run = False

    def enable_run(self) -> None:
        """Ensures `run` is called on the Python object.

        This is the default state and callers don't need to call `enable_run` unless
        `disable_run` has been called.
        """
        self._should_run = True

    def run(self) -> None:
        """Constructs Python object and (maybe) calls its `run` method.

        The `run` method is not called if `disable_run` has ben called previously or
        if the constructed Python object does not have a `run` method.
        """
        instance = self._construct_instance()
        if hasattr(instance, "run") and self._should_run:
            instance.run()  # type: ignore
        else:
            logging.warning(
                "run() not defined on the instance (or disable_run() was called.)."
                "Exiting..."
            )
