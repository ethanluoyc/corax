"""Reverb replay buffers."""


from typing import Callable, Optional, Sequence

import ray
import reverb
from absl import logging

from corax.raypad import address as lp_address
from corax.raypad.nodes import addressing
from corax.raypad.nodes import base
from corax.raypad.nodes import python

PriorityTablesFactory = Callable[[], Sequence[reverb.Table]]
CheckpointerFactory = Callable[[], reverb.checkpointers.CheckpointerBase]

REVERB_PORT_NAME = "reverb"


class ReverbHandle(base.Handle):
    def __init__(self, address: lp_address.Address):
        self._address = address

    def dereference(self):
        address = self._address.resolve()
        actor_handle = ray.get_actor(self._address.resolve())
        address = ray.get(actor_handle.get_reverb_address.remote())
        logging.info("Reverb client connecting to: %s", address)
        return reverb.Client(address)


class ReverbNode(python.PyNode):
    """Represents a Reverb replay buffer in a Launchpad program."""

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
        super().__init__(self.run)
        self._priority_tables_fn = priority_tables_fn
        self._checkpoint_ctor = checkpoint_ctor
        self._checkpoint_time_delta_minutes = checkpoint_time_delta_minutes
        self._address = lp_address.Address(REVERB_PORT_NAME)
        self.allocate_address(self._address)

        if (
            self._checkpoint_time_delta_minutes is not None
            and self._checkpoint_time_delta_minutes <= 0
        ):
            raise ValueError(
                "Replay checkpoint time delta must be positive when specified."
            )

    def create_handle(self):
        return self._track_handle(ReverbHandle(self._address))

    def run(self):
        priority_tables = self._priority_tables_fn()
        if self._checkpoint_ctor is None:
            checkpointer = None
        else:
            checkpointer = self._checkpoint_ctor()

        self._server = reverb.Server(
            tables=priority_tables,
            port=lp_address.get_port_from_address(self._address.resolve()),
            checkpointer=checkpointer,
        )

    def bind_addresses(self, name):
        assert len(self.addresses) == 1
        self.addresses[0].bind(addressing.RayAddressBuilder(name))

    @staticmethod
    def to_executables(nodes: Sequence["ReverbNode"], label: str, launch_context):
        return python.PyNode.to_executables(nodes, label, launch_context)

    @property
    def reverb_address(self) -> lp_address.Address:
        return self._address
