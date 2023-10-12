# type: ignore
import ray
import tree

from corax.raypad import actors
from corax.raypad import address as rp_address
from corax.raypad.nodes import addressing
from corax.raypad.nodes import dereference
from corax.raypad.nodes.base import Handle
from corax.raypad.nodes.python import PyClassNode


class RayHandle(Handle):
    def __init__(self, address: str):
        self._address = address

    def dereference(self) -> str:
        actor_handle = ray.get_actor(self._address.resolve())
        return actors.PyActorClient(actor_handle)


class RayNode(PyClassNode):
    def __init__(self, constructor, *args, **kwargs) -> None:
        super().__init__(self.run)
        self._constructor = constructor
        self._args = args
        self._kwargs = kwargs
        self._address = rp_address.Address()
        self.allocate_address(self._address)
        self._collect_input_handles()
        self._server = None
        self._run_handle = None

    def create_handle(self) -> RayHandle:
        """See `Node.create_handle`."""
        return self._track_handle(RayHandle(self._address))

    def bind_addresses(self, name):
        assert len(self.addresses) == 1
        self.addresses[0].bind(addressing.RayAddressBuilder(name))

    def run(self):
        args, kwargs = tree.map_structure(
            dereference.maybe_dereference, (self._args, self._kwargs)  # type: ignore
        )
        actor_name = self._address.resolve()
        resources = self._launch_context.launch_config or {}

        self._server = actors.PyActorClient.options(
            max_concurrency=16, name=actor_name, **resources
        ).remote(self._constructor, *args, **kwargs)
        if ray.get(self._server.is_runnable.remote()):
            self._run_handle = self._server.call.remote("run")
