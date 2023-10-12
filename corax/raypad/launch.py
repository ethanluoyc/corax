# type: ignore
import ray
import tree

from corax.raypad import actors
from corax.raypad import context
from corax.raypad.nodes import dereference
from corax.raypad.nodes import ray as ray_node
from corax.raypad.nodes import reverb as reverb_node


@ray.remote
class RayProgram:
    def __init__(self, program, ray_resources) -> None:
        for label, nodes in program.groups.items():
            launch_config = ray_resources.get(label, None)
            for node in nodes:
                node._initialize_context(
                    context.LaunchType.RAY, launch_config=launch_config
                )

        for label, nodes in program.groups.items():
            for node in nodes:
                for handle in node.input_handles:
                    handle.connect(node, label)

        # Bind addresses
        for label, nodes in program.groups.items():
            for i, node in enumerate(nodes):
                node.bind_addresses(name=f"{label}_{i}")

        self._should_stop = False
        task_handles = []
        actor_handles = []
        actor_tasks = []

        for node in program.get_all_nodes():
            resources = node.launch_context.launch_config or {}
            if isinstance(node, reverb_node.ReverbNode):
                reverb_handle = actors.ReverbActor.options(  # noqa
                    name=node.reverb_address.resolve(),
                    **resources,
                ).remote(node._priority_tables_fn)
                actor_handles.append(reverb_handle)
            elif isinstance(node, ray_node.RayNode):
                args, kwargs = tree.map_structure(
                    dereference.maybe_dereference, (node._args, node._kwargs)  # type: ignore
                )
                service_handle = actors.PyActor.options(
                    name=node._address.resolve(),
                    **resources,
                    max_concurrency=16,
                ).remote(node._constructor, *args, **kwargs)
                actor_handles.append(service_handle)
                if node._should_run and ray.get(service_handle.is_runnable.remote()):
                    actor_tasks.append(service_handle.call.remote("run"))

        self._task_handles = task_handles
        self._actor_handles = actor_handles
        self._actor_tasks = actor_tasks

    def wait(self):
        pending = self._task_handles + self._actor_tasks
        while True:
            _, pending = ray.wait(pending, timeout=10.0)
            if self._should_stop:
                print("Stopping now")
                for handle in self._task_handles:
                    ray.cancel(handle, force=True)
                for actor in self._actor_handles:
                    ray.kill(actor)
                return

    def stop(self):
        self._should_stop = True


def launch_ray(program, resources):
    ray_resources = resources or {}

    supervisor_handle = RayProgram.options(max_concurrency=4, name="program").remote(
        program, ray_resources
    )
    return ray.get(supervisor_handle.wait.remote())


def launch(program, resources=None):
    launch_ray(program, resources)
