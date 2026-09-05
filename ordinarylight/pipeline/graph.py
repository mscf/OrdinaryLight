"""Single-queue graph compilation with explicit versions and native alias checks."""

from dataclasses import dataclass, replace
import heapq

import vulkan as vk

from .vulkan import VulkanPass, VulkanPassPipeline, VulkanResource, VulkanResourceUse

_WRITE = (
    vk.VK_ACCESS_SHADER_WRITE_BIT
    | vk.VK_ACCESS_TRANSFER_WRITE_BIT
    | vk.VK_ACCESS_MEMORY_WRITE_BIT
    | vk.VK_ACCESS_HOST_WRITE_BIT
    | vk.VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR
    | vk.VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT
    | vk.VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT
)
_READ = (
    vk.VK_ACCESS_SHADER_READ_BIT
    | vk.VK_ACCESS_TRANSFER_READ_BIT
    | vk.VK_ACCESS_MEMORY_READ_BIT
    | vk.VK_ACCESS_HOST_READ_BIT
    | vk.VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR
    | vk.VK_ACCESS_UNIFORM_READ_BIT
    | vk.VK_ACCESS_INDIRECT_COMMAND_READ_BIT
    | vk.VK_ACCESS_INDEX_READ_BIT
    | vk.VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT
    | vk.VK_ACCESS_INPUT_ATTACHMENT_READ_BIT
    | vk.VK_ACCESS_COLOR_ATTACHMENT_READ_BIT
    | vk.VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT
)


def _key(resource):
    return resource.kind, resource.handle


@dataclass(frozen=True)
class ResourceVersion:
    """A logical value in an existing allocation; version zero is imported."""

    resource: VulkanResource
    version: int = 0

    def __post_init__(self):
        if not isinstance(self.resource, VulkanResource):
            raise TypeError("Expected VulkanResource")
        if not isinstance(self.version, int) or self.version < 0:
            raise ValueError("Resource versions must be nonnegative integers")


class VulkanOperation:
    """Recordable passes and submission bookkeeping, independent of scheduling."""

    def __init__(
        self,
        passes,
        *,
        validate=None,
        dependencies=None,
        submitted=None,
        wait_semaphores=(),
        signal_semaphores=(),
        prepare=None,
    ):
        self.prepare = prepare or (lambda context: None)
        self.wait_semaphores = tuple(wait_semaphores)
        self.signal_semaphores = tuple(signal_semaphores)
        self.passes = tuple(passes)
        if not self.passes or not all(isinstance(p, VulkanPass) for p in self.passes):
            raise ValueError("An operation requires VulkanPass records")
        self.validate = validate or (lambda: None)
        self.dependencies = dependencies or (lambda: ())
        self.submitted = submitted or (lambda completion: None)

    def execute(self, runtime, *, after=()):
        graph = VulkanGraph()
        graph.add("operation", self)
        return graph.compile().execute(runtime, after=after)


@dataclass
class _Node:
    name: str
    operation: VulkanOperation
    reads: tuple
    writes: tuple
    after: tuple


class VulkanGraph:
    """Acyclic application graph; only declared nodes execute.

    Read-only external resources default to imported version zero. A sole writer
    defaults to version one, with other readers consuming that version. Multiple
    writers need explicit versions or a complete explicit ordering.
    """

    def __init__(self):
        self._nodes = []

    def add(self, name, operation, *, reads=(), writes=(), after=()):
        if not name or any(n.name == name for n in self._nodes):
            raise ValueError("Graph node names must be unique and nonempty")
        if isinstance(operation, VulkanPass):
            operation = VulkanOperation([operation])
        if not isinstance(operation, VulkanOperation):
            raise TypeError("Expected VulkanOperation or VulkanPass")
        if any(node.operation is operation for node in self._nodes):
            raise ValueError("Create a distinct operation instance for each graph node")
        reads, writes = tuple(reads), tuple(writes)
        for value in (*reads, *writes):
            if not isinstance(value, ResourceVersion):
                raise TypeError("Connections require ResourceVersion")
        self._nodes.append(
            _Node(name, operation, tuple(reads), tuple(writes), tuple(after))
        )
        return self

    def compile(self):
        nodes = tuple(self._nodes)
        names = {n.name: i for i, n in enumerate(nodes)}
        edges = {i: set() for i in range(len(nodes))}
        accesses, versions, all_keys = [], [], set()
        for i, node in enumerate(nodes):
            for dependency in node.after:
                if dependency not in names:
                    raise ValueError(f"Unknown dependency {dependency}")
                edges[names[dependency]].add(i)
            access = {}
            for stage in node.operation.passes:
                for use in stage.uses:
                    if use.access & ~(_READ | _WRITE):
                        raise ValueError(
                            "Unsupported access bits; use conservative MEMORY_READ/WRITE declarations"
                        )
                    key = _key(use.resource)
                    access[key] = access.get(key, 0) | use.access
                    all_keys.add(key)
            explicit = ({}, {})
            for values, target, mask in (
                (node.reads, explicit[0], _READ),
                (node.writes, explicit[1], _WRITE),
            ):
                for value in values:
                    key = _key(value.resource)
                    if key not in access or not access[key] & mask:
                        raise ValueError(
                            f"{node.name}: connection does not match declared resource access"
                        )
                    if key in target:
                        raise ValueError("Duplicate aliased version connection")
                    if mask == _WRITE and value.version == 0:
                        raise ValueError(
                            "Version zero is imported and cannot be written"
                        )
                    target[key] = value.version
            accesses.append(access)
            versions.append(explicit)

        explicit_edges = {i: set(destinations) for i, destinations in edges.items()}

        def reaches(a, b):
            pending, visited = list(explicit_edges[a]), set()
            while pending:
                current = pending.pop()
                if current == b:
                    return True
                if current not in visited:
                    visited.add(current)
                    pending.extend(explicit_edges[current])
            return False

        for key in all_keys:
            writers = [
                i for i, access in enumerate(accesses) if access.get(key, 0) & _WRITE
            ]
            assigned = {}
            if len(writers) > 1 and any(key not in versions[i][1] for i in writers):
                if not all(
                    reaches(a, b) or reaches(b, a)
                    for j, a in enumerate(writers)
                    for b in writers[j + 1 :]
                ):
                    raise ValueError(
                        "Ambiguous writers: declare resource versions or explicit ordering"
                    )
                ordered = sorted(
                    writers, key=lambda a: sum(reaches(b, a) for b in writers)
                )
                assigned = {i: rank + 1 for rank, i in enumerate(ordered)}
            for i in writers:
                value = versions[i][1].get(key, assigned.get(i, 1))
                assigned[i] = value
            by_version = {}
            for i, value in assigned.items():
                if value in by_version:
                    raise ValueError("Multiple writers of the same resource version")
                by_version[value] = i
            if sorted(by_version) != list(range(1, len(by_version) + 1)):
                raise ValueError(
                    "Written versions must be consecutive from imported version zero"
                )
            for value, writer in by_version.items():
                if value > 1:
                    edges[by_version[value - 1]].add(writer)
            for i, access in enumerate(accesses):
                if not access.get(key, 0) & _READ:
                    continue
                version = versions[i][0].get(key)
                if version is None:
                    version = (
                        assigned[i] - 1 if i in assigned else max(by_version, default=0)
                    )
                if version and version not in by_version:
                    raise ValueError(
                        "Read references a resource version without a producer"
                    )
                producer = by_version.get(version)
                if producer is not None and producer != i:
                    edges[producer].add(i)
                following = by_version.get(version + 1)
                if following is not None and following != i:
                    edges[i].add(following)
        indegree = [0] * len(nodes)
        for destinations in edges.values():
            for destination in destinations:
                indegree[destination] += 1
        ready = [i for i, degree in enumerate(indegree) if degree == 0]
        heapq.heapify(ready)
        order = []
        while ready:
            current = heapq.heappop(ready)
            order.append(current)
            for destination in sorted(edges[current]):
                indegree[destination] -= 1
                if indegree[destination] == 0:
                    heapq.heappush(ready, destination)
        if len(order) != len(nodes):
            raise ValueError(
                "Graph contains a dependency cycle or conflicting alias versions"
            )
        return CompiledVulkanGraph(tuple(nodes[i] for i in order))


class CompiledVulkanGraph:
    def __init__(self, nodes):
        self.nodes = nodes
        self._generations = {
            use.resource.owner: use.resource.owner.binding_revision
            for node in nodes
            for stage in node.operation.passes
            for use in stage.uses
            if hasattr(use.resource.owner, "binding_revision")
        }
        self.order = tuple(n.name for n in nodes)

    def execute(self, runtime, *, after=()):
        with runtime.lock:
            for owner, revision in self._generations.items():
                if owner.binding_revision != revision:
                    raise ValueError("Resource bindings changed; recompile the graph")
            dependencies = list(after)
            passes, waits, signals = [], [], []
            context = {}
            for node in self.nodes:
                node.operation.validate()
                node.operation.prepare(context)
                dependencies.extend(node.operation.dependencies())
                waits.extend(node.operation.wait_semaphores)
                signals.extend(node.operation.signal_semaphores)
                passes.extend(
                    replace(p, name=f"{node.name}/{i}/{p.name}")
                    for i, p in enumerate(node.operation.passes)
                )
            completion = VulkanPassPipeline(passes).execute(
                runtime,
                after=tuple(dict.fromkeys(dependencies)),
                wait_semaphores=waits,
                signal_semaphores=signals,
            )
            for node in self.nodes:
                node.operation.submitted(completion)
            return completion


def reflected_operation(kernel, reflection, *, workgroups, push_constants=b""):
    """Use OrdinaryShade's declared whole-resource accesses conservatively.

    Bindings map reflection to actual allocations, so graph alias checks operate
    on native resource identity. Atomics are conservatively read/write.
    """
    bindings = {}
    for item in reflection.resources:
        if item.kind == "push_constants":
            continue
        if item.set != 0 or item.binding not in kernel.bindings:
            raise ValueError("Reflection must match kernel set-0 bindings")
        resource = kernel.bindings[item.binding]
        kinds = {
            "storage_buffer": "buffer",
            "storage_image": "image",
            "acceleration_structure": "acceleration_structure",
        }
        if item.kind not in kinds:
            raise ValueError(
                "This kernel adapter supports storage buffers/images, acceleration structures and push constants"
            )
        expected = kinds[item.kind]
        if resource.kind != expected:
            raise ValueError("Reflected resource kind disagrees with allocation")
        access = {
            "read": vk.VK_ACCESS_SHADER_READ_BIT,
            "write": vk.VK_ACCESS_SHADER_WRITE_BIT,
            "read_write": vk.VK_ACCESS_SHADER_READ_BIT | vk.VK_ACCESS_SHADER_WRITE_BIT,
        }.get(item.access)
        if access is None:
            raise ValueError("Unknown reflected access mode")
        key = _key(resource)
        previous = bindings.get(key)
        if previous:
            access |= previous.access
        bindings[key] = VulkanResourceUse(
            resource,
            vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            access,
            vk.VK_IMAGE_LAYOUT_GENERAL if resource.kind == "image" else None,
        )
    if {
        item.binding for item in reflection.resources if item.kind != "push_constants"
    } != set(kernel.bindings):
        raise ValueError("Reflection must describe every kernel binding")

    def record(command):
        kernel.bind(
            command, push_constants() if callable(push_constants) else push_constants
        )

    return VulkanOperation(
        [VulkanPass("reflected_compute", tuple(bindings.values()), record, workgroups)],
        validate=kernel.require_open,
    )
