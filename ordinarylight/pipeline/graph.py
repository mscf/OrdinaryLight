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
        # Split each allocation at every declared endpoint. Versions and hazards
        # apply independently to each overlapping interval, including full views.
        endpoints = {}
        for node in nodes:
            resources = [
                use.resource for stage in node.operation.passes for use in stage.uses
            ]
            resources += [value.resource for value in (*node.reads, *node.writes)]
            for resource in resources:
                if resource.kind == "buffer":
                    endpoints.setdefault(_key(resource), set()).update(
                        (resource.offset, resource.offset + resource.size)
                    )
        segments = {
            key: tuple(zip(points, points[1:]))
            for key, values in endpoints.items()
            for points in [sorted(values)]
        }

        def resource_keys(resource):
            key = _key(resource)
            if resource.kind != "buffer":
                return (key,)
            return tuple(
                (*key, start, end)
                for start, end in segments[key]
                if resource.offset <= start and end <= resource.offset + resource.size
            )

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
                    for key in resource_keys(use.resource):
                        access[key] = access.get(key, 0) | use.access
                        all_keys.add(key)
            explicit = ({}, {})
            for values, target, mask in (
                (node.reads, explicit[0], _READ),
                (node.writes, explicit[1], _WRITE),
            ):
                for value in values:
                    for key in resource_keys(value.resource):
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

    def prepare_recording(self, runtime, *, after=()):
        """Prepare for an externally owned command buffer and queue submission.

        Caller serializes recording/submission on runtime.lock, honors dependencies
        and semaphore requirements, and retains resources through GPU completion.
        Call submitted only after queue submission succeeds. Replaying cached
        commands is allowed only with unchanged bindings and compatible layouts.
        """
        runtime.require_open()
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
        record, owners, commit = VulkanPassPipeline(passes)._prepare_recording(runtime)
        return VulkanGraphRecording(
            record,
            owners,
            commit,
            self.nodes,
            tuple(dict.fromkeys(dependencies)),
            waits,
            signals,
        )

    def execute(self, runtime, *, after=()):
        with runtime.lock:
            recording = self.prepare_recording(runtime, after=after)
            completion = runtime.submit(
                recording.record,
                resources=recording.resources,
                after=recording.dependencies,
                wait_semaphores=recording.wait_semaphores,
                signal_semaphores=recording.signal_semaphores,
            )
            recording.submitted(completion)
            return completion


class VulkanGraphRecording:
    """Prepared recording with explicit publication after external submission.

    A cached command may be submitted again, publishing each completion. Its owner
    must preserve all borrowed bindings and command-embedded policy between uses.
    """

    def __init__(self, record, resources, commit, nodes, dependencies, waits, signals):
        self._record, self._commit, self._nodes = record, commit, nodes
        self.resources = resources
        self.dependencies = dependencies
        self.wait_semaphores, self.signal_semaphores = tuple(waits), tuple(signals)
        self._recorded = False
        self._attempted = False

    def record(self, command):
        if self._attempted:
            raise RuntimeError("Prepare a new recording to record commands again")
        self._attempted = True
        self._record(command)
        self._recorded = True

    def submitted(self, completion):
        if not self._recorded:
            raise RuntimeError("Record commands before publishing submission")
        self._commit()
        for node in self._nodes:
            node.operation.submitted(completion)


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
            "uniform_buffer": "buffer",
            "sampled_texture_2d": "image",
            "sampler": "sampler",
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
        expected_descriptor = (
            {
                "buffer": "storage_buffer",
                "image": "storage_image",
                "acceleration_structure": "acceleration_structure",
                "sampler": "sampler",
            }[resource.kind]
            if resource.descriptor is None
            else resource.descriptor
        )
        if item.kind != expected_descriptor:
            raise ValueError("Reflected descriptor disagrees with kernel binding")
        if item.kind == "sampler":
            continue  # Immutable sampler state has lifetime, but no memory hazards.
        access = {
            "read": vk.VK_ACCESS_SHADER_READ_BIT,
            "write": vk.VK_ACCESS_SHADER_WRITE_BIT,
            "read_write": vk.VK_ACCESS_SHADER_READ_BIT | vk.VK_ACCESS_SHADER_WRITE_BIT,
        }.get(item.access)
        if access is None:
            raise ValueError("Unknown reflected access mode")
        if item.kind == "uniform_buffer":
            access = vk.VK_ACCESS_UNIFORM_READ_BIT
        key = _key(resource)
        previous = bindings.get(key)
        if previous:
            access |= previous.access
            if resource.kind == "buffer":
                start = min(resource.offset, previous.resource.offset)
                end = max(
                    resource.offset + resource.size,
                    previous.resource.offset + previous.resource.size,
                )
                resource = replace(resource, offset=start, size=end - start)
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
