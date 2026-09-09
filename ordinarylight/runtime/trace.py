"""Explicit scheduling for the ordinary split wavefront path."""

from operator import index
import vulkan as vk
from ..pipeline.graph import VulkanGraph, VulkanOperation
from ..pipeline.vulkan import VulkanPass, VulkanResource, VulkanResourceUse
from .shading import shade_operation


def reset_ray_queue(queue, *, capacity):
    """Reset a 48-byte-record queue header; preserve its record storage.

    The caller retains the allocation until submitted work completes.
    """
    capacity = index(capacity)
    if not 0 < capacity <= min(0xFFFFFFFF, (queue.byte_size - 16) // 48):
        raise ValueError("Ray capacity exceeds queue storage")
    if not queue.usage & vk.VK_BUFFER_USAGE_TRANSFER_DST_BIT:
        raise ValueError("Queue reset requires transfer destination usage")
    queue.require_open()
    resource = VulkanResource.buffer(queue)

    def record(command):
        for offset, value in ((0, 0), (4, capacity), (8, 0), (12, 0)):
            vk.vkCmdFillBuffer(command, resource.handle, offset, 4, value)

    return VulkanOperation(
        [
            VulkanPass(
                "reset",
                (
                    VulkanResourceUse(
                        resource,
                        vk.VK_PIPELINE_STAGE_TRANSFER_BIT,
                        vk.VK_ACCESS_TRANSFER_WRITE_BIT,
                    ),
                ),
                record,
            )
        ],
        validate=queue.require_open,
    )


def split_trace_graph(
    *,
    generate,
    queues,
    dispatches,
    intersections,
    kernels,
    arguments,
    settings,
    capacity,
    primary_guides=None,
):
    """Compose a fixed number of split bounces using two alternating queues.

    Stages and allocations are caller-owned. Supply two dispatch/intersection/
    shading stages bound to queue 0 and queue 1 respectively; shading writes the
    opposite queue. Optional primary_guides is a caller-owned VulkanOperation
    recorded after first intersection and before shading/recycling its hit queue.
    No CPU count readback or early termination is required: empty
    queues generate zero indirect work. Output remains in the shared path buffer.
    """
    queues, dispatches, intersections, kernels = map(
        tuple, (queues, dispatches, intersections, kernels)
    )
    if any(len(items) != 2 for items in (queues, dispatches, intersections, kernels)):
        raise ValueError(
            "Split tracing requires two queues and two stages of each kind"
        )
    if settings.fused_intersection:
        raise ValueError("Split tracing requires separate intersection")
    bounces = index(settings.max_bounces)
    if bounces < 1:
        raise ValueError("Split tracing requires positive max_bounces")
    if primary_guides is not None and not isinstance(primary_guides, VulkanOperation):
        raise TypeError("Primary guides must be a VulkanOperation")
    runtime = kernels[0].runtime
    if queues[0].buffer == queues[1].buffer:
        raise ValueError("Split ray queues must not alias")
    for i in range(2):
        expected = queues[i].buffer
        if (
            dispatches[i].kernel.bindings[0].handle != expected
            or dispatches[i].kernel.bindings[1].handle != arguments.buffer
            or intersections[i].bindings[1].handle != expected
            or kernels[i].bindings[1].handle != expected
            or kernels[i].bindings[6].handle != queues[1 - i].buffer
            or kernels[i].bindings[0].handle != intersections[i].bindings[2].handle
        ):
            raise ValueError("Split stage bindings do not match queue ordering")
        if any(
            stage.runtime is not runtime
            for stage in (dispatches[i], intersections[i], kernels[i])
        ):
            raise ValueError("Split stages must share a runtime")
        if intersections[i].capacity != capacity:
            raise ValueError("Split stage capacity mismatch")
    for binding in (2, 7, 15):
        left, right = kernels[0].bindings[binding], kernels[1].bindings[binding]
        if (left.handle, left.offset, left.size) != (
            right.handle,
            right.offset,
            right.size,
        ):
            raise ValueError(
                "Split shading must share path, medium and secondary state"
            )
    graph = VulkanGraph()
    versions = {}
    previous = None

    def append(name, operation):
        nonlocal previous
        # These split stages use whole buffer bindings. Explicit versions prevent
        # a read in an early bounce from selecting the final recycled value.
        uses = {}
        for stage in operation.passes:
            for use in stage.uses:
                if use.resource.kind == "buffer":
                    key = (use.resource.handle, use.resource.offset, use.resource.size)
                    resource, access = uses.get(key, (use.resource, 0))
                    uses[key] = (resource, access | use.access)
        reads, writes = [], []
        write_mask = vk.VK_ACCESS_SHADER_WRITE_BIT | vk.VK_ACCESS_TRANSFER_WRITE_BIT
        for key, (resource, access) in uses.items():
            version = versions.get(key, 0)
            if access & ~write_mask:
                reads.append(resource.version(version))
            if access & write_mask:
                versions[key] = version + 1
                writes.append(resource.version(version + 1))
        graph.add(
            name,
            operation,
            reads=reads,
            writes=writes,
            after=() if previous is None else (previous,),
        )
        previous = name

    append("generate", generate)
    for bounce in range(bounces):
        i = bounce % 2
        for suffix, operation in (
            ("reset", reset_ray_queue(queues[1 - i], capacity=capacity)),
            ("dispatch", dispatches[i].operation()),
            ("intersect", intersections[i].operation(indirect=arguments)),
            ("shade", shade_operation(kernels[i], settings, indirect=arguments)),
        ):
            name = f"bounce.{bounce}.{suffix}"
            append(name, operation)
            if bounce == 0 and suffix == "intersect" and primary_guides is not None:
                append("primary_guides", primary_guides)
    return graph
