"""Resource-backed dispatch for kernels using the split-shading ABI."""

from dataclasses import replace
from operator import index
import vulkan as vk
from ..wavefront.shading import SHADE_BUFFER_BINDINGS, SHADE_WRITABLE_BUFFERS
from ..pipeline.graph import VulkanOperation
from ..pipeline.vulkan import VulkanPass, VulkanResource, VulkanResourceUse


def shade_operation(kernel, settings, *, indirect=None, ray_capacity=None, after=()):
    """Dispatch a prepared shading kernel with explicit resource hazards.

    The kernel must implement the existing split-shading ABI, including 64-thread
    workgroups. This operation neither compiles material programs nor resets
    queues: callers initialize output queue headers and path/medium state first.
    Exactly one of indirect or ray_capacity selects dispatch sizing. Kernels and
    bindings stay caller-owned through completion. A kernel material_resources
    bundle at set 1 contributes its read dependencies.
    """
    kernel.require_open()
    constants = settings.pack()
    if len(constants) != 56:
        raise ValueError("Shading requires 56-byte constants")
    if (indirect is None) == (ray_capacity is None):
        raise ValueError("Select indirect dispatch or ray capacity")
    required = set(SHADE_BUFFER_BINDINGS.values()) - {14, 16}
    if not (required | {8}) <= kernel.bindings.keys():
        raise ValueError("Incomplete shading resource bindings")
    if kernel.bindings.keys() - (set(SHADE_BUFFER_BINDINGS.values()) | {8}):
        raise ValueError("Unknown shading resource bindings")
    if kernel.image_arrays or kernel.sampled_image_arrays.keys() - {13, 21}:
        raise ValueError("Unsupported shading image bindings")
    writes = {SHADE_BUFFER_BINDINGS[name] for name in SHADE_WRITABLE_BUFFERS}
    uses = {}

    def add(resource, stage, access, layout=None):
        resource.owner.require_open()
        if resource.owner.runtime is not kernel.runtime:
            raise ValueError("Shading resources must share a runtime")
        key = (resource.kind, resource.handle)
        previous = uses.get(key)
        if previous is not None:
            if previous.layout != layout:
                raise ValueError("Aliased shading images must use the same layout")
            stage |= previous.stage
            access |= previous.access
            if resource.kind == "buffer":
                lo = min(resource.offset, previous.resource.offset)
                hi = max(
                    resource.offset + resource.size,
                    previous.resource.offset + previous.resource.size,
                )
                resource = replace(resource, offset=lo, size=hi - lo)
        uses[key] = VulkanResourceUse(resource, stage, access, layout)

    for binding, resource in kernel.bindings.items():
        if resource.kind != ("acceleration_structure" if binding == 8 else "buffer"):
            raise ValueError("Shading binding kind mismatch")
        add(
            resource,
            vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            vk.VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR
            if binding == 8
            else vk.VK_ACCESS_SHADER_READ_BIT
            | (vk.VK_ACCESS_SHADER_WRITE_BIT if binding in writes else 0),
        )
    for binding, pairs in kernel.sampled_image_arrays.items():
        layout = kernel.sampled_image_layouts.get(binding, vk.VK_IMAGE_LAYOUT_GENERAL)
        for image, _sampler in pairs:
            add(
                image,
                vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                vk.VK_ACCESS_SHADER_READ_BIT,
                layout,
            )
    material_resources = getattr(kernel, "material_resources", None)
    if material_resources is not None:
        for use in material_resources.uses:
            add(use.resource, use.stage, use.access, use.layout)
    if indirect is not None:
        if (
            indirect.byte_size < 12
            or not indirect.usage & vk.VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT
        ):
            raise ValueError("Invalid shading indirect buffer")
        if indirect.buffer in {r.handle for r in kernel.bindings.values()}:
            raise ValueError("Indirect arguments must not alias shading buffers")
        add(
            VulkanResource.buffer(indirect),
            vk.VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT,
            vk.VK_ACCESS_INDIRECT_COMMAND_READ_BIT,
        )
    else:
        ray_capacity = index(ray_capacity)
        if not 0 < ray_capacity <= (kernel.bindings[1].size - 16) // 48:
            raise ValueError("Ray capacity exceeds shading input queue")

    def record(command):
        kernel.bind(command, constants)
        if indirect is None:
            vk.vkCmdDispatch(command, (ray_capacity + 63) // 64, 1, 1)
        else:
            vk.vkCmdDispatchIndirect(command, indirect.buffer, 0)

    def validate():
        kernel.require_open()
        if indirect is not None:
            indirect.require_open()

    after = tuple(after)
    return VulkanOperation(
        [VulkanPass("shade", tuple(uses.values()), record)],
        validate=validate,
        dependencies=lambda: after,
    )
