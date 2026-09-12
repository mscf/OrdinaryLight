"""Composable fused primary/hybrid/megakernel compute dispatch."""

from dataclasses import replace
from operator import index
import vulkan as vk
from ..pipeline.graph import VulkanOperation
from ..pipeline.vulkan import VulkanPass, VulkanResourceUse
from ..wavefront.primary_bindings import primary_bindings


def primary_operation(kernel, constants, *, workgroups, after=()):
    """Dispatch a caller-owned kernel with the 176-byte PrimarySettings ABI.

    Shader variant and matching workgroup geometry are selected by the caller.
    Resources, descriptors and pipeline must remain alive through completion.
    This operation covers compute entry points, not ray-tracing/SBT dispatch.
    """
    kernel.require_open()
    constants = bytes(constants)
    groups = tuple(map(index, workgroups))
    if len(constants) != 176:
        raise ValueError("Primary constants must contain 176 bytes")
    if len(groups) != 3 or any(n <= 0 or n > 65535 for n in groups):
        raise ValueError("Invalid primary workgroups")
    contract = primary_bindings(
        native_textures=14 in kernel.sampled_image_arrays,
        profiling=15 in kernel.bindings,
        primary_hits=30 in kernel.bindings,
        custom_history=31 in kernel.bindings,
    )
    scalars = {b.binding: b for b in contract if b.count == 1}
    arrays = {b.binding: b for b in contract if b.count != 1}
    if (
        set(kernel.bindings) != set(scalars)
        or set(kernel.sampled_image_arrays) != set(arrays)
        or kernel.image_arrays
    ):
        raise ValueError("Incomplete or unknown primary bindings")
    uses = {}

    def add(
        resource, access, layout=None, stage=vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT
    ):
        resource.owner.require_open()
        if resource.owner.runtime is not kernel.runtime:
            raise ValueError("Primary resources must share a runtime")
        key = resource.kind, resource.handle
        old = uses.get(key)
        if old:
            if layout != old.layout:
                raise ValueError("Aliased primary images require identical layouts")
            access |= old.access
            stage |= old.stage
            if resource.kind == "buffer":
                lo = min(resource.offset, old.resource.offset)
                hi = max(
                    resource.offset + resource.size,
                    old.resource.offset + old.resource.size,
                )
                resource = replace(resource, offset=lo, size=hi - lo)
        uses[key] = VulkanResourceUse(resource, stage, access, layout)

    for binding, spec in scalars.items():
        resource = kernel.bindings[binding]
        if resource.kind != spec.kind:
            raise ValueError("Primary binding kind mismatch")
        access = vk.VK_ACCESS_SHADER_READ_BIT if "read" in spec.access else 0
        if "write" in spec.access:
            access |= vk.VK_ACCESS_SHADER_WRITE_BIT
        if spec.kind == "acceleration_structure":
            access = vk.VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR
        add(
            resource,
            access,
            vk.VK_IMAGE_LAYOUT_GENERAL if spec.kind == "image" else None,
        )
    for binding, spec in arrays.items():
        pairs = kernel.sampled_image_arrays[binding]
        if len(pairs) != spec.count:
            raise ValueError("Primary sampled array count mismatch")
        for image, sampler in pairs:
            if (
                image.kind != "image"
                or sampler.kind != "sampler"
                or sampler.owner.runtime is not kernel.runtime
            ):
                raise ValueError("Invalid primary sampled resources")
            sampler.owner.require_open()
            add(
                image,
                vk.VK_ACCESS_SHADER_READ_BIT,
                kernel.sampled_image_layouts.get(binding, vk.VK_IMAGE_LAYOUT_GENERAL),
            )
    material = getattr(kernel, "material_resources", None)
    if material is not None:
        for use in material.uses:
            add(use.resource, use.access, use.layout, use.stage)

    geometry = getattr(kernel, "geometry_resources", None)
    if geometry is not None:
        for use in geometry.uses:
            add(use.resource, use.access, use.layout, use.stage)

    def record(command):
        kernel.bind(command, constants)
        vk.vkCmdDispatch(command, *groups)

    after = tuple(after)
    return VulkanOperation(
        [VulkanPass("primary", tuple(uses.values()), record)],
        validate=kernel.require_open,
        dependencies=lambda: after,
    )
