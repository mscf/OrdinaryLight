"""Bind declared custom resources consistently in diagnostics and transport."""


def prepare_resources(scene, supplied):
    import vulkan as vk
    from ..runtime.resources import VulkanBuffer, VulkanImage
    from ..pipeline.vulkan import VulkanResource

    declarations = {}
    for program in scene.programs.values():
        for declaration in program.resources:
            previous = declarations.setdefault(declaration.name, declaration)
            if previous != declaration:
                raise ValueError(
                    "Conflicting declarations for a shared custom resource"
                )
    supplied = dict(supplied or {})
    if declarations.keys() != supplied.keys():
        raise ValueError("Custom resources must exactly match program declarations")
    bindings, source, owners = {}, [], []
    for binding, (name, declaration) in enumerate(sorted(declarations.items()), 16):
        allocation = supplied[name]
        if not isinstance(allocation, (VulkanBuffer, VulkanImage)):
            raise TypeError("Custom resources require runtime buffer/image allocations")
        if allocation.runtime is not scene.runtime:
            raise ValueError("Custom resources must belong to the scene runtime")
        allocation.require_open()
        if declaration.kind == "buffer":
            if (
                not isinstance(allocation, VulkanBuffer)
                or not allocation.usage & vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
            ):
                raise ValueError("Expected a storage buffer")
            if allocation.byte_size % declaration.stride:
                raise ValueError(
                    "Buffer size must be a multiple of its declared std430 stride"
                )
            resource = VulkanResource.buffer(allocation)
        else:
            if (
                not isinstance(allocation, VulkanImage)
                or allocation.format != vk.VK_FORMAT_R32G32B32A32_SFLOAT
                or not allocation.usage & vk.VK_IMAGE_USAGE_STORAGE_BIT
            ):
                raise ValueError("Expected a rgba32f storage image")
            resource = VulkanResource.image(allocation)
        bindings[binding] = resource
        source.append(declaration.declaration(binding))
        owners.append(allocation)
    return bindings, "".join(source), tuple(dict.fromkeys(owners))


def resource_uses(bindings, *, writable=()):
    """Combine aliases into a single pass use, with explicit image layouts."""
    import vulkan as vk
    from ..pipeline.vulkan import VulkanResourceUse

    combined = {}
    for binding, resource in bindings.items():
        key = (resource.kind, resource.handle)
        access = vk.VK_ACCESS_SHADER_READ_BIT
        if binding in writable:
            access |= vk.VK_ACCESS_SHADER_WRITE_BIT
        previous = combined.get(key)
        if previous is not None:
            access |= previous.access
        combined[key] = VulkanResourceUse(
            resource,
            vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            access,
            vk.VK_IMAGE_LAYOUT_GENERAL if resource.kind == "image" else None,
        )
    return tuple(combined.values())
