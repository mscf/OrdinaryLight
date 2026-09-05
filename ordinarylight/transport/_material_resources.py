"""Bind reusable material graph declarations to explicit runtime allocations."""


def prepare_material_resources(scene, supplied, first_binding):
    from ..runtime import VulkanBuffer, VulkanImage, VulkanSampler
    from ..pipeline.vulkan import VulkanResource

    declarations = {}
    for material in scene.materials:
        for resource in getattr(material.program, "resources", ()):
            previous = declarations.setdefault(resource.name, resource)
            if previous != resource:
                raise ValueError("Conflicting material resource declarations")
    supplied = dict(supplied or {})
    if supplied.keys() != declarations.keys():
        raise ValueError("Material resources must exactly match graph declarations")
    bindings, sources, owners = {}, [], []
    binding = first_binding
    for name, declaration in sorted(declarations.items()):
        allocation = supplied[name]
        prefix = f"ol_graph_{name}"
        if declaration.kind in {"buffer", "uniform"}:
            if not isinstance(allocation, VulkanBuffer) or allocation.byte_size % 16:
                raise ValueError(
                    "Material uniform/buffer resources require vec4-aligned buffers"
                )
            if declaration.kind == "uniform":
                if allocation.byte_size != 16:
                    raise ValueError(
                        "Material uniform resources contain exactly one vec4"
                    )
                bindings[binding] = VulkanResource.uniform_buffer(allocation)
                sources.append(
                    f"layout(set=0,binding={binding},std140) uniform {prefix}_block {{ vec4 value; }} {prefix}_data;\nvec4 {prefix}() {{ return {prefix}_data.value; }}\n"
                )
            else:
                bindings[binding] = VulkanResource.buffer(allocation)
                sources.append(
                    f"layout(set=0,binding={binding},std430) readonly buffer {prefix}_block {{ vec4 values[]; }} {prefix}_data;\nvec4 {prefix}(float index) {{ if(isnan(index)||isinf(index)||index<0.0||index>=float({prefix}_data.values.length())) return vec4(0); return {prefix}_data.values[uint(index)]; }}\n"
                )
            owners.append(allocation)
            binding += 1
        else:
            if not isinstance(allocation, (tuple, list)) or len(allocation) != 2:
                raise ValueError("Material textures require an (image, sampler) pair")
            image, sampler = allocation
            if not isinstance(image, VulkanImage) or not isinstance(
                sampler, VulkanSampler
            ):
                raise TypeError(
                    "Material textures require runtime image/sampler allocations"
                )
            bindings[binding] = VulkanResource.sampled_image(image)
            bindings[binding + 1] = VulkanResource.sampler(sampler)
            sources.append(
                f"layout(set=0,binding={binding}) uniform texture2D {prefix}_image;\nlayout(set=0,binding={binding + 1}) uniform sampler {prefix}_sampler;\nvec4 {prefix}(vec2 uv) {{ return textureLod(sampler2D({prefix}_image,{prefix}_sampler),uv,0.0); }}\n"
            )
            owners.extend((image, sampler))
            binding += 2
    for owner in owners:
        if owner.runtime is not scene.runtime:
            raise ValueError("Material resources must share the scene runtime")
        owner.require_open()
    return bindings, "".join(sources), tuple(dict.fromkeys(owners))
