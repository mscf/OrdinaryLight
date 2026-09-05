"""Bind reusable material graph declarations to explicit runtime allocations."""


def prepare_material_resources(
    runtime, programs, supplied, first_binding=0, *, descriptor_set=0
):
    from ..runtime import VulkanBuffer, VulkanImage, VulkanSampler
    from ..pipeline.vulkan import VulkanResource

    declarations = {}
    for program in programs:
        for resource in getattr(program, "resources", ()):
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
                    f"layout(set={descriptor_set},binding={binding},std140) uniform {prefix}_block {{ vec4 value; }} {prefix}_data;\nvec4 {prefix}() {{ return {prefix}_data.value; }}\n"
                )
            else:
                bindings[binding] = VulkanResource.buffer(allocation)
                sources.append(
                    f"layout(set={descriptor_set},binding={binding},std430) readonly buffer {prefix}_block {{ vec4 values[]; }} {prefix}_data;\nvec4 {prefix}(float index) {{ if(isnan(index)||isinf(index)||index<0.0||index>=float({prefix}_data.values.length())) return vec4(0); return {prefix}_data.values[uint(index)]; }}\n"
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
                f"layout(set={descriptor_set},binding={binding}) uniform texture2D {prefix}_image;\nlayout(set={descriptor_set},binding={binding + 1}) uniform sampler {prefix}_sampler;\nvec4 {prefix}(vec2 uv) {{ return textureLod(sampler2D({prefix}_image,{prefix}_sampler),uv,0.0); }}\n"
            )
            owners.extend((image, sampler))
            binding += 2
    for owner in owners:
        if owner.runtime is not runtime:
            raise ValueError("Material resources must share the scene runtime")
        owner.require_open()
    return bindings, "".join(sources), tuple(dict.fromkeys(owners))


class VulkanMaterialResources:
    """Persistent camera-GI graph bindings at descriptor set 1.

    Borrowed allocations and this bundle must outlive all attached renderers.
    Complete producers with synchronize(after=...) before rendering; reset
    progressive history whenever resource contents change.
    """

    def __init__(self, runtime, programs, resources):
        from ..runtime.kernel import VulkanDescriptorSet
        from ._core import MaterialProgram

        self.runtime = runtime
        self.programs = tuple(programs)
        if any(not isinstance(program, MaterialProgram) for program in self.programs):
            raise TypeError("programs must contain compiled MaterialProgram objects")
        self.declarations = tuple(
            sorted(
                {
                    resource
                    for program in self.programs
                    for resource in program.resources
                },
                key=lambda resource: resource.name,
            )
        )
        self._borrowers = set()
        self._descriptors = None
        self.closed = False
        with runtime.lock:
            bindings, self.source, _ = prepare_material_resources(
                runtime,
                self.programs,
                resources,
                descriptor_set=1,
            )
            self._descriptors = VulkanDescriptorSet(runtime, bindings)
            try:
                self.synchronize()
            except Exception:
                self.close()
                raise

    @property
    def layout(self):
        self.require_open()
        return self._descriptors.layout

    def require_open(self):
        if self.closed:
            raise RuntimeError("Material resources are closed")
        self._descriptors.require_open()

    def retain(self, consumer):
        with self.runtime.lock:
            self.require_open()
            self._borrowers.add(consumer)

    def release(self, consumer):
        with self.runtime.lock:
            self._borrowers.discard(consumer)

    def validate(self, programs):
        self.require_open()
        declared = set(self.declarations)
        if any(
            resource not in declared
            for program in programs
            for resource in program.resources
        ):
            raise ValueError(
                "Camera material resources do not cover graph declarations"
            )

    def synchronize(self, *, after=()):
        """Wait for producers and establish shader-read visibility/layouts."""
        from ..transport._custom_resources import resource_uses
        from ..pipeline.vulkan import VulkanPass, VulkanPassPipeline

        self.require_open()
        VulkanPassPipeline(
            [
                VulkanPass(
                    "material-resource-readiness",
                    resource_uses(self._descriptors.bindings),
                    lambda command: None,
                )
            ]
        ).execute(self.runtime, after=after).wait()

    def bind(self, command, pipeline_layout):
        import vulkan as vk

        self.require_open()
        for resource in self._descriptors.bindings.values():
            if (
                resource.kind == "image"
                and resource.owner.layout != vk.VK_IMAGE_LAYOUT_GENERAL
            ):
                raise RuntimeError(
                    "Synchronize material resources after image producers"
                )
        vk.vkCmdPipelineBarrier(
            command,
            vk.VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
            vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0,
            1,
            [
                vk.VkMemoryBarrier(
                    srcAccessMask=vk.VK_ACCESS_MEMORY_WRITE_BIT,
                    dstAccessMask=vk.VK_ACCESS_SHADER_READ_BIT
                    | vk.VK_ACCESS_UNIFORM_READ_BIT,
                )
            ],
            0,
            None,
            0,
            None,
        )
        self._descriptors.bind(command, pipeline_layout, set_index=1)

    def close(self):
        with self.runtime.lock:
            if self.closed:
                return
            if self._borrowers:
                raise RuntimeError("Close attached renderers before material resources")
            if self._descriptors is not None:
                self._descriptors.close()
            self.closed = True

    def __enter__(self):
        self.require_open()
        return self

    def __exit__(self, *_exc):
        self.close()
