"""Native compute-shading bindings using immutable reusable descriptors."""

import vulkan as vk
from ...runtime.kernel import VulkanDescriptorSet
from ...pipeline.vulkan import VulkanResource
from .denoiser_graph import _Binding
from .scene import MAX_NATIVE_TEXTURES, MAX_NATIVE_VOLUMES


def create_shade_descriptors(executor, buffers):
    """Borrow backend-owned resources; parent synchronizes replacement/teardown."""
    core, runtime = executor.core, executor.core.runtime

    def buffer(value):
        owner = _Binding(
            runtime=runtime,
            buffer=value.buffer,
            byte_size=value.size,
            usage=vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            require_open=runtime.require_open,
        )
        return VulkanResource.buffer(owner)

    resources = {binding: buffer(value) for binding, value in buffers}
    # The fixed native layout reserves this slot even without custom attributes.
    resources.setdefault(16, buffer(core.scene_attribute_buffer))
    owner = _Binding(runtime=runtime, require_open=runtime.require_open)
    resources[8] = VulkanResource(
        owner, "acceleration_structure", core.scene_tlas.handle
    )
    arrays = {}
    for binding, kind, count in (
        (13, "textures", MAX_NATIVE_TEXTURES * 2),
        (21, "volumes", MAX_NATIVE_VOLUMES),
    ):
        if kind == "textures" and not core.native_textures_enabled:
            continue
        pairs = []
        for image, sampler in core.scene_resources.sampled_resources(kind, count=count):
            # Native renderer already owns the scene lease. Borrow its views
            # without adding external consumers that would block scene teardown.
            borrowed = _Binding(
                runtime=runtime,
                image=image.handle,
                view=image.owner.view,
                usage=vk.VK_IMAGE_USAGE_SAMPLED_BIT,
                layout=vk.VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                require_open=runtime.require_open,
            )
            pairs.append(
                (
                    VulkanResource.sampled_image(borrowed),
                    VulkanResource(borrowed, "sampler", sampler.handle),
                )
            )
        arrays[binding] = pairs
    return VulkanDescriptorSet(
        runtime,
        resources,
        sampled_image_arrays=arrays,
        sampled_image_layouts={
            binding: vk.VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL for binding in arrays
        },
    )


class _NativeShadeKernel:
    def __init__(self, executor, pipeline, descriptor):
        self.executor, self.pipeline, self.descriptor = executor, pipeline, descriptor
        self.runtime = executor.core.runtime
        self.material_resources = executor.core.material_resources
        self.bindings = descriptor.bindings
        self.image_arrays = descriptor.image_arrays
        self.sampled_image_arrays = descriptor.sampled_image_arrays
        self.sampled_image_layouts = descriptor.sampled_image_layouts

    def require_open(self):
        self.descriptor.require_open()
        if self.material_resources is not None:
            self.material_resources.require_open()

    def bind(self, command, constants):
        self.require_open()
        layout = self.executor.shade_pipeline_layout
        vk.vkCmdBindPipeline(command, vk.VK_PIPELINE_BIND_POINT_COMPUTE, self.pipeline)
        self.descriptor.bind(command, layout)
        if self.material_resources is not None:
            self.material_resources.bind_graph(command, layout)
        raw = vk.ffi.new("uint8_t[]", constants)
        vk.vkCmdPushConstants(
            command, layout, vk.VK_SHADER_STAGE_COMPUTE_BIT, 0, len(constants), raw
        )


def record_shading(executor, command, pipeline, slot, constants):
    from ...runtime.shading import shade_operation
    from ...pipeline.graph import VulkanGraph

    kernel = _NativeShadeKernel(executor, pipeline, executor.shade_descriptors[slot])
    args = executor.indirect_buffer
    indirect = _Binding(
        runtime=kernel.runtime,
        buffer=args.buffer,
        byte_size=args.size,
        usage=vk.VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT,
        require_open=kernel.runtime.require_open,
    )
    settings = _Binding(pack=lambda: bytes(constants))
    # The native parent owns queue serialization, bindings and completion. Image
    # layouts remain read-only, so no deferred layout publication is needed.
    VulkanGraph().add(
        "shade", shade_operation(kernel, settings, indirect=indirect)
    ).compile().prepare_recording(kernel.runtime).record(command)
