"""Borrow native fused pipelines/descriptors through the public primary contract.

The executor owns replacement, serialization and lifetime. No GPU allocations
are made while recording; pipeline statistics and native shader selection survive.
"""

import vulkan as vk
from .denoiser_graph import _Binding
from ...pipeline.vulkan import VulkanResource
from ...pipeline.graph import VulkanGraph
from ...runtime.primary import primary_operation


class NativePrimaryKernel:
    def __init__(self, executor, pipeline, slot):
        self.executor, self.pipeline, self.slot = executor, pipeline, slot
        core = executor.core
        self.runtime = core.runtime
        self.material_resources = core.material_resources
        self.geometry_resources = core.geometry_resources
        self.bindings, self.image_arrays = {}, {}
        self.sampled_image_arrays, self.sampled_image_layouts = {}, {}
        current, previous = core.window_frames[slot], core.window_frames[1 - slot]
        for binding, value in (
            (1, executor.path_buffer),
            (2, core.scene_material_buffer),
            (3, core.scene_vertex_buffer),
            (4, core.scene_attribute_buffer),
            (5, executor.next_ray_buffer),
            (6, executor.medium_buffer),
            (7, executor.camera_buffers[slot]),
            (10, core.scene_light_buffer),
            (11, core.scene_area_light_buffer),
            (12, core.scene_texture_buffer),
            (13, core.scene_texture_binding_buffer),
            (16, current["wavefront_reservoir_buffer"]),
            (17, previous["wavefront_reservoir_buffer"]),
            (18, executor.previous_camera_buffers[slot]),
            (23, executor.secondary_path_buffer),
            (24, core.scene_custom_attribute_buffer or core.scene_attribute_buffer),
            (25, core.scene_volume_header_buffer),
            (26, core.scene_volume_scalar_buffer),
            (27, core.scene_volume_transfer_buffer),
            (28, core.scene_triangle_volume_buffer),
        ):
            self.bindings[binding] = self.buffer(value)
        if core.config.wavefront_profiling:
            self.bindings[15] = self.buffer(executor.work_counter_buffers[slot])
        if core.config.wavefront_primary_hits:
            self.bindings[30] = self.buffer(current["wavefront_primary_hit_buffer"])
        if core.config.geometry_resources is not None and executor._denoiser_signals_active():
            self.bindings[31] = self.buffer(current["wavefront_primary_history_buffer"])
            self.bindings[32] = self.buffer(core.scene_previous_vertex_buffer)
        owner = _Binding(runtime=self.runtime, require_open=self.require_open)
        self.bindings[0] = VulkanResource(
            owner, "acceleration_structure", core.scene_tlas.handle
        )
        for binding, frame, name in (
            (8, current, "position"),
            (9, current, "normal"),
            (19, previous, "position"),
            (20, previous, "normal"),
            (21, current, "material"),
            (22, previous, "material"),
        ):
            self.bindings[binding] = VulkanResource.image(
                _Binding(
                    runtime=self.runtime,
                    require_open=self.require_open,
                    image=frame[f"wavefront_{name}_image"],
                    view=frame[f"wavefront_{name}_view"],
                    layout=vk.VK_IMAGE_LAYOUT_GENERAL,
                    usage=vk.VK_IMAGE_USAGE_STORAGE_BIT,
                )
            )
        for binding, kind, count in ((14, "textures", 128), (29, "volumes", 16)):
            if binding == 14 and not core.native_textures_enabled:
                continue
            self.sampled_image_arrays[binding] = core.scene_resources.sampled_resources(
                kind, count=count
            )
            self.sampled_image_layouts[binding] = (
                vk.VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
            )

    def buffer(self, value):
        return VulkanResource.buffer(
            _Binding(
                runtime=self.runtime,
                require_open=self.require_open,
                buffer=value.buffer,
                byte_size=value.size,
                usage=vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            )
        )

    def require_open(self):
        self.runtime.require_open()

    def bind(self, command, constants):
        executor = self.executor
        vk.vkCmdBindPipeline(command, vk.VK_PIPELINE_BIND_POINT_COMPUTE, self.pipeline)
        vk.vkCmdBindDescriptorSets(
            command,
            vk.VK_PIPELINE_BIND_POINT_COMPUTE,
            executor.primary_pipeline_layout,
            0,
            1,
            [executor.primary_sets[self.slot]],
            0,
            None,
        )
        executor.core._bind_material_resources(
            command, executor.primary_pipeline_layout
        )
        vk.vkCmdPushConstants(
            command,
            executor.primary_pipeline_layout,
            vk.VK_SHADER_STAGE_COMPUTE_BIT,
            0,
            len(constants),
            vk.ffi.from_buffer(constants),
        )


def record_primary(executor, command, pipeline, slot, constants, groups):
    kernel = NativePrimaryKernel(executor, pipeline, slot)
    VulkanGraph().add(
        "primary", primary_operation(kernel, constants, workgroups=groups)
    ).compile().prepare_recording(kernel.runtime).record(command)
