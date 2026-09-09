"""Native descriptor adapter for the shared indirect candidate operation."""

from importlib.resources import files
from ...runtime.kernel import VulkanKernel

import vulkan as vk
from .denoiser_graph import _Binding
from ...pipeline.vulkan import VulkanResource
from ...pipeline.graph import VulkanGraph
from ...runtime.indirect_candidates import indirect_candidates_operation


class _Kernel:
    def __init__(self, executor, slot):
        self.executor, self.slot = executor, slot
        self.runtime = executor.core.runtime
        current = executor.core.window_frames[slot]
        previous = executor.core.window_frames[1 - slot]
        self.bindings = {}
        for b, value in (
            (0, current["wavefront_indirect_reservoir_buffer"]),
            (4, executor.camera_buffers[slot]),
            (5, previous["wavefront_indirect_reservoir_buffer"]),
            (8, executor.camera_buffers[1 - slot]),
            (11, executor.indirect_reuse_counter_buffers[slot]),
        ):
            self.bindings[b] = VulkanResource.buffer(
                _Binding(
                    runtime=self.runtime,
                    buffer=value.buffer,
                    byte_size=value.size,
                    usage=vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                    | vk.VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                    require_open=self.runtime.require_open,
                )
            )
        for b, frame, name, fmt in (
            (1, current, "hdr", vk.VK_FORMAT_R16G16B16A16_SFLOAT),
            (2, current, "position", vk.VK_FORMAT_R32_SFLOAT),
            (3, current, "normal", vk.VK_FORMAT_R32_UINT),
            (6, previous, "position", vk.VK_FORMAT_R32_SFLOAT),
            (7, previous, "normal", vk.VK_FORMAT_R32_UINT),
            (9, current, "material", vk.VK_FORMAT_R32_UINT),
            (10, previous, "material", vk.VK_FORMAT_R32_UINT),
        ):
            width, height = frame["wavefront_allocation_extent"]
            self.bindings[b] = VulkanResource.image(
                _Binding(
                    runtime=self.runtime,
                    image=frame[f"wavefront_{name}_image"],
                    view=frame[f"wavefront_{name}_view"],
                    width=width,
                    height=height,
                    format=fmt,
                    layout=vk.VK_IMAGE_LAYOUT_GENERAL,
                    usage=vk.VK_IMAGE_USAGE_STORAGE_BIT,
                    require_open=self.runtime.require_open,
                )
            )
        owner = _Binding(runtime=self.runtime, require_open=self.runtime.require_open)
        self.bindings[12] = VulkanResource(
            owner, "acceleration_structure", executor.core.scene_tlas.handle
        )

    def require_open(self):
        self.runtime.require_open()


def close_candidates(executor):
    for _key, kernel in executor.candidate_stages.values():
        kernel.close()
    executor.candidate_stages.clear()


def record_candidates(
    executor,
    command,
    slot,
    source_width,
    source_height,
    reservoir_width,
    reservoir_height,
    history_valid,
    frame_index,
    history_limit,
):
    bindings = _Kernel(executor, slot).bindings
    key = tuple(
        (b, r.handle, r.size, getattr(r.owner, "view", None))
        for b, r in sorted(bindings.items())
    )
    old = executor.candidate_stages.get(slot)
    if old is None or old[0] != key:
        if old is not None:
            old[1].close()
        kernel = VulkanKernel(
            executor.core.runtime,
            files("ordinarylight.shaders")
            .joinpath("wavefront_indirect_candidates.comp.spv")
            .read_bytes(),
            bindings,
            push_constant_size=36,
        )
        executor.candidate_stages[slot] = (key, kernel)
    else:
        kernel = old[1]
    operation = indirect_candidates_operation(
        kernel,
        source_extent=(source_width, source_height),
        reservoir_extent=(reservoir_width, reservoir_height),
        history_valid=history_valid,
        frame_index=frame_index,
        history_limit=history_limit,
        spatial=executor.core.config.wavefront_indirect_reuse_spatial,
        profiling=executor.core.config.wavefront_indirect_reuse_profiling,
    )
    VulkanGraph().add("candidates", operation).compile().prepare_recording(
        kernel.runtime
    ).record(command)
