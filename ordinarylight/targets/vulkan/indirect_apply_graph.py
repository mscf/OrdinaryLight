"""Native allocation adapter for reusable indirect output kernels."""

from importlib.resources import files
import vulkan as vk
from .denoiser_graph import _Binding
from ...runtime.kernel import VulkanKernel
from ...runtime.indirect_apply import indirect_apply_operation
from ...pipeline.graph import VulkanGraph
from ...pipeline.vulkan import VulkanResource


def close_indirect_output(executor):
    for _key, kernel in executor.indirect_output_stages.values():
        kernel.close()
    executor.indirect_output_stages.clear()


def record_indirect_output(executor, command, slot, width, height, rw, rh):
    runtime = executor.core.runtime
    frame = executor.core.window_frames[slot]
    bindings = {}
    for b, name in (
        (0, "wavefront_indirect_reservoir_buffer"),
        (2, "wavefront_indirect_seed_buffer"),
    ):
        value = frame[name]
        bindings[b] = VulkanResource.buffer(
            _Binding(
                runtime=runtime,
                buffer=value.buffer,
                byte_size=value.size,
                usage=vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                require_open=runtime.require_open,
            )
        )
    for b, name, fmt in (
        (1, "hdr", vk.VK_FORMAT_R16G16B16A16_SFLOAT),
        (3, "material", vk.VK_FORMAT_R32_UINT),
    ):
        w, h = frame["wavefront_allocation_extent"]
        bindings[b] = VulkanResource.image(
            _Binding(
                runtime=runtime,
                image=frame[f"wavefront_{name}_image"],
                view=frame[f"wavefront_{name}_view"],
                width=w,
                height=h,
                format=fmt,
                layout=vk.VK_IMAGE_LAYOUT_GENERAL,
                usage=vk.VK_IMAGE_USAGE_STORAGE_BIT,
                require_open=runtime.require_open,
            )
        )
    key = tuple(
        (b, r.handle, r.size, getattr(r.owner, "view", None))
        for b, r in sorted(bindings.items())
    )
    old = executor.indirect_output_stages.get(slot)
    if old is None or old[0] != key:
        if old is not None:
            old[1].close()
        kernel = VulkanKernel(
            runtime,
            files("ordinarylight.shaders")
            .joinpath("wavefront_indirect_debug.comp.spv")
            .read_bytes(),
            bindings,
            push_constant_size=28,
        )
        executor.indirect_output_stages[slot] = (key, kernel)
    else:
        kernel = old[1]
    config = executor.core.config
    mode = config.wavefront_indirect_reuse_debug_view
    operation = indirect_apply_operation(
        kernel,
        output_extent=(width, height),
        reservoir_extent=(rw, rh),
        mode="apply" if mode == "off" else mode,
        history_limit=config.wavefront_indirect_reuse_history_limit,
        strength=config.wavefront_indirect_reuse_apply_strength,
    )
    VulkanGraph().add("output", operation).compile().prepare_recording(runtime).record(
        command
    )
