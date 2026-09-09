"""Native-owned descriptor bindings for the shared path resolve operation."""

from collections import OrderedDict
from importlib.resources import files
from ...runtime.kernel import VulkanKernel

import vulkan as vk
from .denoiser_graph import _Binding
from ...pipeline.vulkan import VulkanResource
from ...pipeline.graph import VulkanGraph
from ...runtime.path_resolve import path_resolve_operation


class _NativeResolveKernel:
    def __init__(self, executor, slot):
        self.executor, self.slot = executor, slot
        self.runtime = executor.core.runtime
        frame = executor.core.window_frames[slot]

        def buffer(value):
            return VulkanResource.buffer(
                _Binding(
                    runtime=self.runtime,
                    buffer=value.buffer,
                    byte_size=value.size,
                    usage=vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                    require_open=self.runtime.require_open,
                )
            )

        self.bindings = {
            0: buffer(executor.path_buffer),
            2: buffer(executor.secondary_path_buffer),
            4: buffer(executor.camera_buffers[slot]),
        }
        # Native descriptors use secondary storage as a dormant placeholder when
        # capture is disabled. The shared operation omits those unused bindings.
        for binding, name in (
            (3, "wavefront_indirect_reservoir_buffer"),
            (5, "wavefront_indirect_seed_buffer"),
        ):
            self.bindings[binding] = buffer(
                frame.get(name) or executor.secondary_path_buffer
            )
        width, height = frame["wavefront_allocation_extent"]
        self.bindings[1] = VulkanResource.image(
            _Binding(
                runtime=self.runtime,
                image=frame["wavefront_hdr_image"],
                view=frame["wavefront_hdr_view"],
                width=width,
                height=height,
                format=vk.VK_FORMAT_R16G16B16A16_SFLOAT,
                layout=vk.VK_IMAGE_LAYOUT_GENERAL,
                usage=vk.VK_IMAGE_USAGE_STORAGE_BIT,
                require_open=self.runtime.require_open,
            )
        )

    def require_open(self):
        self.runtime.require_open()


def close_path_resolve(executor):
    for _key, kernel, _graphs in executor.path_resolve_stages.values():
        kernel.close()
    executor.path_resolve_stages.clear()


def record_path_resolve(
    executor, command, slot, path_count, width, height, sample_index=0, sample_count=1
):
    bindings = _NativeResolveKernel(executor, slot).bindings
    key = tuple(
        (b, r.handle, r.size, getattr(r.owner, "view", None))
        for b, r in sorted(bindings.items())
    )
    old = executor.path_resolve_stages.get(slot)
    if old is None or old[0] != key:
        if old is not None:
            old[1].close()
        kernel = VulkanKernel(
            executor.core.runtime,
            files("ordinarylight.shaders")
            .joinpath("wavefront_path_to_hdr.comp.spv")
            .read_bytes(),
            bindings,
            push_constant_size=32,
        )
        graphs = OrderedDict()
        executor.path_resolve_stages[slot] = (key, kernel, graphs)
    else:
        _key, kernel, graphs = old
    config = executor.core.config
    capture = bool(
        config.wavefront_indirect_reuse_candidates
        or executor._denoiser_signals_active()
    )
    extent = executor.core.window_frames[slot].get(
        "wavefront_indirect_reservoir_extent"
    ) or (1, 1)
    dispatch_key = (
        width,
        height,
        path_count,
        sample_index,
        sample_count,
        capture,
        bool(config.denoiser_sampled_indirect),
        tuple(extent),
    )
    compiled = graphs.get(dispatch_key)
    if compiled is None:
        operation = path_resolve_operation(
            kernel,
            capacity=executor.capacity,
            extent=(width, height),
            reservoir_extent=extent,
            path_count=path_count,
            sample_index=sample_index,
            sample_count=sample_count,
            capture_secondary=capture,
            sampled_indirect=bool(config.denoiser_sampled_indirect) and capture,
        )
        compiled = VulkanGraph().add("resolve", operation).compile()
        graphs[dispatch_key] = compiled
        if len(graphs) > 32:
            graphs.popitem(last=False)
    graphs.move_to_end(dispatch_key)
    # Native owner serializes submissions and keeps borrowed images in GENERAL.
    compiled.prepare_recording(kernel.runtime).record(command)
