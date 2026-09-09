"""Native-owned bindings for the shared signal preparation operation."""

from collections import OrderedDict

import vulkan as vk
from .denoiser_graph import _Binding
from ...pipeline.graph import VulkanGraph
from ...pipeline.vulkan import VulkanResource


class _NativePrepareKernel:
    def __init__(self, executor, slot):
        self.executor, self.slot = executor, slot
        self.runtime = executor.core.runtime
        frame = executor.core.window_frames[slot]
        self.bindings = {}
        for binding, value in (
            (0, executor.path_buffer),
            (1, executor.secondary_path_buffer),
            (9, executor.camera_buffers[slot]),
            (10, executor.camera_buffers[1 - slot]),
            (11, executor.core.scene_previous_vertex_buffer),
        ):
            self.bindings[binding] = VulkanResource.buffer(
                _Binding(
                    runtime=self.runtime,
                    buffer=value.buffer,
                    byte_size=value.size,
                    usage=vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                    require_open=self.runtime.require_open,
                )
            )
        rgba, scalar, uint = (
            vk.VK_FORMAT_R16G16B16A16_SFLOAT,
            vk.VK_FORMAT_R32_SFLOAT,
            vk.VK_FORMAT_R32_UINT,
        )
        width, height = frame["wavefront_allocation_extent"]
        for binding, name, format in (
            (2, "wavefront_normal", uint),
            (3, "wavefront_material", uint),
            (4, "wavefront_relax_diffuse", rgba),
            (5, "wavefront_relax_specular", rgba),
            (6, "wavefront_relax_normal_roughness", rgba),
            (7, "wavefront_relax_view_z", scalar),
            (8, "wavefront_relax_motion", rgba),
            (12, "wavefront_relax_identity", uint),
        ):
            self.bindings[binding] = VulkanResource.image(
                _Binding(
                    runtime=self.runtime,
                    image=frame[name + "_image"],
                    view=frame[name + "_view"],
                    width=width,
                    height=height,
                    format=format,
                    layout=vk.VK_IMAGE_LAYOUT_GENERAL,
                    usage=vk.VK_IMAGE_USAGE_STORAGE_BIT,
                    require_open=self.runtime.require_open,
                )
            )

    def require_open(self):
        self.runtime.require_open()


def close_relax_preparation(executor):
    cached = getattr(executor, "relax_preparation_stages", {})
    for _key, stage in cached.values():
        stage.close()
    cached.clear()
    executor.relax_preparation_graphs.clear()


def record_relax_prepare(
    executor, command, slot, path_count, width, height, sample_index=0, sample_count=1
):
    from ...runtime.relax_prepare import VulkanRelaxPrepare

    bindings = _NativePrepareKernel(executor, slot).bindings
    key = tuple(
        (b, r.handle, r.size, getattr(r.owner, "view", None))
        for b, r in sorted(bindings.items())
    )
    cached = executor.relax_preparation_stages
    old = cached.get(slot)
    if old is None or old[0] != key:
        if old is not None:
            old[1].close()
        executor.relax_preparation_graphs.pop(slot, None)
        names = {
            0: "paths",
            1: "secondary_paths",
            2: "packed_normal",
            3: "packed_material",
            4: "diffuse",
            5: "specular",
            6: "normal_roughness",
            7: "view_z",
            8: "motion",
            9: "current_camera",
            10: "previous_camera",
            11: "previous_vertices",
            12: "identity",
        }
        stage = VulkanRelaxPrepare(
            executor.core.runtime,
            capacity=executor.capacity,
            **{names[b]: resource.owner for b, resource in bindings.items()},
        )
        cached[slot] = (key, stage)
    else:
        stage = old[1]
    config = executor.core.config
    graphs = executor.relax_preparation_graphs.setdefault(slot, OrderedDict())
    dispatch_key = (
        width,
        height,
        path_count,
        sample_index,
        sample_count,
        config.denoiser_sampled_indirect,
        config.denoiser_transmission_motion_cap,
        config.denoiser_planar_mirror_guides,
    )
    compiled = graphs.get(dispatch_key)
    if compiled is None:
        operation = stage.operation(
            extent=(width, height),
            path_count=path_count,
            sample_index=sample_index,
            sample_count=sample_count,
            sampled_indirect=config.denoiser_sampled_indirect,
            transmission_motion_cap=config.denoiser_transmission_motion_cap,
            planar_mirror_guides=config.denoiser_planar_mirror_guides,
        )
        compiled = VulkanGraph().add("prepare", operation).compile()
        graphs[dispatch_key] = compiled
        # Bound variants from interactive extent/sample changes per frame slot.
        if len(graphs) > 32:
            graphs.popitem(last=False)
    graphs.move_to_end(dispatch_key)
    # Fresh single-use recording validates owners and prepares current barriers.
    # Parent owns submission/lifetimes; borrowed images remain GENERAL.
    compiled.prepare_recording(stage.runtime).record(command)
