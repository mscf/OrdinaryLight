"""Scatter native path signals and motion guides into ReLAX input images."""

from importlib.resources import files
from operator import index
import struct
import vulkan as vk
from .kernel import VulkanKernel
from ..pipeline.graph import VulkanOperation
from ..pipeline.vulkan import VulkanPass, VulkanResource, VulkanResourceUse


class VulkanRelaxPrepare:
    """Borrow path records, cameras, previous geometry and denoiser images.

    Paths must address unique pixels. Previous vertices must cover every primitive
    referenced by valid secondary records, using the native world-space vec4 ABI.
    Initialize uncovered image pixels before use; the shader preserves them.
    """

    def __init__(
        self,
        runtime,
        *,
        paths,
        secondary_paths,
        current_camera,
        previous_camera,
        previous_vertices,
        packed_normal,
        packed_material,
        diffuse,
        specular,
        normal_roughness,
        view_z,
        motion,
        identity,
        capacity,
    ):
        self.runtime, self.closed, self.completion = runtime, False, None
        self.capacity = index(capacity)
        if not 0 < self.capacity <= 0xFFFFFFFF:
            raise ValueError("Preparation capacity must fit a positive uint32")
        images = (
            packed_normal,
            packed_material,
            diffuse,
            specular,
            normal_roughness,
            view_z,
            motion,
            identity,
        )
        formats = (
            (vk.VK_FORMAT_R32_UINT,) * 2
            + (vk.VK_FORMAT_R16G16B16A16_SFLOAT,) * 3
            + (
                vk.VK_FORMAT_R32_SFLOAT,
                vk.VK_FORMAT_R16G16B16A16_SFLOAT,
                vk.VK_FORMAT_R32_UINT,
            )
        )
        self.extent = (diffuse.width, diffuse.height)
        with runtime.lock:
            runtime.require_open()
            for image, format in zip(images, formats):
                image.require_open()
                if image.format != format or (image.width, image.height) != self.extent:
                    raise ValueError(
                        "Preparation images require matching extents and ABI formats"
                    )
            if len({image.image for image in images}) != len(images):
                raise ValueError("Preparation images must not alias")
            for buffer, size in (
                (paths, self.capacity * 48),
                (secondary_paths, self.capacity * 128),
                (current_camera, 64),
                (previous_camera, 64),
                (previous_vertices, 48),
            ):
                buffer.require_open()
                if buffer.byte_size < size:
                    raise ValueError(
                        "Preparation buffer is smaller than required storage"
                    )
            bindings = {
                0: VulkanResource.buffer(paths),
                1: VulkanResource.buffer(secondary_paths),
                9: VulkanResource.buffer(current_camera),
                10: VulkanResource.buffer(previous_camera),
                11: VulkanResource.buffer(previous_vertices),
            }
            bindings.update(
                {
                    binding: VulkanResource.image(image)
                    for binding, image in zip((2, 3, 4, 5, 6, 7, 8, 12), images)
                }
            )
            self.kernel = VulkanKernel(
                runtime,
                files("ordinarylight.shaders")
                .joinpath("denoiser_relax_prepare.comp.spv")
                .read_bytes(),
                bindings,
                push_constant_size=32,
            )

    def require_open(self):
        if self.closed:
            raise RuntimeError("ReLAX preparation stage is closed")
        self.kernel.require_open()

    def operation(
        self,
        *,
        path_count,
        extent=None,
        sample_index=0,
        sample_count=1,
        sampled_indirect=False,
        transmission_motion_cap=False,
        planar_mirror_guides=False,
        after=(),
    ):
        self.require_open()
        extent = self.extent if extent is None else tuple(map(index, extent))
        if len(extent) != 2 or any(
            n <= 0 or n > limit for n, limit in zip(extent, self.extent)
        ):
            raise ValueError("Active preparation extent exceeds allocation")
        operation = relax_prepare_operation(
            self.kernel,
            capacity=self.capacity,
            extent=extent,
            path_count=path_count,
            sample_index=sample_index,
            sample_count=sample_count,
            sampled_indirect=sampled_indirect,
            transmission_motion_cap=transmission_motion_cap,
            planar_mirror_guides=planar_mirror_guides,
        )
        after = tuple(after)
        operation.validate = self.require_open
        operation.dependencies = lambda: after + (
            (self.completion,) if self.completion else ()
        )
        operation.submitted = lambda completion: setattr(self, "completion", completion)
        return operation

    def close(self):
        with self.runtime.lock:
            if self.closed:
                return
            if self.completion is not None:
                self.completion.wait()
            self.kernel.close()
            self.closed = True

    def __enter__(self):
        self.require_open()
        return self

    def __exit__(self, *_exc):
        self.close()


def relax_prepare_operation(
    kernel,
    *,
    capacity,
    extent,
    path_count,
    sample_index=0,
    sample_count=1,
    sampled_indirect=False,
    transmission_motion_cap=False,
    planar_mirror_guides=False,
):
    """Shared signal preparation recording for standalone and native bindings."""
    kernel.require_open()
    path_count, sample_index, sample_count = map(
        index, (path_count, sample_index, sample_count)
    )
    if not 0 <= path_count <= capacity:
        raise ValueError("Path count exceeds preparation capacity")
    if not 0 <= sample_index < sample_count <= 0xFFFFFFFF:
        raise ValueError("Invalid preparation sample index/count")
    constants = struct.pack(
        "8I",
        *extent,
        path_count,
        int(bool(transmission_motion_cap)),
        sample_index,
        sample_count,
        int(bool(sampled_indirect)),
        int(bool(planar_mirror_guides)),
    )
    uses = {}
    for binding, resource in kernel.bindings.items():
        access = vk.VK_ACCESS_SHADER_READ_BIT
        if binding in (4, 5, 6, 7, 8, 12):
            access = vk.VK_ACCESS_SHADER_WRITE_BIT
            if binding in (4, 5) and sampled_indirect and sample_index:
                access |= vk.VK_ACCESS_SHADER_READ_BIT
        key = (resource.kind, resource.handle)
        previous = uses.get(key)
        if previous is not None:
            access |= previous.access
        uses[key] = VulkanResourceUse(
            resource,
            vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            access,
            vk.VK_IMAGE_LAYOUT_GENERAL if resource.kind == "image" else None,
        )

    def record(command):
        kernel.bind(command, constants)
        vk.vkCmdDispatch(command, (path_count + 63) // 64, 1, 1)

    return VulkanOperation(
        [VulkanPass("prepare_signals", tuple(uses.values()), record)],
        validate=kernel.require_open,
    )
