"""Apply prepared scene identity/base roughness to captured primary paths."""

from importlib.resources import files
from operator import index
import struct
import vulkan as vk
from .kernel import VulkanKernel, compile_compute
from ..pipeline.graph import VulkanOperation
from ..pipeline.vulkan import VulkanPass, VulkanResource, VulkanResourceUse

_SOURCE = files("ordinarylight.shaders").joinpath("primary_metadata.comp").read_text()


def prepare_primary_metadata_shader():
    """Compile the metadata stage without creating a GPU runtime."""
    return compile_compute(_SOURCE)


class VulkanPrimaryMetadata:
    """Bind a prepared metadata buffer; run after tracing, before ReLAX prepare.

    Uses captured primary primitive IDs. Supply unique path pixel indices and a
    table matching the resident scene's triangle ordering. Base roughness only.
    """

    def __init__(
        self,
        runtime,
        *,
        paths,
        secondary_paths,
        metadata,
        material,
        capacity,
        triangle_count,
        spirv,
    ):
        self.runtime, self.closed, self.completion = runtime, False, None
        self.capacity, self.triangle_count = index(capacity), index(triangle_count)
        if (
            not 0 < self.capacity <= 0xFFFFFFFF
            or not 0 < self.triangle_count <= 0xFFFFFFFF
        ):
            raise ValueError("Capacity and triangle count must fit positive uint32")
        if material.format != vk.VK_FORMAT_R32_UINT:
            raise ValueError("Material guide requires R32_UINT")
        for buffer, size in (
            (paths, self.capacity * 48),
            (secondary_paths, self.capacity * 128),
            (metadata, self.triangle_count * 16),
        ):
            if buffer.byte_size < size:
                raise ValueError("Metadata stage buffer is too small")
        if len({paths.buffer, secondary_paths.buffer, metadata.buffer}) != 3:
            raise ValueError("Metadata buffers must not alias")
        self.kernel = VulkanKernel(
            runtime,
            spirv,
            {
                0: VulkanResource.buffer(paths),
                1: VulkanResource.buffer(secondary_paths),
                2: VulkanResource.buffer(metadata),
                3: VulkanResource.image(material),
            },
            push_constant_size=8,
        )

    def require_open(self):
        if self.closed:
            raise RuntimeError("Primary metadata stage is closed")
        self.kernel.require_open()

    def operation(self, *, path_count, after=()):
        self.require_open()
        path_count = index(path_count)
        if not 0 <= path_count <= self.capacity:
            raise ValueError("Path count exceeds capacity")
        constants = struct.pack("2I", path_count, self.triangle_count)

        def record(command):
            self.kernel.bind(command, constants)
            vk.vkCmdDispatch(command, (path_count + 63) // 64, 1, 1)

        uses = tuple(
            VulkanResourceUse(
                r,
                vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                (vk.VK_ACCESS_SHADER_READ_BIT | vk.VK_ACCESS_SHADER_WRITE_BIT)
                if b == 1
                else vk.VK_ACCESS_SHADER_WRITE_BIT
                if b == 3
                else vk.VK_ACCESS_SHADER_READ_BIT,
                vk.VK_IMAGE_LAYOUT_GENERAL if b == 3 else None,
            )
            for b, r in self.kernel.bindings.items()
        )
        after = tuple(after)
        return VulkanOperation(
            [VulkanPass("primary_metadata", uses, record)],
            validate=self.require_open,
            dependencies=lambda: after
            + ((self.completion,) if self.completion else ()),
            submitted=lambda completion: setattr(self, "completion", completion),
        )

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
