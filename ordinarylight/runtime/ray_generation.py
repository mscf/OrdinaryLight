"""Scene-independent primary-ray generation using the native wavefront ABI."""

from importlib.resources import files
from operator import index
import struct
import vulkan as vk
from .kernel import VulkanKernel
from ..pipeline.graph import VulkanOperation
from ..pipeline.vulkan import VulkanPass, VulkanResource, VulkanResourceUse


class VulkanRayGeneration:
    """Borrow camera/queue/path/medium buffers and own a primary-ray kernel.

    The operation resets its ray queue header and initializes the active tile.
    Other queues and secondary path storage remain the caller's responsibility.
    """

    def __init__(self, runtime, *, camera, rays, paths, media, capacity):
        from ..wavefront import RAY_DTYPE, HOT_PATH_STATE_DTYPE, MEDIUM_STACK_DTYPE

        self.runtime, self.closed, self.completion = runtime, False, None
        self.capacity = index(capacity)
        if not 0 < self.capacity <= 0xFFFFFFFF:
            raise ValueError("Ray capacity must fit a positive uint32")
        self.buffers = (rays, paths, media, camera)
        sizes = (
            16 + self.capacity * RAY_DTYPE.itemsize,
            self.capacity * HOT_PATH_STATE_DTYPE.itemsize,
            self.capacity * MEDIUM_STACK_DTYPE.itemsize,
            64,
        )
        with runtime.lock:
            runtime.require_open()
            for buffer, size in zip(self.buffers, sizes):
                buffer.require_open()
                if (
                    buffer.runtime is not runtime
                    or buffer.byte_size < size
                    or not buffer.usage & vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                ):
                    raise ValueError(
                        "Ray-generation buffer runtime, size or storage usage mismatch"
                    )
            if not rays.usage & vk.VK_BUFFER_USAGE_TRANSFER_DST_BIT:
                raise ValueError("Ray queue requires transfer destination usage")
            if len({b.buffer for b in self.buffers}) != 4:
                raise ValueError("Ray-generation buffers must not alias")
            self.bindings = {
                i: VulkanResource.buffer(b) for i, b in enumerate(self.buffers)
            }
            self.kernel = VulkanKernel(
                runtime,
                files("ordinarylight.shaders")
                .joinpath("wavefront_generate.comp.spv")
                .read_bytes(),
                self.bindings,
                push_constant_size=32,
            )

    def require_open(self):
        if self.closed:
            raise RuntimeError("Ray-generation stage is closed")
        self.kernel.require_open()

    def operation(
        self,
        *,
        extent,
        tile_origin=(0, 0),
        tile_extent=None,
        sample_index=0,
        sample_count=1,
        capture_secondary=False,
        after=(),
    ):
        self.require_open()
        width, height = map(index, extent)
        x, y = map(index, tile_origin)
        tw, th = map(index, tile_extent if tile_extent is not None else (width, height))
        sample_index, sample_count = index(sample_index), index(sample_count)
        if (
            min(width, height, tw, th) <= 0
            or min(x, y) < 0
            or x + tw > width
            or y + th > height
            or tw * th > self.capacity
            or width * height > 0xFFFFFFFF
        ):
            raise ValueError("Invalid ray-generation tile or image extent")
        if not (0 <= sample_index <= 0xFFFFFFFF and 0 < sample_count <= 0x7FFFFFFF):
            raise ValueError("Invalid ray-generation sample index/count")
        constants = struct.pack(
            "8I",
            width,
            height,
            x,
            y,
            tw,
            th,
            sample_count | (0x80000000 if capture_secondary else 0),
            sample_index,
        )
        queue = self.bindings[0]

        def reset(command):
            for offset, value in ((0, 0), (4, self.capacity), (8, 0)):
                vk.vkCmdFillBuffer(command, queue.handle, offset, 4, value)

        after = tuple(after)
        return VulkanOperation(
            [
                VulkanPass(
                    "rays.reset",
                    (
                        VulkanResourceUse(
                            queue,
                            vk.VK_PIPELINE_STAGE_TRANSFER_BIT,
                            vk.VK_ACCESS_TRANSFER_WRITE_BIT,
                        ),
                    ),
                    reset,
                ),
                VulkanPass(
                    "rays.generate",
                    tuple(
                        VulkanResourceUse(
                            resource,
                            vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                            vk.VK_ACCESS_SHADER_READ_BIT
                            if i == 3
                            else vk.VK_ACCESS_SHADER_READ_BIT
                            | vk.VK_ACCESS_SHADER_WRITE_BIT,
                        )
                        for i, resource in self.bindings.items()
                    ),
                    lambda command: self.kernel.bind(command, constants),
                    ((tw + 7) // 8, (th + 7) // 8, 1),
                ),
            ],
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
