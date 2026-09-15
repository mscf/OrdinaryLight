"""Resolve split path records into HDR and optional indirect-light signals."""

from importlib.resources import files
from operator import index
import struct
import vulkan as vk
from .kernel import VulkanKernel
from ..pipeline.graph import VulkanOperation
from ..pipeline.vulkan import VulkanPass, VulkanResource, VulkanResourceUse


class VulkanPathResolve:
    """Borrow path/signal buffers and an RGBA16F HDR image.

    Pixel indices come from path metadata; each dispatch must contain at most one
    path per pixel. Untouched pixels are preserved. All six shader bindings are
    explicit, including buffers unused when capture or seeding is disabled.
    operation(seed_reservoirs=False) resolves signals without producing indirect
    seeds; sampled_indirect=True then leaves all secondary records untouched.
    """

    def __init__(
        self,
        runtime,
        *,
        paths,
        hdr,
        secondary_paths,
        reservoirs,
        camera,
        seeds,
        capacity,
        reservoir_extent=(1, 1),
    ):
        self.runtime, self.closed, self.completion = runtime, False, None
        self.capacity = index(capacity)
        self.extent = (hdr.width, hdr.height)
        self.reservoir_extent = tuple(map(index, reservoir_extent))
        if not 0 < self.capacity <= 0xFFFFFFFF:
            raise ValueError("Resolve capacity must fit a positive uint32")
        if len(self.reservoir_extent) != 2 or any(
            not 0 < n <= limit for n, limit in zip(self.reservoir_extent, self.extent)
        ):
            raise ValueError("Reservoir extent must fit the HDR extent")
        count = self.reservoir_extent[0] * self.reservoir_extent[1]
        buffers = (paths, secondary_paths, reservoirs, camera, seeds)
        sizes = (self.capacity * 48, self.capacity * 128, count * 24, 64, count * 4)
        with runtime.lock:
            runtime.require_open()
            if hdr.format != vk.VK_FORMAT_R16G16B16A16_SFLOAT:
                raise ValueError("Path resolve requires RGBA16F HDR")
            for buffer, size in zip(buffers, sizes):
                buffer.require_open()
                if buffer.byte_size < size:
                    raise ValueError("Resolve buffer is smaller than required storage")
            if len({buffer.buffer for buffer in buffers}) != len(buffers):
                raise ValueError("Resolve buffers must not alias")
            self.kernel = VulkanKernel(
                runtime,
                files("ordinarylight.shaders")
                .joinpath("wavefront_path_to_hdr.comp.spv")
                .read_bytes(),
                {
                    0: VulkanResource.buffer(paths),
                    1: VulkanResource.image(hdr),
                    2: VulkanResource.buffer(secondary_paths),
                    3: VulkanResource.buffer(reservoirs),
                    4: VulkanResource.buffer(camera),
                    5: VulkanResource.buffer(seeds),
                },
                push_constant_size=32,
            )

    def require_open(self):
        if self.closed:
            raise RuntimeError("Path resolve stage is closed")
        self.kernel.require_open()

    def operation(
        self,
        *,
        path_count,
        sample_index=0,
        sample_count=1,
        capture_secondary=False,
        sampled_indirect=False,
        seed_reservoirs=True,
        after=(),
    ):
        self.require_open()
        operation = path_resolve_operation(
            self.kernel,
            capacity=self.capacity,
            extent=self.extent,
            reservoir_extent=self.reservoir_extent,
            path_count=path_count,
            sample_index=sample_index,
            sample_count=sample_count,
            capture_secondary=capture_secondary,
            sampled_indirect=sampled_indirect,
            seed_reservoirs=seed_reservoirs,
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


def path_resolve_operation(
    kernel,
    *,
    capacity,
    extent,
    reservoir_extent,
    path_count,
    sample_index=0,
    sample_count=1,
    capture_secondary=False,
    sampled_indirect=False,
    seed_reservoirs=True,
):
    """Resolve HDR, optionally legacy signals and indirect reservoir seeds.

    seed_reservoirs=False leaves reservoir/seed buffers untouched. With sampled
    indirect preparation this is HDR-only and preserves the entire secondary
    record; legacy capture still writes resolved lobe signals. Native dormant
    placeholders may alias secondary storage only when seeding is disabled.
    """
    kernel.require_open()
    path_count, sample_index, sample_count = map(
        index, (path_count, sample_index, sample_count)
    )
    if not 0 <= path_count <= capacity:
        raise ValueError("Path count exceeds resolve capacity")
    if not 0 <= sample_index < sample_count <= 0xFFFFFFFF:
        raise ValueError("Invalid resolve sample index/count")
    if sampled_indirect and not capture_secondary:
        raise ValueError("Sampled indirect requires secondary capture")
    constants = struct.pack(
        "8I",
        path_count,
        *extent,
        sample_index,
        sample_count,
        int(bool(capture_secondary)) | (2 if sampled_indirect else 0)
        | (4 if capture_secondary and not seed_reservoirs else 0),
        *reservoir_extent,
    )
    capture = capture_secondary and sample_index + 1 == sample_count
    seeds = capture and seed_reservoirs
    signals = capture and not sampled_indirect
    if seeds:
        handles=[kernel.bindings[b].handle for b in (0,2,3,4,5)]
        if len(set(handles)) != len(handles):
            raise ValueError("Active resolve seed buffers must not alias path, secondary or camera storage")
    uses = {}
    for binding, resource in kernel.bindings.items():
        if binding == 2 and not (signals or seeds):
            continue
        if binding in (3,4,5) and not seeds:
            continue
        access = vk.VK_ACCESS_SHADER_READ_BIT
        if binding == 1:
            access = vk.VK_ACCESS_SHADER_WRITE_BIT | (access if sample_index else 0)
        elif binding == 2:
            access |= vk.VK_ACCESS_SHADER_WRITE_BIT if signals else 0
        elif binding in (3,5):
            access = vk.VK_ACCESS_SHADER_WRITE_BIT
        key = (resource.kind, resource.handle)
        previous = uses.get(key)
        if previous is not None:
            access |= previous.access
        uses[key] = VulkanResourceUse(
            resource,
            vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            access,
            vk.VK_IMAGE_LAYOUT_GENERAL if binding == 1 else None,
        )

    def record(command):
        kernel.bind(command, constants)
        vk.vkCmdDispatch(command, (path_count + 63) // 64, 1, 1)

    return VulkanOperation(
        [VulkanPass("resolve", tuple(uses.values()), record)],
        validate=kernel.require_open,
    )
