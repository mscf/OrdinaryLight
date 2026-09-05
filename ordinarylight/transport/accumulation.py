"""Persistent GPU sums/counts keyed by application IDs, with a resident HDR resolve."""

import struct
from operator import index

import numpy as np
from ._synchronization import serialized

ACCUMULATION_DTYPE = np.dtype(
    [("radiance", "<f4", (4,)), ("counts", "<u4", (4,)), ("events", "<u4", (4,))]
)

_RESOLVE = """#version 460
layout(local_size_x=64) in;
struct SampleAccumulation { vec4 radiance; uvec4 counts; uvec4 events; };
layout(set=0,binding=0,std430) readonly buffer Samples { SampleAccumulation accumulated[]; };
layout(set=0,binding=1,rgba32f) writeonly uniform image2D hdr;
layout(push_constant) uniform Constants { uint width; uint height; uint capacity; } pc;
void main() {
    uint i=gl_GlobalInvocationID.x; if(i>=pc.width*pc.height) return;
    vec3 value=vec3(0);
    if(i<pc.capacity) {
        SampleAccumulation state=accumulated[i];
        value=state.counts.y>0u?state.radiance.rgb/float(state.counts.y):vec3(0);
        if(state.counts.z!=0u) value=vec3(1,0,1);
    }
    imageStore(hdr,ivec2(i%pc.width,i/pc.width),vec4(value,1));
}
"""


class GpuSampleAccumulator:
    def __init__(self, runtime, capacity, *, extent=None):
        with runtime.lock:
            from ..runtime import VulkanKernel, compile_compute
            from ..pipeline.vulkan import VulkanResource

            self.runtime = runtime
            self.capacity = index(capacity)
            if not 1 <= self.capacity <= 16_777_216:
                raise ValueError("Accumulator capacity must be between 1 and 16777216")
            self.extent = tuple(map(index, extent or (self.capacity, 1)))
            if (
                len(self.extent) != 2
                or min(self.extent) <= 0
                or self.extent[0] * self.extent[1] < self.capacity
            ):
                raise ValueError("HDR extent must cover all accumulator identities")
            self.closed = False
            self._borrowers = set()
            self.last_completion = None
            self.buffer = self.hdr = self._resolve_kernel = None
            runtime.retain(self)
            try:
                self.buffer = runtime.buffer(
                    self.capacity * ACCUMULATION_DTYPE.itemsize,
                    data=np.zeros(self.capacity, ACCUMULATION_DTYPE),
                )
                self.hdr = runtime.image(*self.extent)
                self._resolve_kernel = VulkanKernel(
                    runtime,
                    compile_compute(_RESOLVE),
                    {
                        0: VulkanResource.buffer(self.buffer),
                        1: VulkanResource.image(self.hdr),
                    },
                    push_constant_size=12,
                )
            except Exception:
                self.close()
                raise

    def require_open(self):
        self.runtime.require_open()
        if self.closed:
            raise RuntimeError("Sample accumulator is closed")

    @serialized
    def resolve(self, *, after=()):
        import vulkan as vk
        from ..pipeline.vulkan import (
            VulkanResource,
            VulkanResourceUse,
            VulkanPass,
            VulkanPassPipeline,
        )

        self.require_open()
        dependencies = tuple(after) + (
            (self.last_completion,) if self.last_completion is not None else ()
        )
        uses = (
            VulkanResourceUse(
                VulkanResource.buffer(self.buffer),
                vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                vk.VK_ACCESS_SHADER_READ_BIT,
            ),
            VulkanResourceUse(
                VulkanResource.image(self.hdr),
                vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                vk.VK_ACCESS_SHADER_WRITE_BIT,
                vk.VK_IMAGE_LAYOUT_GENERAL,
            ),
        )
        self.last_completion = VulkanPassPipeline(
            [
                VulkanPass(
                    "resolve_sample_hdr",
                    uses,
                    lambda command: self._resolve_kernel.bind(
                        command, struct.pack("III", *self.extent, self.capacity)
                    ),
                    ((self.extent[0] * self.extent[1] + 63) // 64, 1, 1),
                )
            ]
        ).execute(self.runtime, after=dependencies)
        return self.last_completion

    @serialized
    def read(self, *, strict=True):
        self.require_open()
        if self.last_completion is not None:
            self.last_completion.wait()
        records = np.frombuffer(self.buffer.read(), ACCUMULATION_DTYPE).copy()
        if strict and np.any(records["counts"][:, 2]):
            flags = int(np.bitwise_or.reduce(records["counts"][:, 2]))
            raise RuntimeError(
                f"Transport contains invalid paths (status mask {flags:#x}); inspect read(strict=False)"
            )
        return records

    @serialized
    def means(self, *, strict=True):
        records = self.read(strict=strict)
        return np.divide(
            records["radiance"][:, :3],
            records["counts"][:, 1, None],
            out=np.zeros((self.capacity, 3), np.float64),
            where=records["counts"][:, 1, None] != 0,
        )

    @serialized
    def reset(self, identities=None, *, after=()):
        import vulkan as vk
        from ..pipeline.vulkan import (
            VulkanResource,
            VulkanResourceUse,
            VulkanPass,
            VulkanPassPipeline,
        )

        self.require_open()
        selected = None if identities is None else tuple(index(i) for i in identities)
        if selected is not None and any(i < 0 or i >= self.capacity for i in selected):
            raise ValueError("Reset identity exceeds accumulator capacity")
        dependencies = tuple(after) + (
            (self.last_completion,) if self.last_completion is not None else ()
        )

        def clear(command):
            if selected is None:
                vk.vkCmdFillBuffer(command, self.buffer.buffer, 0, self.buffer.size, 0)
            else:
                for i in selected:
                    vk.vkCmdFillBuffer(
                        command,
                        self.buffer.buffer,
                        i * ACCUMULATION_DTYPE.itemsize,
                        ACCUMULATION_DTYPE.itemsize,
                        0,
                    )

        use = VulkanResourceUse(
            VulkanResource.buffer(self.buffer),
            vk.VK_PIPELINE_STAGE_TRANSFER_BIT,
            vk.VK_ACCESS_TRANSFER_WRITE_BIT,
        )
        self.last_completion = VulkanPassPipeline(
            [VulkanPass("reset_sample_history", (use,), clear)]
        ).execute(self.runtime, after=dependencies)
        return self.last_completion

    @serialized
    def close(self):
        if self.closed:
            return
        if self._borrowers:
            raise RuntimeError("Close transport integrators before their accumulator")
        if self.last_completion is not None:
            self.last_completion.wait()
        if self._resolve_kernel is not None:
            self._resolve_kernel.close()
        if self.hdr is not None:
            self.hdr.close()
        if self.buffer is not None:
            self.buffer.close()
        self.closed = True
        self.runtime.release(self)

    def __enter__(self):
        self.require_open()
        return self

    def __exit__(self, *_exc):
        self.close()
