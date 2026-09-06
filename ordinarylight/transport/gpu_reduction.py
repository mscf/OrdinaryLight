"""GPU-authored counts and deterministic reduction maps, with checked dispatch."""

from importlib.resources import files
from operator import index
import struct

import numpy as np
import vulkan as vk

from ._custom_resources import resource_uses


class GpuSampleReduction:
    """GPU producer ABI: counts uvec4, groups uvec4[], indices uint[], weights vec2[].

    counts.xy are active samples/groups. Groups contain output ID, first index,
    length, reserved. Groups partition indices contiguously in ascending output
    ID order; each group's input indices are strictly ascending. Input identity.x
    must equal its group's output ID. Thus all active inputs occur exactly once.
    Weights are indexed by input slot, as in SampleReduction.
    """

    def __init__(self, runtime, capacity, *, group_capacity=None):
        with runtime.lock:
            self._initialize(runtime, capacity, group_capacity=group_capacity)

    def _initialize(self, runtime, capacity, *, group_capacity):
        runtime.require_open()
        self.runtime = runtime
        self.capacity = index(capacity)
        self.group_capacity = (
            self.capacity if group_capacity is None else index(group_capacity)
        )
        limit = (
            vk.vkGetPhysicalDeviceProperties(
                runtime.physical_device
            ).limits.maxComputeWorkGroupCount[0]
            * 64
        )
        if not 1 <= self.group_capacity <= self.capacity <= min(16_777_216, limit):
            raise ValueError(
                "GPU reduction capacities exceed supported dispatch limits"
            )
        self.closed = False
        self._borrowers = set()
        self.counts = self.groups = self.indices = self.weights = None
        with runtime.lock:
            runtime.retain(self)
            try:
                self.counts = runtime.buffer(16, data=np.zeros(4, np.uint32))
                self.groups = runtime.buffer(self.group_capacity * 16)
                self.indices = runtime.buffer(self.capacity * 4)
                self.weights = runtime.buffer(
                    self.capacity * 8, data=np.ones((self.capacity, 2), np.float32)
                )
            except Exception:
                self.close()
                raise

    def require_open(self):
        self.runtime.require_open()
        if self.closed:
            raise RuntimeError("GPU reduction is closed")
        for buffer in (self.counts, self.groups, self.indices, self.weights):
            buffer.require_open()

    def close(self):
        with self.runtime.lock:
            if self.closed:
                return
            if self._borrowers:
                raise RuntimeError("Close transport integrators before GPU reduction")
            for buffer in (self.counts, self.groups, self.indices, self.weights):
                if buffer is not None:
                    buffer.close()
            self.closed = True
            self.runtime.release(self)

    def __enter__(self):
        self.require_open()
        return self

    def __exit__(self, *_exc):
        self.close()


def indirect_uses(uses, state):
    """Declare both compute access and indirect-command reads for one allocation."""
    from dataclasses import replace

    return tuple(
        replace(
            use,
            stage=use.stage | vk.VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT,
            access=use.access | vk.VK_ACCESS_INDIRECT_COMMAND_READ_BIT,
        )
        if use.resource.owner is state
        else use
        for use in uses
    )


class _GpuReductionPlan:
    def __init__(self, reduction, samples, accumulator):
        from ..runtime import VulkanKernel, compile_compute
        from ..pipeline.vulkan import VulkanResource
        from . import shader_source

        self.reduction = reduction
        self.runtime = reduction.runtime
        self.output_capacity = accumulator.capacity
        self.state = self.kernel = None
        limit = (
            vk.vkGetPhysicalDeviceProperties(
                self.runtime.physical_device
            ).limits.maxComputeWorkGroupCount[0]
            * 64
        )
        if self.output_capacity > limit:
            raise ValueError(
                "Accumulator capacity exceeds GPU reduction dispatch limit"
            )
        try:
            self.state = self.runtime.buffer(
                80,
                usage=vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                | vk.VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT,
            )
            self.bindings = {
                i: VulkanResource.buffer(buffer)
                for i, buffer in enumerate(
                    (
                        reduction.counts,
                        reduction.groups,
                        reduction.indices,
                        reduction.weights,
                        samples.buffer,
                        self.state,
                        accumulator.buffer,
                    )
                )
            }
            source = "#version 460\n" + shader_source("contracts")
            source += (
                files("ordinarylight.shaders")
                .joinpath("transport_v1/gpu_reduction.glsl")
                .read_text()
            )
            self.kernel = VulkanKernel(
                self.runtime,
                compile_compute(source),
                self.bindings,
                push_constant_size=16,
            )
        except Exception:
            self.close()
            raise

    def phase(self, mode, *, indirect_offset=None):
        from ..pipeline.vulkan import VulkanPass

        writable = (6,) if mode == 3 else (5,)
        uses = resource_uses(self.bindings, writable=writable)
        if indirect_offset is not None:
            uses = indirect_uses(uses, self.state)

        def record(command):
            self.kernel.bind(
                command,
                struct.pack(
                    "<4I",
                    self.reduction.capacity,
                    self.reduction.group_capacity,
                    self.output_capacity,
                    mode,
                ),
            )
            if indirect_offset is not None:
                vk.vkCmdDispatchIndirect(command, self.state.buffer, indirect_offset)

        return VulkanPass(
            (
                "gpu_reduction_counts",
                "gpu_reduction_validate",
                "gpu_reduction_dispatch",
                "gpu_reduction_errors",
            )[mode],
            uses,
            record,
            None if indirect_offset is not None else (1, 1, 1),
        )

    def close(self):
        if self.kernel is not None:
            self.kernel.close()
        if self.state is not None:
            self.state.close()
