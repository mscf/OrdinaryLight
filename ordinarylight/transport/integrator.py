"""Reusable, non-camera multi-bounce Vulkan transport."""

from importlib.resources import files
from operator import index
import struct

import numpy as np
from ._synchronization import serialized

from . import shader_source
from .gpu_samples import GpuTransportSamples, SampleReduction, validate_samples
from ._custom_resources import resource_uses
from ._shaders import scene_source, SCENE_BINDINGS
from .media import MediumStack


class VulkanTransportIntegrator:
    """Path trace arbitrary rays/surfaces into persistent application-ID sums.

    Initial materials: Lambertian diffuse and ideal dielectric. Illumination is
    constant environment plus emissive surfaces, sampled by BSDF continuation.
    This is a bounded integrator, not the screen-space GI scheduler.
    """

    def __init__(
        self, scene, samples, accumulator, *, initial_boundaries=(), reduction=None
    ):
        with scene.runtime.lock:
            from ..runtime import VulkanKernel, compile_compute
            from ..pipeline.vulkan import VulkanResource

            self.scene = scene
            self.runtime = scene.runtime
            self.accumulator = accumulator
            scene.require_open()
            accumulator.require_open()
            if accumulator.runtime is not self.runtime:
                raise ValueError("Scene and accumulator must share a runtime")
            self._owns_samples = not isinstance(samples, GpuTransportSamples)
            prepared = validate_samples(samples, scene) if self._owns_samples else None
            if not self._owns_samples:
                samples.require_open()
                if samples.runtime is not self.runtime:
                    raise ValueError("GPU samples must share the scene runtime")
                if reduction is None:
                    raise ValueError("GPU samples require an explicit SampleReduction")
            count = len(prepared) if prepared is not None else samples.count
            capacity = count if prepared is not None else samples.capacity
            if reduction is None:
                reduction = SampleReduction(prepared["identity"][:, 0])
            if (
                not isinstance(reduction, SampleReduction)
                or len(reduction.output_ids) != count
            ):
                raise ValueError("Reduction must map every active input slot")
            groups, order = reduction.pack(accumulator.capacity)
            initial = np.zeros((8, 4), np.uint32)
            initial[0, 1] = 0xFFFFFFFF
            reference = MediumStack()
            for i, identity in enumerate(initial_boundaries, 1):
                if identity not in scene.boundary_indices:
                    raise ValueError("Unknown initial medium boundary")
                boundary_index = scene.boundary_indices[identity]
                boundary = scene.boundaries[boundary_index]
                reference.transmit(boundary, True)
                initial[i, :2] = (boundary.inside, boundary_index)
            self.initial_depth = len(reference.media)
            self._sample_offset = 0
            self.closed = False
            self._kernel = self._initial = self._reducer = None
            self.samples = None
            self._scratch = self._groups = self._indices = None
            self.capacity = capacity
            self.runtime.retain(self)
            scene._borrowers.add(self)
            accumulator._borrowers.add(self)
            try:
                self.samples = (
                    GpuTransportSamples(self.runtime, capacity, samples=prepared)
                    if self._owns_samples
                    else samples
                )
                self.samples._borrowers.add(self)
                self._scratch = self.runtime.buffer(capacity * 48)
                self._groups = self.runtime.buffer(capacity * 16, data=groups)
                self._indices = self.runtime.buffer(capacity * 4, data=order)
                self._mapped_count = count
                self._group_count = len(groups)
                self._initial = self.runtime.buffer(initial.nbytes, data=initial)
                self.bindings = {
                    i: scene.resource(name) for i, name in enumerate(SCENE_BINDINGS)
                }
                self.bindings.update(scene.custom_bindings)
                self.bindings.update(
                    {
                        8: VulkanResource.buffer(self.samples.buffer),
                        9: VulkanResource.buffer(self._scratch),
                        10: VulkanResource.buffer(self._initial),
                    }
                )
                source = (
                    scene_source(scene)
                    + shader_source("contracts")
                    + shader_source("sampling")
                    + shader_source("dielectric")
                )
                source += (
                    files("ordinarylight.shaders")
                    .joinpath("transport_v1/integrator.glsl")
                    .read_text()
                )
                self._kernel = VulkanKernel(
                    self.runtime,
                    compile_compute(source),
                    self.bindings,
                    push_constant_size=64,
                )
                self._reduce_bindings = {
                    0: VulkanResource.buffer(self._scratch),
                    1: VulkanResource.buffer(accumulator.buffer),
                    2: VulkanResource.buffer(self._groups),
                    3: VulkanResource.buffer(self._indices),
                }
                self._reducer = VulkanKernel(
                    self.runtime,
                    compile_compute(
                        files("ordinarylight.shaders")
                        .joinpath("transport_v1/reduce.glsl")
                        .read_text()
                    ),
                    self._reduce_bindings,
                    push_constant_size=4,
                )
            except Exception:
                self.close()
                raise

    @property
    def count(self):
        return self.samples.count

    @serialized
    def update_samples(self, samples, *, reduction=None, after=()):
        """Upload new inputs without rebuilding kernels; history is not reset."""
        self.require_open()
        prepared = validate_samples(samples, self.scene)
        if len(prepared) > self.capacity:
            raise ValueError("Sample update exceeds integrator capacity")
        mapping = (
            SampleReduction(prepared["identity"][:, 0])
            if reduction is None
            else reduction
        )
        if not isinstance(mapping, SampleReduction) or len(mapping.output_ids) != len(
            prepared
        ):
            raise ValueError("Reduction must map every active input slot")
        mapping.pack(self.accumulator.capacity)
        self.samples.update(prepared, after=after)
        self.set_reduction(mapping)

    @serialized
    def set_reduction(self, reduction, *, after=()):
        """Replace output grouping for the current active input count."""
        self.require_open()
        if (
            not isinstance(reduction, SampleReduction)
            or len(reduction.output_ids) != self.count
        ):
            raise ValueError("Reduction must map every active input slot")
        groups, order = reduction.pack(self.accumulator.capacity)
        self.samples._wait(after)
        self._groups.upload(groups)
        self._indices.upload(order)
        self._mapped_count = self.count
        self._group_count = len(groups)

    @serialized
    def accumulate(
        self,
        *,
        samples_per_element=1,
        max_bounces=8,
        seed=0,
        environment=(0, 0, 0),
        tolerance=1e-5,
        ray_epsilon=1e-4,
        max_steps=256,
        max_distance=1e6,
        after=(),
    ):
        from ..pipeline.vulkan import VulkanPass, VulkanPassPipeline

        self.require_open()
        if self.count != self._mapped_count:
            raise ValueError("Sample count changed; call set_reduction before dispatch")
        samples_per_element = index(samples_per_element)
        max_bounces = index(max_bounces)
        seed = index(seed)
        max_steps = index(max_steps)
        if (
            not 1 <= samples_per_element <= 65536
            or not 0 <= max_bounces <= 128
            or not 0 <= seed < 2**32
            or not 1 <= max_steps <= 65536
        ):
            raise ValueError("Invalid sample, bounce, seed or traversal limits")
        if self._sample_offset + samples_per_element >= 2**24:
            raise ValueError(
                "Reset/recreate the integrator before exhausting its sample epoch"
            )
        environment = np.asarray(environment, dtype=np.float32)
        if (
            environment.shape != (3,)
            or not np.isfinite(environment).all()
            or np.any(environment < 0)
        ):
            raise ValueError("Environment must be finite nonnegative RGB radiance")
        if (
            not np.isfinite([tolerance, ray_epsilon, max_distance]).all()
            or not 0 < tolerance * 2 <= ray_epsilon < max_distance
        ):
            raise ValueError(
                "Use a ray epsilon at least twice the positive field tolerance"
            )
        push = struct.pack(
            "<6I2fIf2I4f",
            self.count,
            samples_per_element,
            max_bounces,
            self._sample_offset,
            seed,
            self.initial_depth,
            tolerance,
            ray_epsilon,
            max_steps,
            max_distance,
            0,
            0,
            *environment,
            0,
        )
        uses = resource_uses(self.bindings, writable=(9,))
        dependencies = tuple(after) + (
            (self.accumulator.last_completion,)
            if self.accumulator.last_completion is not None
            else ()
        )
        completion = VulkanPassPipeline(
            [
                VulkanPass(
                    "multi_bounce_surface_transport",
                    uses,
                    lambda command: self._kernel.bind(command, push),
                    ((self.count + 63) // 64, 1, 1),
                ),
                VulkanPass(
                    "reduce_transport_samples",
                    resource_uses(self._reduce_bindings, writable=(1,)),
                    lambda command: self._reducer.bind(
                        command, struct.pack("<I", self._group_count)
                    ),
                    ((self._group_count + 63) // 64, 1, 1),
                ),
            ]
        ).execute(self.runtime, after=dependencies)
        self.accumulator.last_completion = completion
        self._sample_offset += samples_per_element
        return completion

    def require_open(self):
        if self.closed:
            raise RuntimeError("Transport integrator is closed")
        self.scene.require_open()
        self.accumulator.require_open()
        self.samples.require_open()

    @serialized
    def close(self):
        if self.closed:
            return
        if self._kernel is not None:
            self._kernel.close()
        if self._reducer is not None:
            self._reducer.close()
        for buffer in (self._scratch, self._groups, self._indices):
            if buffer is not None:
                buffer.close()
        if self.samples is not None:
            self.samples._borrowers.discard(self)
            if self._owns_samples:
                self.samples.close()
        if self._initial is not None:
            self._initial.close()
        self.scene._borrowers.discard(self)
        self.accumulator._borrowers.discard(self)
        self.closed = True
        self.runtime.release(self)

    def __enter__(self):
        self.require_open()
        return self

    def __exit__(self, *_exc):
        self.close()
