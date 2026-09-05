"""Reusable, non-camera multi-bounce Vulkan transport."""

from importlib.resources import files
from operator import index
import struct

import numpy as np
from ._synchronization import serialized

from . import SURFACE_SAMPLE_DTYPE, shader_source
from ._shaders import scene_source, SCENE_BINDINGS
from .media import MediumStack


class VulkanTransportIntegrator:
    """Path trace arbitrary rays/surfaces into persistent application-ID sums.

    Initial materials: Lambertian diffuse and ideal dielectric. Illumination is
    constant environment plus emissive surfaces, sampled by BSDF continuation.
    This is a bounded integrator, not the screen-space GI scheduler.
    """

    def __init__(self, scene, samples, accumulator, *, initial_boundaries=()):
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
            samples = np.array(samples, copy=True)
            if (
                samples.dtype != SURFACE_SAMPLE_DTYPE
                or samples.ndim != 1
                or not len(samples)
            ):
                raise ValueError(
                    "Use ray_samples() or surface_samples() to supply samples"
                )
            ids = samples["identity"][:, 0]
            if len(np.unique(ids)) != len(ids) or np.max(ids) >= accumulator.capacity:
                raise ValueError(
                    "Sample identities must be unique and fit the accumulator"
                )
            if (
                not np.isfinite(samples["position"]).all()
                or not np.isfinite(samples["incoming"]).all()
                or not np.allclose(
                    np.linalg.norm(samples["incoming"][:, :3], axis=1), 1, rtol=1e-5
                )
            ):
                raise ValueError(
                    "Samples need finite positions and normalized incoming directions"
                )
            surface = (samples["identity"][:, 3] & 1) != 0
            if np.any(samples["identity"][:, 3] > 1):
                raise ValueError("Unknown sample mode")
            for i in np.flatnonzero(surface):
                material_index = int(samples["identity"][i, 2])
                if material_index >= len(scene.materials):
                    raise ValueError("Surface sample material is out of range")
                for field in ("geometric_normal", "shading_normal"):
                    if not np.isfinite(samples[field][i]).all() or not np.isclose(
                        np.linalg.norm(samples[field][i, :3]), 1, rtol=1e-5
                    ):
                        raise ValueError(
                            "Surface sample normals must be finite unit vectors"
                        )
                boundary = int(samples["media"][i, 2])
                boundary = None if boundary == 0xFFFFFFFF else boundary
                samples["media"][i, 2] = scene._boundary_index(
                    boundary, scene.materials[material_index]
                )
                if boundary is not None:
                    definition = scene.boundaries[scene.boundary_indices[boundary]]
                    samples["media"][i, :2] = (definition.outside, definition.inside)
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
            self.count = len(samples)
            self._sample_offset = 0
            self.closed = False
            self._kernel = self._input = self._initial = None
            self.runtime.retain(self)
            scene._borrowers.add(self)
            accumulator._borrowers.add(self)
            try:
                self._input = self.runtime.buffer(samples.nbytes, data=samples)
                self._initial = self.runtime.buffer(initial.nbytes, data=initial)
                self.bindings = {
                    i: scene.resource(name) for i, name in enumerate(SCENE_BINDINGS)
                }
                self.bindings.update(
                    {
                        8: VulkanResource.buffer(self._input),
                        9: VulkanResource.buffer(accumulator.buffer),
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
            except Exception:
                self.close()
                raise

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
        import vulkan as vk
        from ..pipeline.vulkan import VulkanResourceUse, VulkanPass, VulkanPassPipeline

        self.require_open()
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
        uses = tuple(
            VulkanResourceUse(
                resource,
                vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                vk.VK_ACCESS_SHADER_READ_BIT | vk.VK_ACCESS_SHADER_WRITE_BIT
                if binding == 9
                else vk.VK_ACCESS_SHADER_READ_BIT,
            )
            for binding, resource in self.bindings.items()
        )
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
                )
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

    @serialized
    def close(self):
        if self.closed:
            return
        if self._kernel is not None:
            self._kernel.close()
        if self._input is not None:
            self._input.close()
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
