"""GPU-discovered occupancy and bounds within explicit reserved scene capacity."""

from importlib.resources import files
import numpy as np

from ._custom_batch import CUSTOM_DTYPE

GPU_CUSTOM_GEOMETRY_DTYPE = CUSTOM_DTYPE


class GpuCustomGeometry:
    """Validate GPU records, update slot metadata/bounds, and refit acceleration.

    Source has one 64-byte record per reserved slot. metadata is program index
    (0xffffffff disables), custom-local material index, application boundary ID
    (0xffffffff for none), and application geometry ID. Invalid active records
    are disabled and diagnosed before any bounds reach Vulkan acceleration.
    """

    def __init__(self, scene, source):
        from ..runtime import VulkanKernel, compile_compute
        from ..pipeline.vulkan import VulkanResource
        from ._dynamic_scene import _buffer

        with scene.runtime.lock:
            scene.require_open()
            source.require_open()
            if (
                source.runtime is not scene.runtime
                or source.byte_size != scene.custom_capacity * 64
                or not scene.custom_capacity
            ):
                raise ValueError(
                    "GPU geometry source must cover exactly the reserved scene slots"
                )
            self.scene, self.runtime = scene, scene.runtime
            self.binding_revision = scene.binding_revision
            self.closed = False
            self.kernel = self.diagnostics = None
            self.last_completion = None
            scene._gpu_geometry_clients.add(self)
            try:
                self.diagnostics = self.runtime.buffer(scene.custom_capacity * 4)
                self.bindings = {
                    0: VulkanResource.buffer(source),
                    1: scene.resource("custom"),
                    2: _buffer(scene, scene._bounds_buffer),
                    3: VulkanResource.buffer(self.diagnostics),
                }
                header = f"#version 460\n#define CAPACITY {scene.custom_capacity}u\n#define PROGRAMS {len(scene.programs)}u\n#define MATERIALS {len(scene._custom_materials)}u\n#define TRIANGLES {scene.triangle_count}u\n"
                from ..shaders.dynamic import lookup_source
                header += lookup_source('boundaryIndex', 'identity',
                    ((boundary.identity, i) for i, boundary in enumerate(scene.boundaries)), 0xffffffff)
                header += lookup_source('dielectricMaterial', 'index',
                    ((i, material.kind == 'dielectric') for i, material in enumerate(scene._custom_materials)),
                    False, boolean=True)
                shader = (
                    files("ordinarylight.shaders")
                    .joinpath("transport_v1/gpu_geometry.glsl")
                    .read_text()
                )
                self.kernel = VulkanKernel(
                    self.runtime, compile_compute(header + shader), self.bindings
                )
            except Exception:
                self.close()
                raise

    def require_open(self):
        self.scene.require_open()
        if self.closed:
            raise RuntimeError("GPU geometry client is closed")
        if self.scene.binding_revision != self.binding_revision:
            raise ValueError("Scene bindings changed; recreate GPU geometry client")

    def operation(self, *, mode="refit", after=()):
        from ..pipeline.vulkan import VulkanPass
        from ..pipeline.graph import VulkanOperation
        from ._custom_resources import resource_uses
        from ._dynamic_scene import acceleration_passes

        self.require_open()
        if mode not in {"refit", "rebuild"}:
            raise ValueError("GPU acceleration mode must be refit or rebuild")
        passes = [
            VulkanPass(
                "validate_gpu_geometry",
                resource_uses(self.bindings, writable=(1, 2, 3)),
                lambda command: self.kernel.bind(command),
                ((self.scene.custom_capacity + 63) // 64, 1, 1),
            )
        ]
        passes.extend(acceleration_passes(self.scene, mode))

        def submitted(completion):
            self.last_completion = self.scene.last_completion = completion
            self.scene.geometry_revision += 1
            self.scene._gpu_geometry_dirty = True

        return VulkanOperation(
            passes,
            validate=self.require_open,
            submitted=submitted,
            dependencies=lambda: tuple(after)
            + ((self.scene.last_completion,) if self.scene.last_completion else ()),
        )

    def read_diagnostics(self):
        self.require_open()
        if self.last_completion:
            self.last_completion.wait()
        return np.frombuffer(self.diagnostics.read(), np.uint32).copy()

    def close(self):
        with self.runtime.lock:
            if self.closed:
                return
            if self.kernel:
                self.kernel.close()
            if self.diagnostics:
                self.diagnostics.close()
            self.scene._gpu_geometry_clients.discard(self)
            self.closed = True

    def __enter__(self):
        self.require_open()
        return self

    def __exit__(self, *_exc):
        self.close()


def synchronize_geometry_shadow(scene):
    """Explicit growth boundary only: retain device-authored slots across reserve."""
    from ..geometry import CustomGeometry

    records = np.frombuffer(scene._buffers["custom"].read(), GPU_CUSTOM_GEOMETRY_DTYPE)
    programs = tuple(scene.programs.values())
    slots = []
    for record in records:
        program, material, boundary, identity = map(int, record["metadata"])
        if program == 0xFFFFFFFF:
            slots.append(None)
        else:
            slots.append(
                CustomGeometry(
                    (record["lower"][:3], record["upper"][:3]),
                    programs[program],
                    tuple(record["parameters"]),
                    material - scene.triangle_count,
                    None
                    if boundary == 0xFFFFFFFF
                    else scene.boundaries[boundary].identity,
                    identity,
                )
            )
    scene.custom_geometry = tuple(slots)
    scene._custom_bounds = np.array(
        [
            (*slot.bounds[0], *slot.bounds[1]) if slot else (0, 0, 0, 1e-4, 1e-4, 1e-4)
            for slot in slots
        ],
        np.float32,
    )
    scene._gpu_geometry_dirty = False
