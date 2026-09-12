"""Public triangle/custom-geometry scene for the non-camera transport path."""

from dataclasses import dataclass

import numpy as np
from ._synchronization import serialized

from .media import OpticalMedium, MediumBoundary


@dataclass(frozen=True)
class TransportMaterial:
    kind: str = "diffuse"
    albedo: tuple[float, float, float] = (0.8, 0.8, 0.8)
    emission: tuple[float, float, float] = (0.0, 0.0, 0.0)
    emission_two_sided: bool = False
    program: object = None
    roughness: float = 0.0
    metallic: float = 0.0
    ior: float = 1.5

    def __post_init__(self):
        from ..materials import (
            MaterialGraph,
            MaterialProgram,
            MaterialEvaluation,
            LayeredMaterialEvaluation,
        )

        program = (
            self.program.compile()
            if isinstance(self.program, MaterialGraph)
            else self.program
        )
        if program is not None:
            if not isinstance(program, MaterialProgram) or not isinstance(
                program.evaluation, (MaterialEvaluation, LayeredMaterialEvaluation)
            ):
                raise TypeError("Transport graphs must evaluate material parameters")
            if program.required_attributes:
                raise ValueError(
                    "Transport vertex attributes require an explicit attribute binding"
                )
            program.glsl()
        object.__setattr__(self, "program", program)
        if self.kind not in {"diffuse", "dielectric", "pbr", "emission"}:
            raise ValueError(
                "Transport materials are diffuse, dielectric, pbr, or emission"
            )
        if not np.isfinite(self.ior) or self.ior <= 0:
            raise ValueError("Material IOR must be finite and positive")
        if not np.isfinite([self.roughness, self.metallic]).all() or not (
            0 <= self.roughness <= 1 and 0 <= self.metallic <= 1
        ):
            raise ValueError("Roughness and metallic must be finite values in [0,1]")
        albedo, emission = np.asarray(self.albedo), np.asarray(self.emission)
        if (
            albedo.shape != (3,)
            or emission.shape != (3,)
            or not np.isfinite(albedo).all()
            or not np.isfinite(emission).all()
            or np.any(albedo < 0)
            or np.any(albedo > 1)
            or np.any(emission < 0)
        ):
            raise ValueError("Invalid material albedo or emission")
        object.__setattr__(self, "albedo", tuple(float(v) for v in albedo))
        object.__setattr__(self, "emission", tuple(float(v) for v in emission))

    @classmethod
    def from_material(cls, material):
        if (
            any(
                (
                    material.clearcoat,
                    material.anisotropy,
                    material.subsurface,
                    material.thin_walled,
                )
            )
            or material.opacity != 1
            or material.transmission not in (0, 1)
            or any(material.sheen_color)
        ):
            raise ValueError(
                "Supply an explicit TransportMaterial override for unsupported material lobes"
            )
        return cls(
            "dielectric" if material.transmission else "pbr",
            material.base_color,
            material.emission,
            material.emission_two_sided,
            material.program,
            material.roughness,
            material.metallic,
            material.ior,
        )

    def pack(self):
        return (
            *self.albedo,
            float({"diffuse": 0, "dielectric": 1, "pbr": 2, "emission": 3}[self.kind]),
            *self.emission,
            float(self.emission_two_sided),
            float(self.roughness),
            float(self.metallic),
            float(self.ior),
            0.0,
        )


class VulkanTransportScene:
    """Resident triangles and AABB custom geometry in one hardware TLAS.

    Triangles reuse OrdinaryLight's resident buffers and BLASes. Custom callbacks
    run at ray-query AABB candidates. Triangle source data remain a snapshot;
    custom slots can update/refit in place or grow at an explicit allocation
    boundary while preserving triangle, material and history resources.
    """

    def __init__(
        self,
        runtime,
        scene=None,
        *,
        resident=None,
        custom_geometry=(),
        custom_materials=(),
        media=(OpticalMedium(),),
        boundaries=(),
        triangle_boundaries=None,
        material_overrides=None,
        custom_resources=None,
        custom_capacity=None,
        intersection_programs=(),
        lights=None,
        material_resources=None,
    ):
        with runtime.lock:
            from ..targets.vulkan.scene import VulkanSceneUploader

            runtime.require_open()
            self.runtime = runtime
            self.closed = False
            self._borrowers = set()
            self._gpu_geometry_clients = set()
            self._custom_update_clients = set()
            self._gpu_geometry_dirty = False
            self._buffers = {}
            self._custom_owners = ()
            self._resident = None
            self._owns_resident = resident is None
            self._builder = VulkanSceneUploader(runtime)
            from ._custom_batch import CustomSlots, prepare_custom_geometry

            self.custom_geometry = (
                custom_geometry
                if isinstance(custom_geometry, CustomSlots)
                else CustomSlots(custom_geometry)
            )
            self.media = tuple(media)
            self.boundaries = tuple(boundaries)
            from ..scene import Scene
            from ..lights import PointLight, DirectionalLight, SpotLight

            self.lights = tuple(
                getattr(scene, "lights", ()) if lights is None else lights
            )
            if not all(
                isinstance(light, (PointLight, DirectionalLight, SpotLight))
                for light in self.lights
            ):
                raise TypeError(
                    "Transport lights must be point, directional, or spot lights"
                )
            light_records = Scene(lights=list(self.lights)).analytic_light_data()
            self.programs = {}
            from ..geometry import IntersectionProgram

            for program in intersection_programs:
                if (
                    not isinstance(program, IntersectionProgram)
                    or program.name in self.programs
                ):
                    raise ValueError("Supply unique IntersectionProgram declarations")
                self.programs[program.name] = program
            from operator import index

            self.custom_capacity = (
                len(self.custom_geometry)
                if custom_capacity is None
                else index(custom_capacity)
            )
            if (
                self.custom_capacity < len(self.custom_geometry)
                or self.custom_capacity < 0
            ):
                raise ValueError("Custom capacity must cover supplied geometry")
            self.geometry_revision = 0
            self.binding_revision = 0
            self.last_completion = None
            self._custom_materials = tuple(custom_materials)
            self._source_scene = scene if resident is None else resident.scene
            if (
                scene is not None
                and resident is not None
                and resident.scene is not scene
            ):
                raise ValueError("Scene and resident snapshot disagree")
            if (
                not self.media
                or self.media[0] != OpticalMedium()
                or not all(isinstance(m, OpticalMedium) for m in self.media)
            ):
                raise ValueError(
                    "Medium zero must be vacuum; supply OpticalMedium definitions"
                )
            if not all(isinstance(b, MediumBoundary) for b in self.boundaries) or len(
                {b.identity for b in self.boundaries}
            ) != len(self.boundaries):
                raise ValueError("Boundary identities must be unique")
            self.boundary_indices = {
                b.identity: i for i, b in enumerate(self.boundaries)
            }
            if any(
                max(b.inside, b.outside) >= len(self.media) for b in self.boundaries
            ):
                raise ValueError("Boundary references an unknown medium")
            if self._source_scene is not None and (
                self._source_scene.volumes or self._source_scene.textures
            ):
                raise ValueError(
                    "This transport path currently supports untextured surfaces and homogeneous dielectric media"
                )
            if self._source_scene is not None and self._source_scene.lights:
                raise ValueError(
                    "Use emissive geometry and the integrator environment; analytic light sampling is not yet supported here"
                )
            custom_materials = self._custom_materials
            if not all(isinstance(m, TransportMaterial) for m in custom_materials):
                raise TypeError("Expected TransportMaterial values")
            triangle_boundaries = dict(triangle_boundaries or {})
            material_overrides = dict(material_overrides or {})
            mesh_ids = (
                {mesh.id for mesh in self._source_scene.render_meshes}
                if self._source_scene is not None
                else set()
            )
            if (set(triangle_boundaries) | set(material_overrides)) - mesh_ids:
                raise ValueError(
                    "Triangle boundary/material mapping references an unknown mesh"
                )
            materials, triangle_records = [], []
            if self._source_scene is not None:
                for mesh in self._source_scene.render_meshes:
                    material = material_overrides.get(mesh.id)
                    if material is None:
                        material = TransportMaterial.from_material(mesh.material)
                    if not isinstance(material, TransportMaterial):
                        raise TypeError(
                            "Material overrides must be TransportMaterial values"
                        )
                    boundary = triangle_boundaries.get(mesh.id)
                    boundary_index = self._boundary_index(boundary, material)
                    for _ in mesh.indices:
                        triangle_records.append(
                            (len(materials), boundary_index, mesh.id, 0)
                        )
                        materials.append(material)
            self.triangle_count = len(triangle_records)
            packed, custom_bounds, self.custom_geometry = prepare_custom_geometry(
                self, self.custom_geometry, self.custom_capacity
            )
            self.materials = tuple(materials) + custom_materials
            if not self.triangle_count and not self.custom_capacity:
                raise ValueError("Transport scene must contain geometry")
            from ._custom_resources import prepare_resources

            self.custom_bindings, self.custom_declarations, custom_owners = (
                prepare_resources(self, custom_resources)
            )
            from ._material_resources import prepare_material_resources

            material_bindings, material_declarations, material_owners = (
                prepare_material_resources(
                    self.runtime, [m.program for m in self.materials],
                    material_resources, max(self.custom_bindings, default=15) + 1
                )
            )
            self.custom_bindings.update(material_bindings)
            self.custom_declarations += material_declarations
            custom_owners = tuple(dict.fromkeys((*custom_owners, *material_owners)))
            runtime.retain(self)
            try:
                for owner in custom_owners:
                    owner.retain(self)
                    self._custom_owners += (owner,)
                if self.triangle_count:
                    self._resident = resident or runtime.upload_scene(
                        self._source_scene
                    )
                    if self._resident.runtime is not runtime:
                        raise ValueError(
                            "Resident scene belongs to a different runtime"
                        )
                    self._resident.require_open()
                    if self._resident.scene_revision != self._source_scene.revision:
                        raise ValueError("Resident scene is stale")
                    self._resident._borrowers.add(self)
                    instance_bytes = self._builder._scene_instance_bytes(
                        self._resident.instances
                    )
                    self._borrowed_vertices = self._resident.vertex_buffer
                    self._borrowed_attributes = self._resident.attribute_buffer
                else:
                    instance_bytes = b""
                    self._borrowed_vertices = self._borrowed_attributes = None
                    self._allocate("vertices", np.zeros((1, 4), np.float32))
                    self._allocate("attributes", np.zeros((3, 4), np.float32))
                self._allocate(
                    "triangles",
                    np.asarray(triangle_records or [(0, 0, 0, 0)], np.uint32),
                )
                self._allocate("lights", light_records)
                self._allocate(
                    "materials",
                    np.asarray([m.pack() for m in self.materials], np.float32),
                )
                self._allocate(
                    "media",
                    np.asarray(
                        [(*m.absorption, m.ior) for m in self.media], np.float32
                    ),
                )
                self._allocate(
                    "boundaries",
                    np.asarray(
                        [(b.outside, b.inside, b.identity, 0) for b in self.boundaries]
                        or [(0, 0, 0, 0)],
                        np.uint32,
                    ),
                )
                self._allocate("custom", packed)
                self._triangle_instance_bytes = instance_bytes
                from ._dynamic_scene import build_acceleration

                self._custom_bounds = custom_bounds
                build_acceleration(self)
                self.scene_revision = (
                    None if self._source_scene is None else self._source_scene.revision
                )
            except Exception:
                self.close()
                raise

    def update_custom_geometry_operation(self, updates, *, mode="auto", after=()):
        from ._dynamic_scene import update_operation

        with self.runtime.lock:
            return update_operation(self, updates, mode=mode, after=after)

    def update_custom_geometry(self, updates, *, mode="auto", after=()):
        return self.update_custom_geometry_operation(
            updates, mode=mode, after=after
        ).execute(self.runtime)

    def prepare_custom_geometry_update(self, slots, geometry):
        """Own staging for bulk replacements, or geometry=None for removal.

        Use the returned context manager's operation() in the execution graph.
        """
        from .bulk_updates import VulkanCustomGeometryUpdate

        return VulkanCustomGeometryUpdate(self, slots, geometry)

    def reserve_custom_geometry(self, capacity):
        from ._dynamic_scene import reserve

        with self.runtime.lock:
            return reserve(self, capacity)

    def _boundary_index(self, identity, material):
        if material.kind == "dielectric" and identity is None:
            raise ValueError(
                "Every dielectric surface needs an explicit medium boundary"
            )
        if material.kind != "dielectric" and identity is not None:
            raise ValueError("Medium boundaries require dielectric material")
        if identity is None:
            return 0xFFFFFFFF
        if identity not in self.boundary_indices:
            raise ValueError("Geometry references an unknown boundary")
        return self.boundary_indices[identity]

    def _allocate(self, name, data):
        data = np.ascontiguousarray(data)
        # Traversal randomly reads custom records; keep the large table local
        # to the device rather than making every candidate fetch host memory.
        self._buffers[name] = self.runtime.buffer(
            data.nbytes, data=data, memory="device" if name == "custom" else "host"
        )

    def resource(self, name):
        from ..pipeline.vulkan import VulkanResource

        self.require_open()
        if name == "tlas":
            return VulkanResource(self, "acceleration_structure", self.tlas.handle)
        borrowed = {
            "vertices": self._borrowed_vertices,
            "attributes": self._borrowed_attributes,
        }.get(name)
        buffer = borrowed if borrowed is not None else self._buffers[name]
        return VulkanResource(self, "buffer", buffer.buffer, buffer.size)

    @serialized
    def retain(self, consumer):
        """Lease the TLAS, its BLAS dependencies and resident buffers."""
        self.require_open()
        self._borrowers.add(consumer)

    @serialized
    def release(self, consumer):
        self._borrowers.discard(consumer)

    def require_open(self):
        self.runtime.require_open()
        if self.closed:
            raise RuntimeError("Transport scene is closed")
        if (
            self._source_scene is not None
            and self._source_scene.revision != self.scene_revision
        ):
            raise ValueError("Transport scene changed; upload a replacement snapshot")
        for owner in self._custom_owners:
            owner.require_open()
        if self._resident is not None:
            self._resident.require_open()

    @serialized
    def close(self):
        import vulkan as vk

        if self.closed:
            return
        if self._borrowers or self._gpu_geometry_clients or self._custom_update_clients:
            raise RuntimeError(
                "Close transport integrators and geometry update clients before their scene"
            )
        vk.vkDeviceWaitIdle(self.runtime.device)
        self._builder._release_resources(
            self._builder._structures, self._builder._buffers
        )
        self._builder._structures.clear()
        self._builder._buffers.clear()
        for buffer in self._buffers.values():
            buffer.close()
        if self._resident is not None:
            self._resident._borrowers.discard(self)
            if self._owns_resident:
                self._resident.close()
        for owner in self._custom_owners:
            owner.release(self)
        self._custom_owners = ()
        self.closed = True
        self.runtime.release(self)

    def __enter__(self):
        self.require_open()
        return self

    def __exit__(self, *_exc):
        self.close()
