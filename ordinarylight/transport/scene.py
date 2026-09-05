"""Public triangle/custom-geometry scene for the non-camera transport path."""

from dataclasses import dataclass
import struct

import numpy as np
from ._synchronization import serialized

from .media import OpticalMedium, MediumBoundary


@dataclass(frozen=True)
class TransportMaterial:
    kind: str = "diffuse"
    albedo: tuple[float, float, float] = (0.8, 0.8, 0.8)
    emission: tuple[float, float, float] = (0.0, 0.0, 0.0)
    emission_two_sided: bool = False

    def __post_init__(self):
        if self.kind not in {"diffuse", "dielectric"}:
            raise ValueError(
                "Initial transport materials are diffuse or ideal dielectric"
            )
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
        if material.transmission and material.roughness != 0:
            raise ValueError(
                "Ideal dielectric transport requires roughness=0 or an explicit override"
            )
        if (
            any(
                (
                    material.metallic,
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
            "dielectric" if material.transmission else "diffuse",
            material.base_color,
            material.emission,
            material.emission_two_sided,
        )

    def pack(self):
        return (
            *self.albedo,
            float(self.kind == "dielectric"),
            *self.emission,
            float(self.emission_two_sided),
        )


class VulkanTransportScene:
    """Resident triangles and AABB custom geometry in one hardware TLAS.

    Triangles reuse OrdinaryLight's resident buffers and BLASes. Custom callbacks
    run at ray-query AABB candidates. A snapshot is immutable while consumers
    use it; a new snapshot replaces it after geometry/material edits.
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
    ):
        with runtime.lock:
            import vulkan as vk
            from ..targets.vulkan.scene import (
                VulkanSceneUploader,
                BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
            )

            runtime.require_open()
            self.runtime = runtime
            self.closed = False
            self._borrowers = set()
            self._buffers = {}
            self._custom_owners = ()
            self._resident = None
            self._owns_resident = resident is None
            self._builder = VulkanSceneUploader(runtime)
            self.custom_geometry = tuple(custom_geometry)
            self.media = tuple(media)
            self.boundaries = tuple(boundaries)
            self.programs = {}
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
            custom_materials = tuple(custom_materials)
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
            custom_records = []
            for geometry in self.custom_geometry:
                from ..geometry import CustomGeometry

                if not isinstance(geometry, CustomGeometry) or geometry.material >= len(
                    custom_materials
                ):
                    raise ValueError(
                        "Custom geometry must reference a supplied custom material"
                    )
                material = custom_materials[geometry.material]
                boundary = self._boundary_index(geometry.boundary, material)
                program = geometry.program
                if (
                    program.name in self.programs
                    and self.programs[program.name] != program
                ):
                    raise ValueError("Conflicting custom intersection program names")
                self.programs[program.name] = program
                program_index = list(self.programs).index(program.name)
                custom_records.append((geometry, program_index, boundary))
            self.materials = tuple(materials) + custom_materials
            if not self.triangle_count and not self.custom_geometry:
                raise ValueError("Transport scene must contain geometry")
            from ._custom_resources import prepare_resources

            self.custom_bindings, self.custom_declarations, custom_owners = (
                prepare_resources(self, custom_resources)
            )
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
                custom_dtype = np.dtype(
                    [
                        ("lower", "<f4", (4,)),
                        ("upper", "<f4", (4,)),
                        ("parameters", "<f4", (4,)),
                        ("metadata", "<u4", (4,)),
                    ]
                )
                packed = np.zeros(max(1, len(custom_records)), custom_dtype)
                for index, (geometry, program_index, boundary) in enumerate(
                    custom_records
                ):
                    packed[index]["lower"][:3] = geometry.bounds[0]
                    packed[index]["upper"][:3] = geometry.bounds[1]
                    packed[index]["parameters"] = geometry.parameters
                    packed[index]["metadata"] = (
                        program_index,
                        self.triangle_count + geometry.material,
                        boundary,
                        geometry.identity,
                    )
                self._allocate("custom", packed)
                if custom_records:
                    aabbs = np.asarray(
                        [
                            np.asarray(geometry.bounds).reshape(-1)
                            for geometry, _, _ in custom_records
                        ],
                        np.float32,
                    )
                    bounds_buffer = self._builder._create_uploaded_device_buffer(
                        aabbs,
                        vk.VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR
                        | BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                        device_address=True,
                    )
                    aabb_data = vk.VkAccelerationStructureGeometryAabbsDataKHR(
                        data=vk.VkDeviceOrHostAddressConstKHR(
                            deviceAddress=self._builder._buffer_address(bounds_buffer)
                        ),
                        stride=24,
                    )
                    shape = vk.VkAccelerationStructureGeometryKHR(
                        geometryType=vk.VK_GEOMETRY_TYPE_AABBS_KHR,
                        geometry=vk.VkAccelerationStructureGeometryDataKHR(
                            aabbs=aabb_data
                        ),
                        flags=0,
                    )
                    blas = self._builder._make_as(
                        shape,
                        len(custom_records),
                        vk.VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR,
                    )
                    instance_bytes += struct.pack(
                        "<12fIIQ",
                        *np.eye(4, dtype=np.float32)[:3].reshape(-1),
                        2 << 24,
                        0,
                        self._builder._as_address(blas),
                    )
                instances = self._builder._create_buffer(
                    len(instance_bytes),
                    vk.VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR
                    | BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                    vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
                    | vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                    data=instance_bytes,
                    device_address=True,
                )
                self.tlas = self._builder._make_as(
                    self._builder._tlas_geometry(instances),
                    len(instance_bytes) // 64,
                    vk.VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR,
                )
                self.scene_revision = (
                    None if self._source_scene is None else self._source_scene.revision
                )
            except Exception:
                self.close()
                raise

    def _boundary_index(self, identity, material):
        if material.kind == "dielectric" and identity is None:
            raise ValueError(
                "Every dielectric surface needs an explicit medium boundary"
            )
        if material.kind != "dielectric" and identity is not None:
            raise ValueError("Medium boundaries require ideal dielectric material")
        if identity is None:
            return 0xFFFFFFFF
        if identity not in self.boundary_indices:
            raise ValueError("Geometry references an unknown boundary")
        return self.boundary_indices[identity]

    def _allocate(self, name, data):
        data = np.ascontiguousarray(data)
        self._buffers[name] = self.runtime.buffer(data.nbytes, data=data)

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
        if self._borrowers:
            raise RuntimeError("Close transport integrators before their scene")
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
