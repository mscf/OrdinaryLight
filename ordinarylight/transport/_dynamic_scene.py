"""Incremental custom slots: metadata writes, bounds refits, capacity rebuilds."""

from operator import index
import struct
import numpy as np
import vulkan as vk

from ..geometry import CustomGeometry
from ..pipeline.vulkan import VulkanResource, VulkanResourceUse, VulkanPass
from ..pipeline.graph import VulkanOperation

_BUILD = vk.VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR
_READ = vk.VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR
_WRITE = vk.VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR


def build_acceleration(scene):
    """Build only the custom BLAS and combined TLAS, retaining triangle BLASes."""
    builder = scene._builder
    instance_bytes = scene._triangle_instance_bytes
    scene._custom_blas = None
    scene._bounds_buffer = None
    if scene.custom_capacity:
        scene._bounds_buffer = builder._create_uploaded_device_buffer(
            scene._custom_bounds,
            vk.VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR
            | 0x00020000
            | vk.VK_BUFFER_USAGE_TRANSFER_DST_BIT
            | vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            device_address=True,
        )
        scene._custom_shape = vk.VkAccelerationStructureGeometryKHR(
            geometryType=vk.VK_GEOMETRY_TYPE_AABBS_KHR,
            geometry=vk.VkAccelerationStructureGeometryDataKHR(
                aabbs=vk.VkAccelerationStructureGeometryAabbsDataKHR(
                    data=vk.VkDeviceOrHostAddressConstKHR(
                        deviceAddress=builder._buffer_address(scene._bounds_buffer)
                    ),
                    stride=24,
                )
            ),
            flags=0,
        )
        scene._custom_blas = builder._make_as(
            scene._custom_shape,
            scene.custom_capacity,
            vk.VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR,
            allow_update=True,
        )
        instance_bytes += struct.pack(
            "<12fIIQ",
            *np.eye(4, dtype=np.float32)[:3].reshape(-1),
            2 << 24,
            0,
            builder._as_address(scene._custom_blas),
        )
    scene._instance_buffer = builder._create_buffer(
        len(instance_bytes),
        vk.VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR
        | 0x00020000,
        vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
        | vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        data=instance_bytes,
        device_address=True,
    )
    scene._instance_count = len(instance_bytes) // 64
    scene.tlas = builder._make_as(
        builder._tlas_geometry(scene._instance_buffer),
        scene._instance_count,
        vk.VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR,
        allow_update=True,
    )


def _pack(scene, geometry):
    if geometry is None:
        return struct.pack("<12f4I", *([0.0] * 12), 0xFFFFFFFF, 0, 0xFFFFFFFF, 0)
    if not isinstance(geometry, CustomGeometry) or not 0 <= geometry.material < len(
        scene._custom_materials
    ):
        raise ValueError("Custom geometry must reference an existing material")
    if (
        geometry.program.name not in scene.programs
        or scene.programs[geometry.program.name] != geometry.program
    ):
        raise ValueError("New intersection programs require a replacement scene")
    boundary = scene._boundary_index(
        geometry.boundary, scene._custom_materials[geometry.material]
    )
    return struct.pack(
        "<12f4I",
        *geometry.bounds[0],
        0,
        *geometry.bounds[1],
        0,
        *geometry.parameters,
        list(scene.programs).index(geometry.program.name),
        scene.triangle_count + geometry.material,
        boundary,
        geometry.identity,
    )


def _buffer(scene, allocation):
    return VulkanResource(scene, "buffer", allocation.buffer, allocation.size)


def _as(scene, structure):
    return VulkanResource(scene, "acceleration_structure", structure.handle)


def _record_build(scene, structure, geometry, count, kind, mode):
    def record(command):
        build = vk.VkAccelerationStructureBuildGeometryInfoKHR(
            type=kind,
            flags=vk.VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR
            | vk.VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR,
            mode=mode,
            srcAccelerationStructure=structure.handle
            if mode == vk.VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR
            else vk.VK_NULL_HANDLE,
            dstAccelerationStructure=structure.handle,
            geometryCount=1,
            pGeometries=[geometry],
            scratchData=vk.VkDeviceOrHostAddressKHR(
                deviceAddress=scene._builder._buffer_address(structure.scratch)
            ),
        )
        region = vk.VkAccelerationStructureBuildRangeInfoKHR(
            primitiveCount=count, primitiveOffset=0, firstVertex=0, transformOffset=0
        )
        pointers = vk.ffi.new(
            "VkAccelerationStructureBuildRangeInfoKHR*[]", [vk.ffi.addressof(region)]
        )
        scene._builder.build_as(command, 1, [build], pointers)

    return record


def update_operation(scene, updates, *, mode, after):
    scene.require_open()
    if mode not in {"auto", "refit", "rebuild"}:
        raise ValueError("Acceleration mode must be auto, refit, or rebuild")
    updates = {index(slot): geometry for slot, geometry in dict(updates).items()}
    if not updates or any(
        slot < 0 or slot >= scene.custom_capacity for slot in updates
    ):
        raise ValueError(
            "Updates require allocated custom slots; reserve capacity first"
        )
    records = {slot: _pack(scene, geometry) for slot, geometry in updates.items()}
    bounds = {
        slot: np.asarray(geometry.bounds, np.float32).reshape(6)
        for slot, geometry in updates.items()
        if geometry is not None
    }
    binding_revision = scene.binding_revision
    # Conservative: refit for every update carrying geometry; null removal only
    # disables callback metadata, leaving a valid Vulkan AABB in the same slot.
    rebuild_bounds = bool(bounds)

    def upload(command):
        for slot, record in records.items():
            vk.vkCmdUpdateBuffer(
                command,
                scene._buffers["custom"].buffer,
                slot * 64,
                64,
                vk.ffi.new("uint8_t[]", record),
            )
        for slot, value in bounds.items():
            vk.vkCmdUpdateBuffer(
                command,
                scene._bounds_buffer.buffer,
                slot * 24,
                24,
                vk.ffi.new("uint8_t[]", value.tobytes()),
            )

    uses = [
        VulkanResourceUse(
            scene.resource("custom"),
            vk.VK_PIPELINE_STAGE_TRANSFER_BIT,
            vk.VK_ACCESS_TRANSFER_WRITE_BIT,
        )
    ]
    if bounds:
        uses.append(
            VulkanResourceUse(
                _buffer(scene, scene._bounds_buffer),
                vk.VK_PIPELINE_STAGE_TRANSFER_BIT,
                vk.VK_ACCESS_TRANSFER_WRITE_BIT,
            )
        )
    passes = [VulkanPass("custom_slot_updates", tuple(uses), upload)]
    if rebuild_bounds:
        passes.extend(acceleration_passes(scene, mode))

    def validate():
        scene.require_open()
        if scene.binding_revision != binding_revision:
            raise ValueError("Scene bindings changed; recreate acceleration operation")

    def submitted(completion):
        from ._custom_batch import CustomSlots

        slots = scene.custom_geometry
        if not isinstance(slots, CustomSlots):
            slots = CustomSlots(slots)
        slots = slots.updated(updates)
        for slot, value in bounds.items():
            scene._custom_bounds[slot] = value
        scene.custom_geometry = slots
        scene.geometry_revision += 1
        scene.last_completion = completion

    return VulkanOperation(
        passes,
        validate=validate,
        dependencies=lambda: tuple(after)
        + ((scene.last_completion,) if scene.last_completion is not None else ()),
        submitted=submitted,
    )


def reserve(scene, capacity):
    """Grow custom storage, preserving triangle/material/input/history resources.

    Structural allocation changes are synchronized at this explicit boundary.
    Existing integrators rebind; previously compiled graphs must be recreated.
    """
    from ..targets.vulkan.scene import VulkanSceneUploader

    scene.require_open()
    if getattr(scene, "_gpu_geometry_clients", ()):
        raise RuntimeError(
            "Close GPU geometry update clients before growing scene capacity"
        )
    if getattr(scene, "_gpu_geometry_dirty", False):
        from .gpu_geometry import synchronize_geometry_shadow

        synchronize_geometry_shadow(scene)
    capacity = index(capacity)
    if capacity <= scene.custom_capacity:
        return False
    # Build a replacement before retiring the old resources, so ordinary build
    # failures leave the current scene usable.
    from types import SimpleNamespace

    candidate = SimpleNamespace(
        _builder=VulkanSceneUploader(scene.runtime),
        _triangle_instance_bytes=scene._triangle_instance_bytes,
        custom_capacity=capacity,
        _custom_bounds=np.concatenate(
            (
                scene._custom_bounds,
                np.tile(
                    np.array([0, 0, 0, 1e-4, 1e-4, 1e-4], np.float32),
                    (capacity - scene.custom_capacity, 1),
                ),
            )
        ),
    )
    from ._custom_batch import prepare_custom_geometry

    packed, _, slots = prepare_custom_geometry(scene, scene.custom_geometry, capacity)
    data = packed.tobytes()
    buffer = None
    try:
        buffer = scene.runtime.buffer(len(data), data=data)
        build_acceleration(candidate)
    except Exception:
        candidate._builder._release_resources(
            candidate._builder._structures, candidate._builder._buffers
        )
        if buffer is not None:
            buffer.close()
        raise
    vk.vkDeviceWaitIdle(scene.runtime.device)
    for consumer in tuple(scene._borrowers):
        consumer._kernel.close()
        consumer._kernel = None
    previous_builder, previous_buffer = scene._builder, scene._buffers["custom"]
    for name in (
        "_builder",
        "_custom_bounds",
        "_custom_blas",
        "_bounds_buffer",
        "_instance_buffer",
        "_instance_count",
        "tlas",
    ):
        setattr(scene, name, getattr(candidate, name))
    if capacity:
        scene._custom_shape = candidate._custom_shape
    scene._buffers["custom"] = buffer
    scene.custom_capacity, scene.custom_geometry = capacity, slots
    scene.binding_revision += 1
    scene.geometry_revision += 1
    previous_builder._release_resources(
        previous_builder._structures, previous_builder._buffers
    )
    previous_buffer.close()
    for consumer in tuple(scene._borrowers):
        consumer._refresh_scene_bindings()
    return True


def acceleration_passes(scene, mode="refit"):
    passes = []
    build_mode = (
        vk.VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR
        if mode == "rebuild"
        else vk.VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR
    )
    blas = scene._custom_blas
    passes.append(
        VulkanPass(
            "custom_blas_update",
            (
                VulkanResourceUse(_buffer(scene, scene._bounds_buffer), _BUILD, _READ),
                VulkanResourceUse(_as(scene, blas), _BUILD, _READ | _WRITE),
                VulkanResourceUse(_buffer(scene, blas.scratch), _BUILD, _READ | _WRITE),
            ),
            _record_build(
                scene,
                blas,
                scene._custom_shape,
                scene.custom_capacity,
                vk.VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR,
                build_mode,
            ),
        )
    )
    passes.append(
        VulkanPass(
            "custom_tlas_update",
            (
                VulkanResourceUse(_as(scene, blas), _BUILD, _READ),
                VulkanResourceUse(scene.resource("tlas"), _BUILD, _READ | _WRITE),
                VulkanResourceUse(
                    _buffer(scene, scene._instance_buffer), _BUILD, _READ
                ),
                VulkanResourceUse(
                    _buffer(scene, scene.tlas.scratch), _BUILD, _READ | _WRITE
                ),
            ),
            _record_build(
                scene,
                scene.tlas,
                scene._builder._tlas_geometry(scene._instance_buffer),
                scene._instance_count,
                vk.VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR,
                build_mode,
            ),
        )
    )
    return passes
