"""Staged host-authored custom geometry edits with coalesced buffer copies."""

import numpy as np
import vulkan as vk

from ..geometry import CustomGeometryBatch
from ..pipeline.graph import VulkanOperation
from ..pipeline.vulkan import VulkanPass, VulkanResource, VulkanResourceUse
from ._custom_batch import (
    CUSTOM_DTYPE,
    CustomSlots,
    PackedChanges,
    prepare_custom_geometry,
)
from ._dynamic_scene import _buffer, acceleration_passes


def _ranges(indices, stride, source_offset=0):
    """Contiguous destination runs over densely packed, sorted source rows."""
    starts = np.r_[0, np.flatnonzero(np.diff(indices) != 1) + 1]
    ends = np.r_[starts[1:], len(indices)]
    return tuple(
        (
            int(first) * stride + source_offset,
            int(indices[first]) * stride,
            int(end - first) * stride,
        )
        for first, end in zip(starts, ends)
    )


def _plan(scene, slots, geometry):
    raw = np.asarray(slots)
    if (
        raw.ndim != 1
        or not len(raw)
        or raw.dtype.kind not in "iu"
        or np.any(raw < 0)
        or np.any(raw >= scene.custom_capacity)
    ):
        raise ValueError(
            "Slots must be a nonempty integer array within reserved capacity"
        )
    order = np.argsort(raw)
    indices = raw[order].astype(np.int64)
    if np.any(np.diff(indices) == 0):
        raise ValueError("Duplicate update slots are not allowed")
    if geometry is None:
        records = np.zeros(len(indices), CUSTOM_DTYPE)
        records["metadata"][:, 0] = 0xFFFFFFFF
        records["metadata"][:, 2] = 0xFFFFFFFF
        bounds = None
    else:
        if not isinstance(geometry, CustomGeometryBatch):
            raise TypeError(
                "Bulk updates require CustomGeometryBatch or None for removal"
            )
        if len(geometry) != len(indices):
            raise ValueError("Supply one geometry row per update slot")
        if scene.programs.get(geometry.program.name) != geometry.program:
            raise ValueError("New intersection programs require a replacement scene")
        records, bounds, _ = prepare_custom_geometry(scene, geometry, len(geometry))
        records, bounds = records[order], bounds[order]
    records.flags.writeable = indices.flags.writeable = False
    if bounds is not None:
        bounds.flags.writeable = False
    change = PackedChanges.for_scene(scene, indices, records)
    payload = records.tobytes() + (b"" if bounds is None else bounds.tobytes())
    return (
        change,
        bounds,
        payload,
        _ranges(indices, 64),
        (() if bounds is None else _ranges(indices, 24, records.nbytes)),
    )


class VulkanCustomGeometryUpdate:
    """Owned reusable staging for slot-indexed CustomGeometryBatch edits.

    Pass geometry=None to disable all selected slots. The payload is a snapshot;
    close this client after its last use to release staging memory. Operations
    copy only selected records/bounds and refit or rebuild existing acceleration.
    They do not resize the scene or reset application lighting history.
    """

    def __init__(self, scene, slots, geometry):
        self.scene, self.runtime = scene, scene.runtime
        self.closed = False
        self.last_completion = None
        self._staging = None
        with self.runtime.lock:
            scene.require_open()
            (
                self._change,
                self._bounds,
                payload,
                self._record_ranges,
                self._bound_ranges,
            ) = _plan(scene, slots, geometry)
            self.binding_revision = scene.binding_revision
            # Only transfer-source usage is needed; stage once, reuse unchanged.
            self._staging = self.runtime.buffer(
                len(payload), data=payload, usage=vk.VK_BUFFER_USAGE_TRANSFER_SRC_BIT
            )
            scene._custom_update_clients.add(self)

    def require_open(self):
        self.scene.require_open()
        if self.closed:
            raise RuntimeError("Custom geometry update is closed")
        if self.scene.binding_revision != self.binding_revision:
            raise ValueError(
                "Scene bindings changed; recreate the custom geometry update"
            )
        self._staging.require_open()

    def operation(self, *, mode="auto", after=()):
        with self.runtime.lock:
            self.require_open()
            if mode not in {"auto", "refit", "rebuild"}:
                raise ValueError("Acceleration mode must be auto, refit, or rebuild")
            after = tuple(after)
            source = VulkanResource.buffer(self._staging)
            uses = [
                VulkanResourceUse(
                    source,
                    vk.VK_PIPELINE_STAGE_TRANSFER_BIT,
                    vk.VK_ACCESS_TRANSFER_READ_BIT,
                ),
                VulkanResourceUse(
                    self.scene.resource("custom"),
                    vk.VK_PIPELINE_STAGE_TRANSFER_BIT,
                    vk.VK_ACCESS_TRANSFER_WRITE_BIT,
                ),
            ]
            if self._bounds is not None:
                uses.append(
                    VulkanResourceUse(
                        _buffer(self.scene, self.scene._bounds_buffer),
                        vk.VK_PIPELINE_STAGE_TRANSFER_BIT,
                        vk.VK_ACCESS_TRANSFER_WRITE_BIT,
                    )
                )

            def copy(command):
                destinations = [
                    (self.scene._buffers["custom"].buffer, self._record_ranges)
                ]
                if self._bounds is not None:
                    destinations.append(
                        (self.scene._bounds_buffer.buffer, self._bound_ranges)
                    )
                for destination, ranges in destinations:
                    # Bound host FFI allocation for very fragmented updates.
                    for first in range(0, len(ranges), 4096):
                        regions = [
                            vk.VkBufferCopy(srcOffset=src, dstOffset=dst, size=size)
                            for src, dst, size in ranges[first : first + 4096]
                        ]
                        vk.vkCmdCopyBuffer(
                            command,
                            self._staging.buffer,
                            destination,
                            len(regions),
                            regions,
                        )

            passes = [VulkanPass("bulk_custom_slot_updates", tuple(uses), copy)]
            if self._bounds is not None:
                passes.extend(acceleration_passes(self.scene, mode))

            def submitted(completion):
                slots = self.scene.custom_geometry
                if not isinstance(slots, CustomSlots):
                    slots = CustomSlots(slots)
                self.scene.custom_geometry = slots.bulk_updated(self._change)
                if self._bounds is not None:
                    self.scene._custom_bounds[self._change.indices] = self._bounds
                self.scene.geometry_revision += 1
                self.last_completion = self.scene.last_completion = completion

            return VulkanOperation(
                passes,
                validate=self.require_open,
                submitted=submitted,
                dependencies=lambda: after
                + (
                    (self.scene.last_completion,)
                    if self.scene.last_completion is not None
                    else ()
                ),
            )

    def close(self):
        with self.runtime.lock:
            if self.closed:
                return
            if self.last_completion is not None:
                self.last_completion.wait()
            self._staging.close()
            self.scene._custom_update_clients.discard(self)
            self.closed = True

    def __enter__(self):
        self.require_open()
        return self

    def __exit__(self, *_exc):
        self.close()
