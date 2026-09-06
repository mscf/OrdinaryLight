"""CPU custom-geometry packing shared by scene construction and capacity growth."""

from collections.abc import Sequence
from operator import index

import numpy as np

from ..geometry import CustomGeometry, CustomGeometryBatch

CUSTOM_DTYPE = np.dtype(
    [
        ("lower", "<f4", (4,)),
        ("upper", "<f4", (4,)),
        ("parameters", "<f4", (4,)),
        ("metadata", "<u4", (4,)),
    ]
)


class CustomSlots(Sequence):
    """Lazy batch plus spare slots and sparse host edits."""

    def __init__(self, base, capacity=None, updates=None, bulk=None):
        self.base = base if isinstance(base, CustomGeometryBatch) else tuple(base)
        self.capacity = len(self.base) if capacity is None else capacity
        if self.capacity < len(self.base):
            raise ValueError("Custom capacity must cover supplied geometry")
        self.updates = dict(updates or {})
        self.bulk = bulk

    def __len__(self):
        return self.capacity

    def __getitem__(self, slot):
        if isinstance(slot, slice):
            return tuple(self[i] for i in range(*slot.indices(len(self))))
        slot = index(slot)
        if slot < 0:
            slot += len(self)
        if not 0 <= slot < len(self):
            raise IndexError(slot)
        if slot in self.updates:
            return self.updates[slot]
        if self.bulk is not None:
            row = np.searchsorted(self.bulk.indices, slot)
            if row < len(self.bulk.indices) and self.bulk.indices[row] == slot:
                return self.bulk.geometry(row)
        return self.base[slot] if slot < len(self.base) else None

    def updated(self, updates):
        return CustomSlots(
            self.base, self.capacity, {**self.updates, **updates}, self.bulk
        )

    def resized(self, capacity):
        return CustomSlots(self.base, capacity, self.updates, self.bulk)

    def bulk_updated(self, change):
        # Latest writes win without retaining a growing list of frame snapshots.
        bulk = change if self.bulk is None else self.bulk.merged(change)
        updates = self.updates.copy()
        if updates:
            for slot in np.intersect1d(list(updates), change.indices):
                del updates[int(slot)]
        return CustomSlots(self.base, self.capacity, updates, bulk)


class PackedChanges:
    """Sparse scene-indexed records with lazy declaration reconstruction."""

    def __init__(self, indices, records, programs, boundaries, triangles):
        self.indices = indices
        self.records = records
        self.programs = programs
        self.boundaries = boundaries
        self.triangles = triangles

    @classmethod
    def for_scene(cls, scene, indices, records):
        return cls(
            indices,
            records,
            tuple(scene.programs.values()),
            tuple(b.identity for b in scene.boundaries),
            scene.triangle_count,
        )

    def merged(self, change):
        if np.array_equal(self.indices, change.indices):
            return change
        keep = ~np.isin(self.indices, change.indices, assume_unique=True)
        indices = np.concatenate((self.indices[keep], change.indices))
        records = np.concatenate((self.records[keep], change.records))
        order = np.argsort(indices)
        return PackedChanges(
            indices[order],
            records[order],
            change.programs,
            change.boundaries,
            change.triangles,
        )

    def geometry(self, row):
        record = self.records[row]
        program, material, boundary, identity = map(int, record["metadata"])
        if program == 0xFFFFFFFF:
            return None
        return CustomGeometry(
            (record["lower"][:3], record["upper"][:3]),
            self.programs[program],
            record["parameters"],
            material=material - self.triangles,
            boundary=None if boundary == 0xFFFFFFFF else self.boundaries[boundary],
            identity=identity,
        )


def prepare_custom_geometry(scene, geometry, capacity):
    """Validate, register programs, and pack without expanding array batches."""
    slots = (
        geometry.resized(capacity)
        if isinstance(geometry, CustomSlots)
        else CustomSlots(geometry, capacity)
    )
    packed = np.zeros(max(1, capacity), CUSTOM_DTYPE)
    packed["metadata"][:, 0] = 0xFFFFFFFF
    bounds = np.tile(np.array([0, 0, 0, 1e-4, 1e-4, 1e-4], np.float32), (capacity, 1))

    def program_index(program):
        if program.name in scene.programs and scene.programs[program.name] != program:
            raise ValueError("Conflicting custom intersection program names")
        scene.programs[program.name] = program
        return list(scene.programs).index(program.name)

    def write(slot, item):
        if item is None:
            packed[slot] = np.zeros((), CUSTOM_DTYPE)
            packed["metadata"][slot] = (0xFFFFFFFF, 0, 0xFFFFFFFF, 0)
            return
        if not isinstance(item, CustomGeometry) or not 0 <= item.material < len(
            scene._custom_materials
        ):
            raise ValueError(
                "Custom geometry must reference a supplied custom material"
            )
        boundary = scene._boundary_index(
            item.boundary, scene._custom_materials[item.material]
        )
        packed["lower"][slot, :3], packed["upper"][slot, :3] = item.bounds
        packed["parameters"][slot] = item.parameters
        packed["metadata"][slot] = (
            program_index(item.program),
            scene.triangle_count + item.material,
            boundary,
            item.identity,
        )
        bounds[slot] = np.asarray(item.bounds, np.float32).reshape(6)

    base = slots.base
    if isinstance(base, CustomGeometryBatch):
        count = len(base)
        if np.any(base.materials >= len(scene._custom_materials)):
            raise ValueError(
                "Custom geometry must reference a supplied custom material"
            )
        dielectric = np.array(
            [m.kind == "dielectric" for m in scene._custom_materials]
        )[base.materials]
        has_boundary = base.boundaries != 0xFFFFFFFF
        if np.any(dielectric & ~has_boundary):
            raise ValueError(
                "Every dielectric surface needs an explicit medium boundary"
            )
        if np.any(~dielectric & has_boundary):
            raise ValueError("Medium boundaries require dielectric material")
        unique, inverse = np.unique(base.boundaries, return_inverse=True)
        if any(
            int(b) != 0xFFFFFFFF and int(b) not in scene.boundary_indices
            for b in unique
        ):
            raise ValueError("Geometry references an unknown boundary")
        mapped = np.array(
            [
                0xFFFFFFFF if b == 0xFFFFFFFF else scene.boundary_indices[int(b)]
                for b in unique
            ],
            np.uint32,
        )
        packed["lower"][:count, :3] = base.bounds[:, 0]
        packed["upper"][:count, :3] = base.bounds[:, 1]
        packed["parameters"][:count] = base.parameters
        packed["metadata"][:count, 0] = program_index(base.program)
        packed["metadata"][:count, 1] = base.materials + scene.triangle_count
        packed["metadata"][:count, 2] = mapped[inverse]
        packed["metadata"][:count, 3] = base.identities
        bounds[:count] = base.bounds.reshape(count, 6)
    else:
        for slot, item in enumerate(base):
            write(slot, item)
    if slots.bulk is not None:
        change = slots.bulk
        # A shadow can also be used to construct another scene, whose program,
        # triangle and boundary indices need not match the originating scene.
        packed[change.indices] = change.records
        for program in np.unique(change.records["metadata"][:, 0]):
            if program == 0xFFFFFFFF:
                continue
            mask = change.records["metadata"][:, 0] == program
            rows = change.records[mask]
            boundary_indices = rows["metadata"][:, 2]
            boundary_ids = np.full(len(rows), 0xFFFFFFFF, np.uint32)
            has_boundary = boundary_indices != 0xFFFFFFFF
            boundary_ids[has_boundary] = np.asarray(change.boundaries, np.uint32)[
                boundary_indices[has_boundary]
            ]
            declarations = CustomGeometryBatch(
                np.stack((rows["lower"][:, :3], rows["upper"][:, :3]), axis=1),
                change.programs[int(program)],
                rows["parameters"],
                materials=rows["metadata"][:, 1] - change.triangles,
                boundaries=boundary_ids,
                identities=rows["metadata"][:, 3],
            )
            remapped, remapped_bounds, _ = prepare_custom_geometry(
                scene, declarations, len(declarations)
            )
            packed[change.indices[mask]] = remapped
            bounds[change.indices[mask]] = remapped_bounds
    for slot, item in slots.updates.items():
        write(slot, item)
    return packed, bounds, slots
