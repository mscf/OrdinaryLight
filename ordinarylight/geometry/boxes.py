"""Resource-backed, grouped axis-aligned boxes for custom-intersection clients."""

from importlib.resources import files
from operator import index

import numpy as np

from .intersection import CustomGeometry, IntersectionProgram
from .resources import IntersectionResource


def _validated_bounds(bounds):
    bounds = np.array(bounds, dtype=np.float32, copy=True)
    if (
        bounds.ndim != 3
        or bounds.shape[1:] != (2, 3)
        or not 1 <= len(bounds) <= 16_777_216
        or not np.isfinite(bounds).all()
        or np.any(bounds[:, 0] >= bounds[:, 1])
    ):
        raise ValueError(
            "Box bounds must be finite nonempty (N, 2, 3) with positive extents"
        )
    return bounds


class BoxBatch:
    """Pack independent world-space boxes; group them without changing hit IDs.

    Records are three vec4s: lower.xyz/active, upper.xyz/reserved, and bitcast
    uvec4(material, boundary application ID, application identity, reserved).
    This is a bounded linear-scan reference, not a union or grid traversal.
    """

    def __init__(
        self,
        bounds,
        *,
        materials=0,
        boundaries=None,
        identities=None,
        active=True,
        resource_name="boxes",
    ):
        bounds = _validated_bounds(bounds)
        count = len(bounds)

        def ids(value, label):
            raw = np.asarray(value)
            if (
                raw.dtype.kind not in "iu"
                or np.any(raw < 0)
                or np.any(raw > 0xFFFFFFFF)
            ):
                raise ValueError(f"{label} must contain uint32 values")
            try:
                return np.broadcast_to(raw, (count,)).astype(np.uint32)
            except ValueError as exc:
                raise ValueError(
                    f"{label} must be scalar or have one value per box"
                ) from exc

        resource = IntersectionResource(resource_name)
        self.records = np.zeros((count, 3, 4), np.float32)
        self.records[:, :2, :3] = bounds
        enabled = np.asarray(active)
        if enabled.dtype.kind != "b":
            raise ValueError("active must contain booleans")
        self.records[:, 0, 3] = np.broadcast_to(enabled, (count,))
        metadata = self.records[:, 2].view(np.uint32)
        metadata[:, 0] = ids(materials, "materials")
        metadata[:, 1] = ids(
            0xFFFFFFFF if boundaries is None else boundaries, "boundaries"
        )
        metadata[:, 2] = ids(
            np.arange(count, dtype=np.uint32) if identities is None else identities,
            "identities",
        )
        self.records.flags.writeable = False
        self.resource_name = resource_name
        entry = f"boxBatch_{resource_name}"
        source = (
            files("ordinarylight.shaders")
            .joinpath("transport_v1/box_batch.glsl")
            .read_text()
        )
        self.program = IntersectionProgram(
            entry,
            source.replace("OL_BOX_ENTRY", entry)
            .replace("OL_BOX_RESOURCE", resource_name)
            .replace("OL_BOX_INDEX_COUNT", f"uint({resource_name}.length())/3u")
            .replace("OL_BOX_INDEX", "slot"),
            resources=(resource,),
            hit_version=2,
        )

    def geometry(self, first=0, count=None):
        """Declare one enclosing custom primitive for a contiguous record range.

        Bounds include inactive records so activation needs no acceleration
        update. Moving boxes beyond those bounds requires an explicit update.
        Bind records as scene custom_resources[resource_name].
        """
        first = index(first)
        count = len(self.records) - first if count is None else index(count)
        if first < 0 or count < 1 or first + count > len(self.records):
            raise ValueError("Box group must be a nonempty in-range record interval")
        records = self.records[first : first + count]
        material, boundary, identity, _ = map(int, records[0, 2].view(np.uint32))
        return CustomGeometry(
            (records[:, 0, :3].min(axis=0), records[:, 1, :3].max(axis=0)),
            self.program,
            (first, count, 0, 0),
            material=material,
            boundary=None if boundary == 0xFFFFFFFF else boundary,
            identity=identity,
        )

    def partition(self, max_boxes=8):
        """Spatially partition records while retaining their original buffer slots."""
        return BoxPartition(self, max_boxes=max_boxes)


class BoxPartition:
    """Host-built median partitions accelerated as separate Vulkan primitives.

    Upload indices alongside batch.records. Both allocations remain fixed for
    activation/material edits. Moving boxes requires refitting group bounds or
    rebuilding this partition; GPU indices are not automatically regenerated.
    """

    def __init__(self, batch, *, max_boxes=8):
        if not isinstance(batch, BoxBatch):
            raise TypeError("Expected BoxBatch")
        max_boxes = index(max_boxes)
        if max_boxes < 1:
            raise ValueError("max_boxes must be positive")
        self.batch = batch
        self.index_resource_name = batch.resource_name + "_indices"
        resource = IntersectionResource(self.index_resource_name, element_type="uint")
        centers = batch.records[:, :2, :3].astype(np.float64).mean(axis=1)
        leaves = []

        def split(indices):
            if len(indices) <= max_boxes:
                leaves.append(indices)
                return
            points = centers[indices]
            axis = int(np.argmax(np.ptp(points, axis=0)))
            ordered = indices[np.argsort(points[:, axis], kind="stable")]
            middle = len(ordered) // 2
            split(ordered[:middle])
            split(ordered[middle:])

        split(np.arange(len(batch.records), dtype=np.uint32))
        self.indices = np.concatenate(leaves)
        self.indices.flags.writeable = False
        entry = f"partitionedBoxBatch_{batch.resource_name}"
        source = (
            files("ordinarylight.shaders")
            .joinpath("transport_v1/box_batch.glsl")
            .read_text()
        )
        source = source.replace("OL_BOX_ENTRY", entry).replace(
            "OL_BOX_RESOURCE", batch.resource_name
        )
        source = source.replace(
            "OL_BOX_INDEX_COUNT", f"uint({self.index_resource_name}.length())"
        )
        source = source.replace("OL_BOX_INDEX", f"{self.index_resource_name}[slot]")
        self.program = IntersectionProgram(
            entry, source, resources=(*batch.program.resources, resource), hit_version=2
        )
        geometry = []
        first = 0
        for leaf in leaves:
            records = batch.records[leaf]
            material, boundary, identity, _ = map(int, records[0, 2].view(np.uint32))
            geometry.append(
                CustomGeometry(
                    (records[:, 0, :3].min(axis=0), records[:, 1, :3].max(axis=0)),
                    self.program,
                    (first, len(leaf), 0, 0),
                    material=material,
                    boundary=None if boundary == 0xFFFFFFFF else boundary,
                    identity=identity,
                )
            )
            first += len(leaf)
        self.geometries = tuple(geometry)

    def refit(self, bounds):
        """Return replacement group geometries for moved boxes in the same slots.

        This is a host bounds calculation. Upload updated records and apply the
        returned geometries with scene.update_custom_geometry (or its operation)
        before tracing. The immutable partition and its index order do not change.
        """
        bounds = _validated_bounds(bounds)
        if len(bounds) != len(self.indices):
            raise ValueError("Refit must preserve the number of box slots")
        result = []
        for geometry in self.geometries:
            first, count = map(int, geometry.parameters[:2])
            selected = bounds[self.indices[first : first + count]]
            result.append(
                CustomGeometry(
                    (selected[:, 0].min(axis=0), selected[:, 1].max(axis=0)),
                    self.program,
                    geometry.parameters,
                    material=geometry.material,
                    boundary=geometry.boundary,
                    identity=geometry.identity,
                )
            )
        return tuple(result)
