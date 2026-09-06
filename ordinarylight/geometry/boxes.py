"""Resource-backed, grouped axis-aligned boxes for custom-intersection clients."""

from importlib.resources import files
from operator import index

import numpy as np

from .intersection import CustomGeometry, IntersectionProgram
from .resources import IntersectionResource


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
            source.replace("OL_BOX_ENTRY", entry).replace(
                "OL_BOX_RESOURCE", resource_name
            ),
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
        return CustomGeometry(
            (records[:, 0, :3].min(axis=0), records[:, 1, :3].max(axis=0)),
            self.program,
            (first, count, 0, 0),
        )
