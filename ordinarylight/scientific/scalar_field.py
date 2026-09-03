"""Coordinate-aware three-dimensional scalar fields."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..scene import Transform
from .transfer import TransferFunction


@dataclass(frozen=True, slots=True)
class ProbeResult:
    world_position: tuple[float, float, float]
    index_position: tuple[float, float, float]
    nearest_index: tuple[int, int, int]
    value: float
    valid: bool
    unit: str | None


@dataclass(frozen=True, slots=True)
class SampleResult:
    """A trilinearly interpolated physical field sample."""

    world_position: tuple[float, float, float]
    index_position: tuple[float, float, float]
    value: float
    valid: bool
    unit: str | None


@dataclass(frozen=True, slots=True)
class RayProbeResult:
    """The first transfer-visible sample selected along a world-space ray."""

    world_position: tuple[float, float, float]
    index_position: tuple[float, float, float]
    value: float
    normalized_value: float
    rgba: tuple[float, float, float, float]
    distance: float
    accumulated_opacity: float
    unit: str | None


@dataclass(slots=True)
class ScalarField3D:
    """A NumPy ``(z, y, x)`` field with explicit voxel-center coordinates."""

    data: np.ndarray = field(repr=False)
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0)
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0)
    direction: np.ndarray = field(default_factory=lambda: np.eye(3), repr=False)
    unit: str | None = None
    name: str | None = None
    metadata: dict = field(default_factory=dict, repr=False)
    revision: int = field(default=0, init=False)
    _updates: list = field(default_factory=list, init=False, repr=False)

    def __post_init__(self):
        data = np.asarray(self.data)
        if data.ndim != 3 or any(size < 2 for size in data.shape):
            raise ValueError("data must have shape (z, y, x) with dimensions >= 2")
        if not np.issubdtype(data.dtype, np.number):
            raise TypeError("data must have a numeric dtype")
        spacing = np.asarray(self.spacing, dtype=np.float64)
        origin = np.asarray(self.origin, dtype=np.float64)
        direction = np.asarray(self.direction, dtype=np.float64)
        if spacing.shape != (3,) or not np.isfinite(spacing).all() or np.any(spacing <= 0):
            raise ValueError("spacing must contain three positive finite values")
        if origin.shape != (3,) or not np.isfinite(origin).all():
            raise ValueError("origin must contain three finite values")
        if direction.shape != (3, 3) or not np.isfinite(direction).all():
            raise ValueError("direction must be a finite 3x3 matrix")
        if not np.allclose(direction.T @ direction, np.eye(3), atol=1e-6):
            raise ValueError("direction must be orthonormal")
        if abs(float(np.linalg.det(direction))) < 1e-8:
            raise ValueError("direction must be invertible")
        if self.unit is not None and not isinstance(self.unit, str):
            raise TypeError("unit must be a string or None")
        if not hasattr(self.metadata, "items"):
            raise TypeError("metadata must be a mapping")
        self.data = data
        self.spacing = tuple(map(float, spacing))
        self.origin = tuple(map(float, origin))
        self.direction = direction
        self.metadata = dict(self.metadata)

    @property
    def index_to_world_matrix(self):
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, :3] = self.direction @ np.diag(self.spacing)
        matrix[:3, 3] = self.origin
        return matrix

    def index_to_world(self, index_xyz):
        index = np.asarray(index_xyz, dtype=np.float64)
        return np.asarray(index @ self.index_to_world_matrix[:3, :3].T + self.origin)

    def world_to_index(self, world_position):
        position = np.asarray(world_position, dtype=np.float64)
        if position.shape[-1:] != (3,) or not np.isfinite(position).all():
            raise ValueError("world_position must end in three finite coordinates")
        inverse = np.linalg.inv(self.index_to_world_matrix)
        return np.asarray(position @ inverse[:3, :3].T + inverse[:3, 3])

    def probe(self, world_position):
        """Return the exact nearest stored sample and both coordinate systems."""
        index_xyz = self.world_to_index(world_position)
        nearest_xyz = np.floor(index_xyz + 0.5).astype(np.int64)
        shape_xyz = np.asarray(self.data.shape[::-1])
        inside = bool(np.all(nearest_xyz >= 0) and np.all(nearest_xyz < shape_xyz))
        value = np.nan if not inside else float(self.data[tuple(nearest_xyz[::-1])])
        return ProbeResult(
            tuple(map(float, world_position)), tuple(map(float, index_xyz)),
            tuple(map(int, nearest_xyz)), value, inside and bool(np.isfinite(value)),
            self.unit,
        )

    def sample(self, world_position):
        """Trilinearly interpolate the physical field at a world position.

        Samples outside the voxel-center bounds, or interpolation cells that
        contain any non-finite corner, are returned as invalid.
        """
        index = np.asarray(self.world_to_index(world_position), np.float64)
        shape_xyz = np.asarray(self.data.shape[::-1], np.int64)
        inside = bool(np.all(index >= 0.0) and np.all(index <= shape_xyz - 1))
        if not inside:
            return SampleResult(
                tuple(map(float, world_position)), tuple(map(float, index)),
                float("nan"), False, self.unit,
            )
        lower = np.floor(index).astype(np.int64)
        upper = np.minimum(lower + 1, shape_xyz - 1)
        weight = index - lower
        x0, y0, z0 = lower
        x1, y1, z1 = upper
        corners = np.asarray((
            self.data[z0, y0, x0], self.data[z0, y0, x1],
            self.data[z0, y1, x0], self.data[z0, y1, x1],
            self.data[z1, y0, x0], self.data[z1, y0, x1],
            self.data[z1, y1, x0], self.data[z1, y1, x1],
        ), np.float64)
        valid = bool(np.isfinite(corners).all())
        if valid:
            wx, wy, wz = weight
            c00 = corners[0] * (1 - wx) + corners[1] * wx
            c10 = corners[2] * (1 - wx) + corners[3] * wx
            c01 = corners[4] * (1 - wx) + corners[5] * wx
            c11 = corners[6] * (1 - wx) + corners[7] * wx
            value = ((c00 * (1 - wy) + c10 * wy) * (1 - wz)
                     + (c01 * (1 - wy) + c11 * wy) * wz)
        else:
            value = float("nan")
        return SampleResult(
            tuple(map(float, world_position)), tuple(map(float, index)),
            float(value), valid, self.unit,
        )

    def probe_ray(
        self, origin, direction, transfer_function, *, step_size=None,
        opacity_threshold=0.01, reference_step_size=None, max_steps=16384,
        clipping=None,
    ):
        """Return the first transfer-visible trilinear sample along a ray.

        ``opacity_threshold`` selects by front-to-back accumulated opacity,
        making transparent leading samples pass through. Alpha is adjusted for
        step length using the same reference-opacity equation as volume
        integration.
        """
        if not isinstance(transfer_function, TransferFunction):
            raise TypeError("transfer_function must be a TransferFunction")
        if clipping is not None:
            from .clipping import ClipRegion
            if not isinstance(clipping, ClipRegion):
                raise TypeError("clipping must be a ClipRegion or None")
        origin = np.asarray(origin, np.float64)
        direction = np.asarray(direction, np.float64)
        if origin.shape != (3,) or direction.shape != (3,) or not (
            np.isfinite(origin).all() and np.isfinite(direction).all()
        ):
            raise ValueError("origin and direction must be finite three-vectors")
        length = float(np.linalg.norm(direction))
        if length <= 1e-12:
            raise ValueError("direction must be nonzero")
        direction /= length
        if step_size is None:
            step_size = min(self.spacing) * 0.5
        step_size = float(step_size)
        reference_step_size = step_size if reference_step_size is None else float(reference_step_size)
        opacity_threshold = float(opacity_threshold)
        max_steps = int(max_steps)
        if not np.isfinite(step_size) or step_size <= 0.0:
            raise ValueError("step_size must be positive and finite")
        if not np.isfinite(reference_step_size) or reference_step_size <= 0.0:
            raise ValueError("reference_step_size must be positive and finite")
        if not 0.0 <= opacity_threshold <= 1.0:
            raise ValueError("opacity_threshold must be in [0, 1]")
        if max_steps < 1:
            raise ValueError("max_steps must be positive")

        inverse = np.linalg.inv(self.index_to_world_matrix)
        index_origin = origin @ inverse[:3, :3].T + inverse[:3, 3]
        index_direction = direction @ inverse[:3, :3].T
        upper_bound = np.asarray(self.data.shape[::-1]) - 1
        parallel = np.abs(index_direction) <= 1e-14
        if np.any(parallel & ((index_origin < 0.0) | (index_origin > upper_bound))):
            return None
        safe = np.where(parallel, 1.0, index_direction)
        bounds_a = np.where(parallel, -np.inf, -index_origin / safe)
        bounds_b = np.where(
            parallel, np.inf, (upper_bound - index_origin) / safe,
        )
        entry = max(float(np.max(np.minimum(bounds_a, bounds_b))), 0.0)
        exit_distance = float(np.min(np.maximum(bounds_a, bounds_b)))
        if exit_distance < entry:
            return None
        count = min(max_steps, max(1, int(np.ceil((exit_distance - entry) / step_size))))
        dt = (exit_distance - entry) / count if exit_distance > entry else step_size
        transmittance = 1.0
        last_visible = None
        for step in range(count):
            distance = entry + (step + 0.5) * dt
            world = origin + direction * distance
            sample = self.sample(world)
            if not sample.valid or (
                clipping is not None
                and not clipping.contains(self, world, sample.index_position)
            ):
                continue
            normalized, valid = transfer_function.map(np.asarray(sample.value))
            if not bool(valid):
                continue
            normalized_value = float(normalized)
            rgba = np.asarray(transfer_function.colors(np.asarray(sample.value)), np.float64)
            reference_alpha = float(np.clip(rgba[3], 0.0, 1.0 - 1e-7))
            alpha = 1.0 - (1.0 - reference_alpha) ** (dt / reference_step_size)
            accumulated = 1.0 - transmittance * (1.0 - alpha)
            if alpha > 0.0:
                last_visible = RayProbeResult(
                    tuple(map(float, world)), sample.index_position, sample.value,
                    normalized_value, tuple(map(float, rgba)), float(distance),
                    float(accumulated), self.unit,
                )
            if accumulated >= opacity_threshold and last_visible is not None:
                return last_visible
            transmittance *= 1.0 - alpha
        return last_visible if opacity_threshold == 0.0 else None

    def probe_pixel(
        self, camera, viewport_size, pixel, transfer_function, *, mapping=None,
        **probe_options,
    ):
        """Probe beneath a UI pixel, including viewport/DPI render mapping."""
        from ..selection import ViewportMapping, camera_ray
        if mapping is not None:
            if not isinstance(mapping, ViewportMapping):
                raise TypeError("mapping must be a ViewportMapping")
            pixel = mapping.map_pixel(pixel)
            if pixel is None:
                return None
            viewport_size = mapping.render_size
        origin, direction = camera_ray(camera, viewport_size, pixel)
        return self.probe_ray(
            origin, direction, transfer_function, **probe_options,
        )

    def volume_transform(self):
        """Transform the renderer unit cube onto the field's voxel centers."""
        shape_minus_one = np.asarray(self.data.shape[::-1], dtype=np.float64) - 1.0
        matrix = self.index_to_world_matrix.copy()
        matrix[:3, :3] = matrix[:3, :3] @ np.diag(shape_minus_one)
        return Transform(matrix)

    def add_volume(
        self, scene, transfer_function: TransferFunction, *, clipping=None,
        **material_options,
    ):
        """Adapt this field to the current volume renderer with shared mapping."""
        if not isinstance(transfer_function, TransferFunction):
            raise TypeError("transfer_function must be a scientific TransferFunction")
        normalized, valid = transfer_function.encode_volume(self.data)
        metadata = dict(self.metadata)
        metadata["scientific"] = self.snapshot(transfer_function)
        metadata["valid_sample_count"] = int(np.count_nonzero(valid))
        clip_planes = ()
        if clipping is not None:
            from .clipping import ClipRegion
            if not isinstance(clipping, ClipRegion):
                raise TypeError("clipping must be a ClipRegion or None")
            clip_planes = tuple(
                (*plane.normal, plane.offset)
                for plane in clipping.world_planes(self)
            )
            metadata["scientific"]["clipping"] = clipping.snapshot()
        return scene.add_volume(
            normalized, transfer_function.volume_material(**material_options),
            transform=self.volume_transform(), value_range=(0.0, 1.0),
            name=self.name, metadata=metadata, clip_planes=clip_planes,
        )

    def slice(self, axis, index, transfer_function, *, clipping=None):
        """Extract one coordinate-aware orthogonal slice."""
        from .slice import ScalarSlice
        return ScalarSlice(self, axis, index, transfer_function, clipping)

    def isosurface(self, value, transfer_function, *, clipping=None):
        """Extract a deterministic isosurface in the field's coordinates."""
        from .isosurface import ScalarIsosurface
        return ScalarIsosurface.from_field(
            self, value, transfer_function, clipping=clipping,
        )

    def update(self, offset, values):
        """Update a z/y/x region and return its new monotonic revision."""
        if len(offset) != 3:
            raise ValueError("offset must contain z, y, x")
        offset = tuple(int(value) for value in offset)
        if any(value < 0 for value in offset):
            raise ValueError("offset values cannot be negative")
        values = np.asarray(values)
        if values.ndim != 3 or any(size < 1 for size in values.shape):
            raise ValueError("values must be a nonempty three-dimensional array")
        stop = tuple(start + size for start, size in zip(offset, values.shape))
        if any(end > size for end, size in zip(stop, self.data.shape)):
            raise ValueError("updated region lies outside the field")
        if not self.data.flags.writeable:
            raise ValueError("field data is read-only")
        self.data[offset[0]:stop[0], offset[1]:stop[1], offset[2]:stop[2]] = values
        self.revision += 1
        self._updates.append((self.revision, offset, tuple(values.shape)))
        return self.revision

    def updates_since(self, revision):
        """Return ordered ``(revision, offset, shape)`` changes after revision."""
        revision = int(revision)
        if revision < 0 or revision > self.revision:
            raise ValueError("revision is outside this field's history")
        return tuple(update for update in self._updates if update[0] > revision)

    def sync_volume(self, scene, volume, transfer_function, *, since_revision):
        """Apply field changes to an existing renderer volume by dirty region."""
        updates = self.updates_since(since_revision)
        if not updates:
            return self.revision
        # Data-derived ranges can change when any sample changes, requiring a
        # full remap. Explicit and normalized ranges remain region-local.
        mapping = transfer_function.mapping
        region_safe = mapping.value_range is not None or mapping.mode == "normalized"
        if not region_safe:
            encoded, _valid = transfer_function.encode_volume(self.data)
            scene.update_volume(volume, data=encoded, value_range=(0.0, 1.0))
            return self.revision
        for _revision, offset, shape in updates:
            stop = tuple(start + size for start, size in zip(offset, shape))
            values = self.data[
                offset[0]:stop[0], offset[1]:stop[1], offset[2]:stop[2]
            ]
            encoded, _valid = transfer_function.encode_volume(values)
            scene.update_volume_region(volume, offset, encoded)
        return self.revision

    def snapshot(self, transfer_function=None):
        result = {
            "kind": "scalar_field_3d", "axis_order": "zyx",
            "shape": list(self.data.shape), "dtype": str(self.data.dtype),
            "spacing_xyz": list(self.spacing), "origin_xyz": list(self.origin),
            "direction": self.direction.tolist(), "unit": self.unit,
            "name": self.name, "metadata": dict(self.metadata),
            "revision": self.revision,
        }
        if transfer_function is not None:
            result["transfer_function"] = transfer_function.snapshot(self.data)
        return result
