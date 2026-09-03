"""Shared scientific clipping planes and regions of interest."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class ClipPlane:
    """World-space half-space retaining ``dot(normal, position) >= offset``."""

    normal: tuple[float, float, float]
    offset: float

    def __post_init__(self):
        normal = np.asarray(self.normal, np.float64)
        offset = float(self.offset)
        if normal.shape != (3,) or not np.isfinite(normal).all():
            raise ValueError("normal must contain three finite values")
        length = float(np.linalg.norm(normal))
        if length <= 1e-12 or not np.isfinite(offset):
            raise ValueError("clip plane must have a nonzero normal and finite offset")
        object.__setattr__(self, "normal", tuple(map(float, normal / length)))
        object.__setattr__(self, "offset", offset / length)

    def signed_distance(self, positions):
        return np.asarray(positions) @ np.asarray(self.normal) - self.offset


@dataclass(frozen=True, slots=True)
class RegionOfInterest:
    """Inclusive axis-aligned bounds in world or field-index coordinates."""

    minimum: tuple[float, float, float]
    maximum: tuple[float, float, float]
    space: str = "index"

    def __post_init__(self):
        minimum = np.asarray(self.minimum, np.float64)
        maximum = np.asarray(self.maximum, np.float64)
        if minimum.shape != (3,) or maximum.shape != (3,) or not (
            np.isfinite(minimum).all() and np.isfinite(maximum).all()
        ):
            raise ValueError("ROI bounds must contain three finite values")
        if np.any(maximum < minimum):
            raise ValueError("ROI maximum must not be below minimum")
        if self.space not in {"index", "world"}:
            raise ValueError("ROI space must be 'index' or 'world'")
        object.__setattr__(self, "minimum", tuple(map(float, minimum)))
        object.__setattr__(self, "maximum", tuple(map(float, maximum)))


@dataclass(frozen=True, slots=True)
class ClipRegion:
    """A reproducible intersection of planes and an optional ROI."""

    planes: tuple[ClipPlane, ...] = ()
    roi: RegionOfInterest | None = None

    def __post_init__(self):
        planes = tuple(self.planes)
        if any(not isinstance(plane, ClipPlane) for plane in planes):
            raise TypeError("planes must contain ClipPlane values")
        if self.roi is not None and not isinstance(self.roi, RegionOfInterest):
            raise TypeError("roi must be a RegionOfInterest or None")
        object.__setattr__(self, "planes", planes)

    def contains(self, field, world_positions, index_positions=None):
        world = np.asarray(world_positions, np.float64)
        scalar = world.ndim == 1
        world = np.atleast_2d(world)
        keep = np.ones(len(world), bool)
        for plane in self.planes:
            keep &= plane.signed_distance(world) >= -1e-9
        if self.roi is not None:
            coordinates = world if self.roi.space == "world" else (
                field.world_to_index(world) if index_positions is None
                else np.atleast_2d(index_positions)
            )
            keep &= np.all(coordinates >= np.asarray(self.roi.minimum) - 1e-9, axis=1)
            keep &= np.all(coordinates <= np.asarray(self.roi.maximum) + 1e-9, axis=1)
        return bool(keep[0]) if scalar else keep

    def world_planes(self, field):
        """Return all constraints as normalized world-space planes."""
        result = list(self.planes)
        if self.roi is None:
            return tuple(result)
        if self.roi.space == "world":
            minimum, maximum = map(np.asarray, (self.roi.minimum, self.roi.maximum))
            for axis in range(3):
                normal = np.zeros(3); normal[axis] = 1
                result.append(ClipPlane(tuple(normal), minimum[axis]))
                result.append(ClipPlane(tuple(-normal), -maximum[axis]))
        else:
            inverse = np.linalg.inv(field.index_to_world_matrix)
            for axis in range(3):
                row = inverse[axis, :3]
                translation = inverse[axis, 3]
                result.append(ClipPlane(tuple(row), self.roi.minimum[axis] - translation))
                result.append(ClipPlane(tuple(-row), -self.roi.maximum[axis] + translation))
        return tuple(result)

    def clip_mesh(self, field, vertices, indices, attributes=()):
        """Clip triangles exactly and interpolate any per-vertex attributes."""
        vertices = np.asarray(vertices, np.float64)
        attributes = tuple(np.asarray(values, np.float64) for values in attributes)
        output_vertices, output_attributes, output_indices = [], [[] for _ in attributes], []
        for triangle in np.asarray(indices, np.int64):
            polygon = [(vertices[i], tuple(values[i] for values in attributes)) for i in triangle]
            for plane in self.world_planes(field):
                clipped = []
                if not polygon:
                    break
                previous = polygon[-1]
                previous_distance = float(plane.signed_distance(previous[0]))
                for current in polygon:
                    current_distance = float(plane.signed_distance(current[0]))
                    if (current_distance >= 0) != (previous_distance >= 0):
                        weight = previous_distance / (previous_distance - current_distance)
                        point = previous[0] + weight * (current[0] - previous[0])
                        values = tuple(a + weight * (b - a) for a, b in zip(previous[1], current[1]))
                        clipped.append((point, values))
                    if current_distance >= 0:
                        clipped.append(current)
                    previous, previous_distance = current, current_distance
                polygon = clipped
            if len(polygon) < 3:
                continue
            base = len(output_vertices)
            for point, values in polygon:
                output_vertices.append(point)
                for output, value in zip(output_attributes, values):
                    output.append(value)
            for corner in range(1, len(polygon) - 1):
                output_indices.append((base, base + corner, base + corner + 1))
        vertices_result = np.asarray(output_vertices, np.float32).reshape((-1, 3))
        indices_result = np.asarray(output_indices, np.uint32).reshape((-1, 3))
        attributes_result = tuple(
            np.asarray(values, np.float32).reshape((-1, source.shape[1]))
            for values, source in zip(output_attributes, attributes)
        )
        return vertices_result, indices_result, attributes_result

    def snapshot(self):
        return {
            "planes": [
                {"normal": list(plane.normal), "offset": plane.offset}
                for plane in self.planes
            ],
            "roi": None if self.roi is None else {
                "minimum": list(self.roi.minimum), "maximum": list(self.roi.maximum),
                "space": self.roi.space,
            },
        }
