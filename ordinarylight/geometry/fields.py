"""Small field contract with explicit sphere-tracing guarantees."""

from dataclasses import dataclass
from enum import Enum
from typing import Protocol, runtime_checkable

import numpy as np


class FieldKind(str, Enum):
    EXACT_DISTANCE = "exact_signed_distance"
    CONSERVATIVE_DISTANCE = "conservative_signed_distance_bound"
    SCALAR = "scalar_field"


@runtime_checkable
class BoundedField(Protocol):
    kind: FieldKind

    @property
    def bounds(self): ...
    def evaluate(self, points): ...
    def gradient(self, point): ...


def vector3(value, name):
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (3,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must contain three finite numbers")
    return result


@dataclass(frozen=True)
class UniformTransform:
    """A rigid transform with positive uniform scale; preserves distance units.

    Nonuniform affine transforms are deliberately rejected: multiplying a field
    by an arbitrary scale does not preserve exact signed distance.
    """

    matrix: object

    def __post_init__(self):
        matrix = np.array(self.matrix, dtype=np.float64, copy=True)
        if (
            matrix.shape != (4, 4)
            or not np.isfinite(matrix).all()
            or not np.allclose(matrix[3], [0, 0, 0, 1])
        ):
            raise ValueError("Expected a finite affine 4x4 transform")
        linear = matrix[:3, :3]
        scale = float(np.linalg.norm(linear[:, 0]))
        if (
            scale <= 0
            or not np.allclose(
                linear.T @ linear, np.eye(3) * scale**2, rtol=1e-7, atol=1e-12
            )
            or np.linalg.det(linear) <= 0
        ):
            raise ValueError(
                "SDF transform requires positive uniform scale and a rotation"
            )
        matrix.flags.writeable = False
        object.__setattr__(self, "matrix", matrix)

    @property
    def scale(self):
        return float(np.linalg.norm(self.matrix[:3, 0]))

    def point(self, point):
        return self.matrix[:3, :3] @ vector3(point, "point") + self.matrix[:3, 3]

    def gradient(self, gradient):
        result = self.matrix[:3, :3] @ vector3(gradient, "gradient")
        norm = np.linalg.norm(result)
        if norm == 0:
            raise ValueError("A zero gradient has no surface normal")
        return result / norm


@dataclass(frozen=True)
class SdfSphere:
    center: tuple[float, float, float] = (0.0, 0.0, 0.0)
    radius: float = 1.0
    kind: FieldKind = FieldKind.EXACT_DISTANCE

    def __post_init__(self):
        object.__setattr__(self, "center", tuple(vector3(self.center, "center")))
        if not np.isfinite(self.radius) or self.radius <= 0:
            raise ValueError("Sphere radius must be finite and positive")
        if self.kind != FieldKind.EXACT_DISTANCE:
            raise ValueError("An analytic sphere is an exact signed distance field")

    @property
    def bounds(self):
        center = np.asarray(self.center)
        return center - self.radius, center + self.radius

    def evaluate(self, points):
        points = np.asarray(points, dtype=np.float64)
        if points.shape[-1:] != (3,) or not np.isfinite(points).all():
            raise ValueError("Field points must be finite (..., 3) values")
        return np.linalg.norm(points - self.center, axis=-1) - self.radius

    def gradient(self, point):
        offset = vector3(point, "point") - self.center
        norm = np.linalg.norm(offset)
        if norm == 0:
            raise ValueError("Sphere gradient is undefined at its center")
        return offset / norm

    def transformed(self, transform):
        if not isinstance(transform, UniformTransform):
            transform = UniformTransform(transform)
        return SdfSphere(
            tuple(transform.point(self.center)), self.radius * transform.scale
        )

    def intersect(
        self, origin, direction, *, t_min=0.0, t_max=1e6, tolerance=1e-5, max_steps=256
    ):
        """Bounded sphere tracing, including inside starts; exhaustion is explicit.

        Returns (distance, geometric_normal), or None for a proven bounds miss.
        Tangencies may exhaust the step budget; they are never silently misses.
        """
        return intersect_field(
            self,
            origin,
            direction,
            t_min=t_min,
            t_max=t_max,
            tolerance=tolerance,
            max_steps=max_steps,
        )

    def geometry(self, *, material=0, boundary=None, identity=0):
        from .intersection import CustomGeometry, IntersectionProgram

        return CustomGeometry(
            self.bounds,
            IntersectionProgram.sdf_sphere(),
            (*self.center, self.radius),
            material,
            boundary,
            identity,
        )


def intersect_field(
    field, origin, direction, *, t_min=0.0, t_max=1e6, tolerance=1e-5, max_steps=256
):
    """Sphere-trace only fields declaring a safe world-distance bound.

    Arbitrary scalar fields require a root-finding intersection implementation.
    A distance estimate must be a conservative bound, not merely approximate.
    """
    if not isinstance(field, BoundedField) or field.kind not in (
        FieldKind.EXACT_DISTANCE,
        FieldKind.CONSERVATIVE_DISTANCE,
    ):
        raise ValueError(
            "Sphere tracing requires an exact distance or conservative distance bound"
        )
    origin = vector3(origin, "origin")
    direction = vector3(direction, "direction")
    if not np.isclose(np.linalg.norm(direction), 1.0, rtol=1e-7):
        raise ValueError("Ray direction must have unit length")
    if (
        not 0 <= t_min < t_max
        or not np.isfinite([t_min, t_max, tolerance]).all()
        or tolerance <= 0
        or max_steps <= 0
    ):
        raise ValueError("Invalid traversal limits")
    lower, upper = field.bounds
    if np.any(np.asarray(lower) > upper):
        return None
    near, far = float(t_min), float(t_max)
    for axis in range(3):
        if abs(direction[axis]) < 1e-15:
            if origin[axis] < lower[axis] or origin[axis] > upper[axis]:
                return None
        else:
            first, last = sorted(
                (
                    (lower[axis] - origin[axis]) / direction[axis],
                    (upper[axis] - origin[axis]) / direction[axis],
                )
            )
            near, far = max(near, first), min(far, last)
    if near > far:
        return None
    distance = near
    start = origin + distance * direction
    start_value = float(field.evaluate(start))
    leaving_origin_root = (
        abs(start_value) <= tolerance
        and start_value * np.dot(field.gradient(start), direction) > 0
    )
    for _ in range(max_steps):
        point = origin + distance * direction
        value = float(field.evaluate(point))
        if leaving_origin_root and abs(value) > tolerance:
            leaving_origin_root = False
        if abs(value) <= tolerance and not leaving_origin_root:
            return distance, field.gradient(point)
        distance += max(abs(value), tolerance) if leaving_origin_root else abs(value)
        if distance > far + tolerance:
            return None
    raise RuntimeError("SDF traversal exhausted its step budget")


@dataclass(frozen=True)
class FieldComposition:
    left: BoundedField
    right: BoundedField
    operation: str = "union"

    def __post_init__(self):
        if (
            not isinstance(self.left, BoundedField)
            or not isinstance(self.right, BoundedField)
            or self.operation not in {"union", "intersection", "difference"}
        ):
            raise ValueError(
                "Field composition needs bounded fields and a supported operation"
            )

    @property
    def kind(self):
        return (
            FieldKind.SCALAR
            if FieldKind.SCALAR in (self.left.kind, self.right.kind)
            else FieldKind.CONSERVATIVE_DISTANCE
        )

    @property
    def bounds(self):
        a, b = np.asarray(self.left.bounds), np.asarray(self.right.bounds)
        if self.operation == "union":
            return np.minimum(a[0], b[0]), np.maximum(a[1], b[1])
        if self.operation == "difference":
            return a[0], a[1]
        return np.maximum(a[0], b[0]), np.minimum(a[1], b[1])

    def evaluate(self, points):
        a, b = self.left.evaluate(points), self.right.evaluate(points)
        if self.operation == "union":
            return np.minimum(a, b)
        return np.maximum(a, -b if self.operation == "difference" else b)

    def gradient(self, point):
        a, b = float(self.left.evaluate(point)), float(self.right.evaluate(point))
        if self.operation == "difference":
            b = -b
        left = (a <= b) if self.operation == "union" else (a >= b)
        if left:
            return self.left.gradient(point)
        return self.right.gradient(point) * (
            -1 if self.operation == "difference" else 1
        )
