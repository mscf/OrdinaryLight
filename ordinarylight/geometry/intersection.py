"""Common host hit contract and the programmable custom-intersection boundary."""

from dataclasses import dataclass
import re

import numpy as np
from .fields import FieldKind, vector3
from .resources import IntersectionResource


@dataclass(frozen=True)
class SurfaceHit:
    distance: float
    position: object
    geometric_normal: object
    shading_normal: object
    primitive: int
    identity: int
    material: int
    boundary: int | None


@dataclass(frozen=True)
class IntersectionProgram:
    """A bounded GLSL callback, independent of the voxel representation.

    uint entry(vec3 origin, vec3 unit_direction, float t_min, float t_max,
               vec4 parameters, float tolerance, uint max_steps,
               out float distance, out vec3 geometric_normal)

    Return 0 for miss, 1 for hit, 2 for unresolved/error. Bounds and hit distances
    are in world units. The callback must respect its declared stepping guarantee.
    Scalar fields require their own root-finding callback, not sphere tracing.
    """

    name: str
    source: str
    field_kind: FieldKind | None = None
    resources: tuple[IntersectionResource, ...] = ()

    def __post_init__(self):
        object.__setattr__(self, "resources", tuple(self.resources))
        if not all(isinstance(r, IntersectionResource) for r in self.resources):
            raise TypeError("Expected IntersectionResource declarations")
        if len({r.name for r in self.resources}) != len(self.resources):
            raise ValueError("Duplicate intersection resource names")
        if (
            not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", self.name)
            or not self.source.strip()
        ):
            raise ValueError("Intersection program needs a GLSL identifier and source")
        if self.field_kind is not None and not isinstance(self.field_kind, FieldKind):
            raise TypeError("field_kind must be FieldKind or None")

    @classmethod
    def sdf_sphere(cls):
        from importlib.resources import files

        return cls(
            "ordinarylightSdfSphere",
            files("ordinarylight.shaders")
            .joinpath("transport_v1/sdf_sphere.glsl")
            .read_text(),
            FieldKind.EXACT_DISTANCE,
        )


@dataclass(frozen=True)
class CustomGeometry:
    bounds: object
    program: IntersectionProgram
    parameters: tuple[float, float, float, float]
    material: int = 0
    boundary: int | None = None
    identity: int = 0

    def __post_init__(self):
        bounds = np.asarray(self.bounds, dtype=np.float64)
        if (
            bounds.shape != (2, 3)
            or not np.isfinite(bounds).all()
            or np.any(bounds[1] <= bounds[0])
        ):
            raise ValueError("Custom geometry needs finite, nondegenerate world bounds")
        object.__setattr__(self, "bounds", tuple(tuple(row) for row in bounds))
        params = np.asarray(self.parameters, dtype=np.float64)
        if params.shape != (4,) or not np.isfinite(params).all():
            raise ValueError("Custom geometry parameters must be four finite numbers")
        if (
            not np.isfinite(bounds.astype(np.float32)).all()
            or np.any(bounds.astype(np.float32)[1] <= bounds.astype(np.float32)[0])
            or not np.isfinite(params.astype(np.float32)).all()
        ):
            raise ValueError(
                "Geometry bounds and parameters must be representable in float32"
            )
        if not isinstance(self.program, IntersectionProgram):
            raise TypeError("Expected IntersectionProgram")
        if min(self.material, self.identity) < 0 or (
            self.boundary is not None and self.boundary < 0
        ):
            raise ValueError("Geometry identifiers must be nonnegative")
        object.__setattr__(self, "parameters", tuple(params))


def intersect_triangle(
    origin,
    direction,
    vertices,
    *,
    t_min=0.0,
    t_max=1e6,
    identity=0,
    material=0,
    boundary=None,
):
    origin = vector3(origin, "origin")
    direction = vector3(direction, "direction")
    vertices = np.asarray(vertices, dtype=np.float64)
    if vertices.shape != (3, 3) or not np.isfinite(vertices).all():
        raise ValueError("Triangle vertices must be finite (3,3) coordinates")
    edge1, edge2 = vertices[1] - vertices[0], vertices[2] - vertices[0]
    p = np.cross(direction, edge2)
    determinant = np.dot(edge1, p)
    if abs(determinant) < 1e-12:
        return None
    u = np.dot(origin - vertices[0], p) / determinant
    q = np.cross(origin - vertices[0], edge1)
    v = np.dot(direction, q) / determinant
    distance = float(np.dot(edge2, q) / determinant)
    if u < 0 or v < 0 or u + v > 1 or not t_min <= distance <= t_max:
        return None
    normal = np.cross(edge1, edge2)
    normal /= np.linalg.norm(normal)
    return SurfaceHit(
        distance,
        origin + distance * direction,
        normal,
        normal.copy(),
        0,
        identity,
        material,
        boundary,
    )
