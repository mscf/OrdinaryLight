"""Bounded custom intersections and signed-distance utilities.

Geometry construction and artwork reconstruction remain application concerns.
"""

from .resources import IntersectionResource

from .fields import (
    BoundedField,
    FieldComposition,
    FieldKind,
    SdfSphere,
    UniformTransform,
    intersect_field,
)
from .intersection import (
    CustomGeometry,
    IntersectionProgram,
    SurfaceHit,
    intersect_triangle,
)

__all__ = [
    "IntersectionResource",
    "BoundedField",
    "FieldComposition",
    "intersect_field",
    "FieldKind",
    "SdfSphere",
    "UniformTransform",
    "CustomGeometry",
    "IntersectionProgram",
    "SurfaceHit",
    "intersect_triangle",
]
