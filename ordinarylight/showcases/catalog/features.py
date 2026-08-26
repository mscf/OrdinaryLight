"""Geometry, material, and vertex feature showcases."""

from ordinarylight.integrations.workbench import OrbitCamera, Showcase

from ordinarylight.showcases.primitives import build_primitive_showcase
from ordinarylight.showcases.materials import build_showcase_scene as build_material_showcase
from ordinarylight.showcases.vertex_attributes import build_vertex_attribute_showcase


SHOWCASES = (
    Showcase(
        "vertex-attributes", "Vertex attributes", build_vertex_attribute_showcase,
        camera=OrbitCamera(target=(0.0, 1.25, 0.0), radius=8.5, height=3.2),
        tags=("materials", "attributes"),
    ),
    Showcase(
        "primitives", "Points, lines, and glyphs", build_primitive_showcase,
        tags=("geometry", "primitives"),
    ),
    Showcase(
        "python-materials", "Python-authored materials", build_material_showcase,
        tags=("materials", "shaders"),
    ),
)
