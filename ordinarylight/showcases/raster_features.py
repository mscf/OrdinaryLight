"""Small scenes that isolate portable raster-renderer features."""

from __future__ import annotations

import math

from ..lights import DirectionalLight, SpotLight
from ..scene import Material, Scene


def _box(center, size, material):
    cx, cy, cz = center
    sx, sy, sz = (value * 0.5 for value in size)
    corners = [
        (cx + x * sx, cy + y * sy, cz + z * sz)
        for x, y, z in (
            (-1,-1,-1), (1,-1,-1), (1,1,-1), (-1,1,-1),
            (-1,-1,1), (1,-1,1), (1,1,1), (-1,1,1),
        )
    ]
    corner_triangles = (
        (0,2,1),(0,3,2), (4,5,6),(4,6,7),
        (0,1,5),(0,5,4), (2,3,7),(2,7,6),
        (1,2,6),(1,6,5), (3,0,4),(3,4,7),
    )
    vertices, normals, indices = [], [], []
    # Keep each face's vertices independent and provide explicit flat normals.
    # The eight-corner form causes generated normals to be averaged across the
    # box's hard 90-degree edges.
    for face in range(0, len(corner_triangles), 2):
        first, second = corner_triangles[face:face + 2]
        face_corners = tuple(dict.fromkeys((*first, *second)))
        base = len(vertices)
        vertices.extend(corners[index] for index in face_corners)
        lookup = {corner: base + index for index, corner in enumerate(face_corners)}
        indices.extend(
            tuple(lookup[corner] for corner in triangle)
            for triangle in (first, second)
        )
        p0, p1, p2 = (corners[index] for index in first)
        edge1 = tuple(b - a for a, b in zip(p0, p1))
        edge2 = tuple(b - a for a, b in zip(p0, p2))
        normal = (
            edge1[1] * edge2[2] - edge1[2] * edge2[1],
            edge1[2] * edge2[0] - edge1[0] * edge2[2],
            edge1[0] * edge2[1] - edge1[1] * edge2[0],
        )
        length = math.sqrt(sum(value * value for value in normal))
        normal = tuple(value / length for value in normal)
        normals.extend((normal,) * len(face_corners))
    return vertices, indices, material, normals


def _shadow_receivers():
    scene = Scene()
    scene.add_mesh(
        ((-5,0,-4),(5,0,-4),(5,0,4),(-5,0,4)),
        ((0,2,1),(0,3,2)),
        Material(base_color=(0.68, 0.72, 0.78), roughness=0.82),
    )
    for center, size, color in (
        ((-1.35,0.8,0.2),(1.3,1.6,1.3),(0.86,0.28,0.16)),
        ((1.15,1.25,-0.35),(1.6,2.5,1.6),(0.18,0.42,0.88)),
    ):
        vertices, indices, material, normals = _box(
            center, size, Material(base_color=color, roughness=0.38),
        )
        scene.add_mesh(vertices, indices, material, normals=normals)
    return scene


def build_directional_shadow_scene():
    scene = _shadow_receivers()
    scene.add_light(DirectionalLight(
        direction=(0.15, -1.0, 1.0), color=(1.0, 0.93, 0.82),
        intensity=5.0,
    ))
    return scene


def build_spot_shadow_scene():
    scene = _shadow_receivers()
    scene.add_light(SpotLight(
        position=(-2.8, 5.5, 3.8), direction=(0.35, -1.0, -0.55),
        color=(0.82, 0.90, 1.0), intensity=65.0,
        inner_cone_angle=math.radians(18),
        outer_cone_angle=math.radians(34), range=12.0,
    ))
    return scene


__all__ = ["build_directional_shadow_scene", "build_spot_shadow_scene"]
