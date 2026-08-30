"""Small scenes that isolate portable raster-renderer features."""

from __future__ import annotations

import math
import numpy as np

from ..lights import DirectionalLight, PointLight, SpotLight
from ..materials import unlit_material
from ..scene import Material, Scene, Texture
from .materials import diffuse, fresnel_glass, mirror, sphere


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


def _checker_texture(first, second, size=64, cells=8):
    y, x = np.indices((size, size))
    mask = ((x // (size // cells)) + (y // (size // cells))) % 2
    pixels = np.empty((size, size, 4), np.uint8)
    pixels[..., :3] = np.where(
        mask[..., None], np.asarray(second, np.uint8),
        np.asarray(first, np.uint8),
    )
    pixels[..., 3] = 255
    return Texture(pixels)


def build_advanced_material_scene():
    """Exercise every GPU material channel and tangent-space normal mapping."""
    scene = Scene()
    scene.add_light(PointLight(
        (-3.5, 5.5, 4.0), color=(1.0, 0.86, 0.72), intensity=95.0,
    ))
    floor_vertices = ((-7,0,-5),(7,0,-5),(7,0,5),(-7,0,5))
    floor_indices = ((0,2,1),(0,3,2))
    scene.add_mesh(
        floor_vertices, floor_indices,
        Material(base_color=(0.48, 0.52, 0.58), roughness=0.78),
        texcoords=((0,0),(4,0),(4,3),(0,3)),
    )

    base = _checker_texture((35, 70, 150), (220, 120, 35))
    mr = _checker_texture((0, 55, 255), (0, 235, 20))
    normal = _checker_texture((75, 128, 235), (181, 128, 235), cells=16)
    occlusion = _checker_texture((60, 60, 60), (255, 255, 255), cells=4)
    emission = _checker_texture((0, 0, 0), (255, 90, 18), cells=8)
    transmission = _checker_texture((20, 20, 20), (255, 255, 255), cells=8)
    definitions = (
        ((-2.8, 1.2, 0.0), Material(
            base_color=(1,1,1), roughness=0.72,
            base_color_texture=base, normal_texture=normal, normal_scale=1.0,
        )),
        ((0.0, 1.2, 0.0), Material(
            base_color=(0.86,0.74,0.32), metallic=1.0, roughness=1.0,
            metallic_roughness_texture=mr, occlusion_texture=occlusion,
            occlusion_strength=0.8,
        )),
        ((2.8, 1.2, 0.0), Material(
            base_color=(0.65,0.85,1.0), transmission=1.0, roughness=0.08,
            emission=(0.7,0.15,0.03), emissive_texture=emission,
            transmission_texture=transmission,
        )),
    )
    from .vertex_attributes import uv_sphere
    for center, material in definitions:
        vertices, indices, normals, texcoords = uv_sphere(
            center, 1.2, rings=24, segments=48,
        )
        scene.add_mesh(
            vertices, indices, material, normals=normals,
            texcoords=texcoords,
        )
    return scene


def build_material_program_parity_scene():
    """Show deterministic raster equivalents for GI material programs."""
    scene = Scene()
    scene.add_light(PointLight(
        (-3.5, 5.0, 3.5), color=(1.0, 0.88, 0.74), intensity=90.0,
    ))
    scene.add_mesh(
        ((-7,0,-5),(7,0,-5),(7,0,5),(-7,0,5)),
        ((0,2,1),(0,3,2)),
        Material(base_color=(0.58,0.62,0.68), roughness=0.8, program=diffuse),
    )
    definitions = (
        ((-3.0,1.15,0), Material(
            base_color=(0.18,0.52,0.92), roughness=0.65, program=diffuse,
        )),
        ((-1.0,1.15,0), Material(
            base_color=(0.94,0.68,0.18), metallic=1.0, program=mirror,
        )),
        ((1.0,1.15,0), Material(
            base_color=(0.75,0.9,1.0), transmission=1.0,
            program=fresnel_glass,
        )),
        ((3.0,1.15,0), Material(
            base_color=(0.9,0.16,0.48), program=unlit_material,
        )),
    )
    for center, material in definitions:
        vertices, indices = sphere(center, 1.05, rings=20, segments=40)
        scene.add_mesh(vertices, indices, material)
    return scene


__all__ = [
    "build_advanced_material_scene", "build_directional_shadow_scene",
    "build_material_program_parity_scene", "build_spot_shadow_scene",
]
