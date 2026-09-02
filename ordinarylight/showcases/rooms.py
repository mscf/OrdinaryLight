"""Deterministic procedural scenes used by the ReSTIR regression matrix."""

from dataclasses import dataclass
from typing import Callable

import numpy as np
import ordinarylight as ol

from .area_lights import build_area_light_showcase
from .materials import diffuse, fresnel_glass, mirror, quad, sphere


@dataclass(frozen=True)
class RestirSceneSpec:
    name: str
    description: str
    build: Callable[[], ol.Scene]
    orbit_radius: float = 8.5
    camera_height: float = 3.2
    target: tuple[float, float, float] = (0.0, 1.25, 0.0)
    # These fixtures are rooms open on -Z, rather than subjects intended for
    # a full turntable orbit.  Presentation demos therefore oscillate across
    # this front-facing arc instead of moving the camera through the walls.
    presentation_arc_radians: float = 0.48


def _add_quad(scene, corners, material, *, texcoords=None):
    vertices, indices = quad(*corners)
    return scene.add_mesh(
        vertices, indices, material,
        texcoords=None if texcoords is None else np.asarray(texcoords, np.float32),
    )


def _add_room(scene, floor=(0.68, 0.70, 0.74)):
    surfaces = (
        (((-6, 0, -5), (6, 0, -5), (6, 0, 5), (-6, 0, 5)), floor),
        (((-6, 0, 5), (6, 0, 5), (6, 7, 5), (-6, 7, 5)), (0.70, 0.72, 0.76)),
        (((-6, 0, -5), (-6, 0, 5), (-6, 7, 5), (-6, 7, -5)), (0.62, 0.18, 0.14)),
        (((6, 0, 5), (6, 0, -5), (6, 7, -5), (6, 7, 5)), (0.14, 0.24, 0.58)),
    )
    for corners, color in surfaces:
        _add_quad(scene, corners, ol.Material(base_color=color, program=diffuse))


def _add_emitter(scene, corners, emission=(12.0, 9.0, 6.0)):
    _add_quad(scene, corners, ol.Material(
        base_color=(1.0, 0.9, 0.75), emission=emission,
        emission_two_sided=True, program=diffuse,
    ))


def build_diffuse_room():
    scene = ol.Scene()
    _add_room(scene)
    _add_emitter(scene, (
        (-1.8, 6.7, -1.2), (1.8, 6.7, -1.2),
        (1.8, 6.7, 1.2), (-1.8, 6.7, 1.2),
    ))
    for center, radius, color in (
        ((-2.0, 1.35, 0.4), 1.35, (0.75, 0.38, 0.18)),
        ((1.8, 1.05, -0.3), 1.05, (0.22, 0.62, 0.36)),
    ):
        vertices, indices = sphere(center, radius)
        scene.add_mesh(vertices, indices, ol.Material(
            base_color=color, roughness=0.85, program=diffuse,
        ))
    return scene


def build_glossy_glass():
    scene = ol.Scene()
    _add_room(scene)
    _add_emitter(scene, (
        (-2.4, 6.7, -1.1), (0.4, 6.7, -1.1),
        (0.4, 6.7, 1.1), (-2.4, 6.7, 1.1),
    ), (16.0, 10.0, 5.0))
    vertices, indices = sphere((-1.8, 1.45, 0.2), 1.45)
    scene.add_mesh(vertices, indices, ol.Material(
        base_color=(0.92, 0.72, 0.22), metallic=1.0, roughness=0.12,
        program=mirror,
    ))
    vertices, indices = sphere((1.6, 1.55, -0.1), 1.55)
    scene.add_mesh(vertices, indices, ol.Material(
        base_color=(0.9, 0.97, 1.0), transmission=1.0, roughness=0.04,
        ior=1.52, program=fresnel_glass,
    ))
    return scene


def build_textured_room():
    scene = ol.Scene()
    _add_room(scene, floor=(1.0, 1.0, 1.0))
    checker = np.asarray((
        ((235, 235, 235, 255), (38, 55, 82, 255)),
        ((38, 55, 82, 255), (235, 235, 235, 255)),
    ), dtype=np.uint8)
    texture = ol.Texture(checker, linear_filter=False)
    # Overlay the room floor by a small offset to exercise UV/material identity
    # without introducing coplanar geometry.
    _add_quad(scene, (
        (-5.8, 0.002, -4.8), (5.8, 0.002, -4.8),
        (5.8, 0.002, 4.8), (-5.8, 0.002, 4.8),
    ), ol.Material(
        base_color=(1.0, 1.0, 1.0), base_color_texture=texture,
        roughness=0.72, program=diffuse,
    ), texcoords=((0, 0), (8, 0), (8, 8), (0, 8)))
    _add_emitter(scene, (
        (-1.5, 6.7, -1.0), (1.5, 6.7, -1.0),
        (1.5, 6.7, 1.0), (-1.5, 6.7, 1.0),
    ))
    vertices, indices = sphere((0.0, 1.4, 0.0), 1.4)
    scene.add_mesh(vertices, indices, ol.Material(
        base_color=(0.72, 0.32, 0.18), roughness=0.55, program=diffuse,
    ))
    return scene


def build_small_emitter():
    scene = ol.Scene()
    _add_room(scene, floor=(0.58, 0.60, 0.64))
    _add_emitter(scene, (
        (-0.38, 5.1, 4.86), (-0.38, 5.85, 4.86),
        (0.38, 5.85, 4.86), (0.38, 5.1, 4.86),
    ), (90.0, 58.0, 28.0))
    vertices, indices = sphere((-1.4, 1.45, 0.2), 1.45)
    scene.add_mesh(vertices, indices, ol.Material(
        base_color=(0.72, 0.76, 0.82), roughness=0.35, program=diffuse,
    ))
    vertices, indices = sphere((1.8, 0.95, -0.5), 0.95)
    scene.add_mesh(vertices, indices, ol.Material(
        base_color=(0.25, 0.35, 0.72), roughness=0.8, program=diffuse,
    ))
    return scene


def build_occlusion_room():
    scene = ol.Scene()
    _add_room(scene)
    _add_emitter(scene, (
        (-2.0, 6.7, -1.2), (2.0, 6.7, -1.2),
        (2.0, 6.7, 1.2), (-2.0, 6.7, 1.2),
    ), (18.0, 13.0, 8.0))
    dark = ol.Material(base_color=(0.09, 0.11, 0.14), roughness=0.8, program=diffuse)
    for x in (-1.5, -0.5, 0.5, 1.5):
        _add_quad(scene, (
            (x - 0.12, 0.0, 0.2), (x + 0.12, 0.0, 0.2),
            (x + 0.12, 4.4, 0.2), (x - 0.12, 4.4, 0.2),
        ), dark)
    vertices, indices = sphere((0.0, 1.2, -1.4), 1.2)
    scene.add_mesh(vertices, indices, ol.Material(
        base_color=(0.72, 0.43, 0.19), roughness=0.62, program=diffuse,
    ))
    return scene


def build_nested_glass():
    scene = build_diffuse_room()
    for radius, ior, color in (
        (1.65, 1.52, (0.92, 0.97, 1.0)),
        (1.18, 1.33, (0.82, 0.94, 1.0)),
        (0.72, 1.62, (1.0, 0.88, 0.72)),
    ):
        vertices, indices = sphere((0.2, 1.68, -0.25), radius)
        scene.add_mesh(vertices, indices, ol.Material(
            base_color=color, transmission=1.0, roughness=0.025,
            ior=ior, program=fresnel_glass,
        ))
    return scene


def build_dense_geometry():
    scene = ol.Scene()
    _add_room(scene)
    _add_emitter(scene, (
        (-2.2, 6.7, -1.3), (2.2, 6.7, -1.3),
        (2.2, 6.7, 1.3), (-2.2, 6.7, 1.3),
    ), (15.0, 12.0, 9.0))
    sphere_vertices, sphere_indices = sphere(
        (0.0, 0.0, 0.0), 1.0, rings=12, segments=24
    )
    sphere_resource = scene.create_mesh(
        sphere_vertices, sphere_indices,
        ol.Material(base_color=(0.5, 0.5, 0.5), program=diffuse),
    )
    for row in range(5):
        for column in range(8):
            radius = 0.34 + 0.04 * ((row + column) % 3)
            center = (
                -4.25 + column * 1.22,
                radius,
                -3.0 + row * 1.35,
            )
            color = (
                0.22 + 0.07 * (column % 4),
                0.28 + 0.08 * (row % 3),
                0.32 + 0.06 * ((row + column) % 4),
            )
            scene.add_instance(
                sphere_resource,
                transform=(
                    ol.Transform.translation(center)
                    @ ol.Transform.scale(radius)
                ),
                material=ol.Material(
                    base_color=color, roughness=0.3 + 0.12 * (row % 4),
                    program=diffuse,
                ),
            )
    return scene


def build_object_motion_room():
    """Small rigid-motion fixture for temporal reconstruction testing."""
    scene = ol.Scene()
    _add_room(scene, floor=(0.52, 0.55, 0.60))
    _add_emitter(scene, (
        (-1.7, 6.7, -1.0), (1.7, 6.7, -1.0),
        (1.7, 6.7, 1.0), (-1.7, 6.7, 1.0),
    ), (15.0, 12.0, 9.0))
    vertices, indices = sphere((0.0, 0.0, 0.0), 0.85)
    moving = scene.add_mesh(
        vertices, indices,
        ol.Material(base_color=(0.92, 0.28, 0.12), roughness=0.42,
                    program=diffuse),
        transform=ol.Transform.translation((-2.0, 1.0, 0.0)),
        name="moving-sphere",
    )
    vertices, indices = sphere((0.0, 0.0, 0.0), 1.0)
    scene.add_mesh(
        vertices, indices,
        ol.Material(base_color=(0.18, 0.42, 0.88), roughness=0.3,
                    program=diffuse),
        transform=ol.Transform.translation((1.8, 1.05, -0.35)),
        name="stationary-sphere",
    )
    scene.add_animation(ol.AnimationClip((ol.AnimationTrack(
        moving, "translation", (0.0, 1.5, 3.0),
        ((-2.0, 1.0, 0.0), (0.0, 1.8, 0.0), (-2.0, 1.0, 0.0)),
    ),), name="rigid-object-motion"))
    return scene


def animate_object_motion_room(scene, time):
    """Advance the rigid-motion fixture without moving its camera."""
    scene.apply_animation(scene.animations[0], time, loop=True)


SCENES = {
    spec.name: spec for spec in (
        RestirSceneSpec("area_lights", "mixed metal, glass, and two emitters", build_area_light_showcase),
        RestirSceneSpec("diffuse", "diffuse-only baseline", build_diffuse_room),
        RestirSceneSpec("glossy_glass", "glossy and transmissive boundaries", build_glossy_glass),
        RestirSceneSpec("textured", "repeating texture and material identity", build_textured_room),
        RestirSceneSpec("small_emitter", "small high-energy emitter", build_small_emitter),
        RestirSceneSpec("occlusion", "thin moving-camera occlusion boundaries", build_occlusion_room),
        RestirSceneSpec("nested_glass", "three nested transmissive boundaries", build_nested_glass),
        RestirSceneSpec("dense", "forty objects and dense acceleration geometry", build_dense_geometry),
    )
}


def get_restir_scene(name):
    return SCENES[name]
