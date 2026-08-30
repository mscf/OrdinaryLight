"""Reflection, refraction, absorption, and transparency showcases."""

from __future__ import annotations

import numpy as np

from ..lights import EnvironmentLight, ReflectionProbe, SpotLight
from ..scene import Material, Scene, Texture
from .materials import quad, sphere


def _environment(width=512, height=256):
    u = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :]
    v = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    image = np.empty((height, width, 3), np.float32)
    image[..., 0] = 0.04 + 0.45 * u + 1.4 * np.exp(-((u - .22) ** 2 + (v - .34) ** 2) / .002)
    image[..., 1] = 0.08 + 0.25 * (1.0 - v) + 1.0 * np.exp(-((u - .22) ** 2 + (v - .34) ** 2) / .002)
    image[..., 2] = 0.16 + 0.55 * (1.0 - v)
    image[v[:, 0] > .72] *= np.asarray((.38, .24, .12), np.float32)
    # High-frequency architectural landmarks make roughness, reflection
    # direction, IOR, and absorption visually distinguishable.
    bands = ((np.floor(u * 18.0) + np.floor(v * 9.0)) % 2.0) > 0.5
    image += bands[..., None] * np.asarray((.10, .035, .16), np.float32)
    image[:, width // 2 - 5:width // 2 + 5] += (1.8, .45, .12)
    image[height // 3 - 4:height // 3 + 4, :] += (.15, .8, 1.4)
    return image


def _checker_texture(first, second, *, size=128, cells=12):
    y, x = np.indices((size, size))
    cell = max(size // cells, 1)
    mask = ((x // cell) + (y // cell)) % 2
    pixels = np.empty((size, size, 4), np.uint8)
    pixels[..., :3] = np.where(
        mask[..., None], np.asarray(second, np.uint8),
        np.asarray(first, np.uint8),
    )
    pixels[..., 3] = 255
    # Clamp keeps the authored 0..1 corner UVs distinct. Repeat wrapping is
    # evaluated per vertex by the packed-atlas path and would fold UV=1 to 0.
    return Texture(
        pixels, wrap_s="clamp", wrap_t="clamp", linear_filter=False,
    )


def _add_quad(scene, corners, material, name, texcoords=None):
    vertices, indices = quad(*corners)
    return scene.add_mesh(
        vertices, indices, material, name=name,
        texcoords=None if texcoords is None else np.asarray(texcoords, np.float32),
    )


def _add_diorama(scene):
    floor_texture = _checker_texture((220, 223, 216), (42, 53, 72), cells=16)
    wall_texture = _checker_texture((28, 72, 112), (224, 116, 48), cells=10)
    _add_quad(scene, (
        (-6.5, 0, -4.5), (-6.5, 0, 4.5),
        (6.5, 0, 4.5), (6.5, 0, -4.5),
    ), Material(
        base_color=(1, 1, 1), base_color_texture=floor_texture,
        metallic=.03, roughness=.42,
    ), "checker-floor", ((0, 0), (0, 1), (1, 1), (1, 0)))
    _add_quad(scene, (
        (-6.5, 0, -4.5), (6.5, 0, -4.5),
        (6.5, 5.8, -4.5), (-6.5, 5.8, -4.5),
    ), Material(
        base_color=(1, 1, 1), base_color_texture=wall_texture,
        roughness=.65,
    ), "pattern-wall", ((0, 0), (1, 0), (1, 1), (0, 1)))

    # Saturated panels and luminous strips provide recognizable shapes in
    # both true GI rays and the raster environment approximation.
    for x, color in ((-4.7, (.9, .08, .06)), (4.7, (.06, .35, 1.0))):
        _add_quad(scene, (
            (x - .65, .65, -4.46), (x + .65, .65, -4.46),
            (x + .65, 3.7, -4.46), (x - .65, 3.7, -4.46),
        ), Material(base_color=color, roughness=.25),
        f"reference-panel-{x:+.1f}")
    _add_quad(scene, (
        (-2.3, 4.65, -4.43), (2.3, 4.65, -4.43),
        (2.3, 5.05, -4.43), (-2.3, 5.05, -4.43),
    ), Material(
        base_color=(1.0, .82, .55), emission=(7.0, 3.5, 1.2),
        emission_two_sided=True, roughness=.2,
    ), "warm-light-strip")


def _base_scene(*, environment=True):
    scene = Scene()
    scene.add_light(SpotLight(
        # Keep the key light across the subjects from the default camera.  A
        # camera-side light hides nearly all of its cast shadows behind the
        # optical samples, which made the showcase appear unshadowed even
        # though its native shadow pass was active.
        position=(-4.0, 5.5, -2.5), direction=(.62, -.70, .39),
        color=(1.0, .78, .54), intensity=95.0,
        inner_cone_angle=np.deg2rad(22.0),
        outer_cone_angle=np.deg2rad(34.0), range=16.0,
    ))
    if environment:
        scene.set_environment(EnvironmentLight(
            image=_environment(), intensity=1.25, rotation=0.2,
        ))
    _add_diorama(scene)
    return scene


def _add_sphere(scene, center, radius, material, name):
    vertices, indices = sphere(center, radius, rings=28, segments=56)
    return scene.add_mesh(vertices, indices, material, name=name)


def build_environment_reflection_scene():
    scene = _base_scene()
    for index, roughness in enumerate((.025, .16, .42)):
        _add_sphere(scene, ((index - 1) * 2.6, 1.15, 0), 1.08, Material(
            base_color=(.72, .76, .82), metallic=1.0, roughness=roughness,
        ), f"reflection-{index}")
    return scene


def build_reflection_probe_scene():
    scene = _base_scene(environment=False)
    scene.add_reflection_probe(ReflectionProbe(
        _environment(), position=(0, 1, 0), radius=12.0,
        intensity=1.4, rotation=-.35,
    ))
    _add_sphere(scene, (0, 1.2, 0), 1.15, Material(
        base_color=(.82, .82, .82), metallic=1.0, roughness=.07,
    ), "probe-reflection")
    return scene


def build_refraction_scene():
    scene = _base_scene()
    for index, ior in enumerate((1.1, 1.33, 1.52)):
        _add_sphere(scene, ((index - 1) * 2.6, 1.2, 0), 1.12, Material(
            base_color=(1, 1, 1), transmission=1.0, roughness=.025,
            ior=ior, thickness=2.0,
        ), f"refraction-{index}")
    return scene


def build_absorption_scene():
    scene = _base_scene()
    for index, distance in enumerate((.35, 1.0, 3.0)):
        _add_sphere(scene, ((index - 1) * 2.6, 1.2, 0), 1.12, Material(
            base_color=(1, 1, 1), transmission=1.0, roughness=.02,
            ior=1.5, attenuation_color=(.08, .55, .92),
            attenuation_distance=distance, thickness=2.0,
        ), f"absorption-{index}")
    return scene


def build_nested_dielectric_scene():
    scene = _base_scene()
    _add_sphere(scene, (0, 1.45, 0), 1.4, Material(
        base_color=(1, 1, 1), transmission=1.0, roughness=.015,
        ior=1.52, attenuation_color=(.72, .92, 1.0),
        attenuation_distance=3.0, thickness=2.8,
    ), "outer-glass")
    _add_sphere(scene, (0, 1.45, 0), .72, Material(
        base_color=(1, 1, 1), transmission=1.0, roughness=.01,
        ior=1.33, attenuation_color=(1.0, .45, .18),
        attenuation_distance=1.2, thickness=1.44,
    ), "inner-liquid")
    return scene


def build_transparency_scene():
    scene = _base_scene()
    colors = ((1, .12, .08), (.1, .9, .25), (.12, .3, 1.0))
    for index, color in enumerate(colors):
        z = (index - 1) * .65
        vertices, indices = quad(
            (-2.5 + index * .65, .2, z), (1.0 + index * .65, .2, z),
            (1.0 + index * .65, 3.2, z), (-2.5 + index * .65, 3.2, z),
        )
        scene.add_mesh(vertices, indices, Material(
            base_color=color, roughness=.55, opacity=.42,
            alpha_mode="blend",
        ), name=f"transparent-layer-{index}")
    return scene


__all__ = [
    "build_absorption_scene", "build_environment_reflection_scene",
    "build_nested_dielectric_scene", "build_reflection_probe_scene",
    "build_refraction_scene", "build_transparency_scene",
]
