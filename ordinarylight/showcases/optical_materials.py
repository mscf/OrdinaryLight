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


def _diorama_radiance_probe(position=(0.0, 1.2, 0.0), width=512, height=256):
    """Bake the static opaque diorama into an equirectangular local probe."""
    u = (np.arange(width, dtype=np.float32) + 0.5) / width
    v = (np.arange(height, dtype=np.float32) + 0.5) / height
    phi = (u[None, :] - 0.5) * (2.0 * np.pi)
    theta = v[:, None] * np.pi
    directions = np.empty((height, width, 3), np.float32)
    directions[..., 0] = np.cos(phi) * np.sin(theta)
    directions[..., 1] = np.cos(theta)
    directions[..., 2] = np.sin(phi) * np.sin(theta)
    origin = np.asarray(position, np.float32)
    radiance = np.broadcast_to(_environment(width, height),
                               (height, width, 3)).copy()
    nearest = np.full((height, width), np.inf, np.float32)

    # Floor plane and its authored 16-cell checker texture.
    floor_t = -origin[1] / np.where(
        np.abs(directions[..., 1]) > 1e-6, directions[..., 1], np.nan,
    )
    floor_x = origin[0] + directions[..., 0] * floor_t
    floor_z = origin[2] + directions[..., 2] * floor_t
    floor_hit = ((floor_t > 0.0) & (np.abs(floor_x) <= 6.5)
                 & (np.abs(floor_z) <= 4.5))
    floor_cells = ((np.floor((floor_x + 6.5) / 13.0 * 16.0)
                    + np.floor((floor_z + 4.5) / 9.0 * 16.0)) % 2) > 0
    floor_colors = np.where(
        floor_cells[..., None],
        np.asarray((42, 53, 72), np.float32) / 255.0,
        np.asarray((220, 223, 216), np.float32) / 255.0,
    )
    radiance[floor_hit] = floor_colors[floor_hit] * 3.5
    nearest[floor_hit] = floor_t[floor_hit]

    # Back wall, including its 30-cell comparison pattern and light strip.
    wall_t = (-4.5 - origin[2]) / np.where(
        np.abs(directions[..., 2]) > 1e-6, directions[..., 2], np.nan,
    )
    wall_x = origin[0] + directions[..., 0] * wall_t
    wall_y = origin[1] + directions[..., 1] * wall_t
    wall_hit = ((wall_t > 0.0) & (wall_t < nearest)
                & (np.abs(wall_x) <= 6.5) & (wall_y >= 0.0)
                & (wall_y <= 5.8))
    wall_cells = ((np.floor((wall_x + 6.5) / 13.0 * 30.0)
                   + np.floor(wall_y / 5.8 * 30.0)) % 2) > 0
    wall_colors = np.where(
        wall_cells[..., None],
        np.asarray((224, 116, 48), np.float32) / 255.0,
        np.asarray((28, 72, 112), np.float32) / 255.0,
    )
    radiance[wall_hit] = wall_colors[wall_hit] * 3.5
    strip = (wall_hit & (np.abs(wall_x) <= 2.3)
             & (wall_y >= 4.65) & (wall_y <= 5.05))
    radiance[strip] = (7.0, 3.5, 1.2)
    return radiance


def _add_quad(scene, corners, material, name, texcoords=None):
    vertices, indices = quad(*corners)
    return scene.add_mesh(
        vertices, indices, material, name=name,
        texcoords=None if texcoords is None else np.asarray(texcoords, np.float32),
    )


def _add_diorama(scene, *, reference_panels=True, wall_cells=10):
    floor_texture = _checker_texture((220, 223, 216), (42, 53, 72), cells=16)
    wall_texture = _checker_texture(
        (28, 72, 112), (224, 116, 48), cells=wall_cells,
    )
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
    if reference_panels:
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


def _base_scene(*, environment=True, reference_panels=True, wall_cells=10):
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
    _add_diorama(
        scene, reference_panels=reference_panels, wall_cells=wall_cells,
    )
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


def build_automatic_probe_scene():
    """A room with two renderer-captured, overlapping box probes."""
    scene = _base_scene(environment=False)
    scene.add_reflection_probe(ReflectionProbe(
        # Keep capture origins in free space.  The former origins coincided
        # with the outer demonstration spheres, so the cubemap captured the
        # inside of those spheres instead of the surrounding room.
        position=(-2.8, 2.4, 2.2), radius=6.0,
        projection="box", box_min=(-5.0, 0.0, -3.0),
        box_max=(0.6, 4.5, 3.0), blend_distance=2.5,
        refresh_policy="scene-change", capture_resolution=128,
    ))
    scene.add_reflection_probe(ReflectionProbe(
        position=(2.8, 2.4, 2.2), radius=6.0,
        projection="box", box_min=(-0.6, 0.0, -3.0),
        box_max=(5.0, 4.5, 3.0), blend_distance=2.5,
        refresh_policy="scene-change", capture_resolution=128,
    ))
    for index, x in enumerate((-2.4, 0.0, 2.4)):
        _add_sphere(scene, (x, 1.15, 0), 1.08, Material(
            base_color=(.72, .76, .82), metallic=1.0,
            roughness=(.04, .16, .32)[index],
        ), f"automatic-probe-{index}")
    return scene


def build_refraction_scene():
    # A common checkerboard receiver makes the three samples an IOR
    # comparison. Colored panels behind only the outside subjects confounded
    # refraction strength with background luminance and made those materials
    # appear to lose energy in both raster and GI views.
    scene = _base_scene(reference_panels=False, wall_cells=30)
    scene.add_reflection_probe(ReflectionProbe(
        _diorama_radiance_probe(), position=(0.0, 1.2, 0.0),
        radius=8.0, intensity=1.0,
    ))
    for index, ior in enumerate((1.1, 1.33, 1.52)):
        _add_sphere(scene, ((index - 1) * 2.6, 1.2, 0), 1.12, Material(
            base_color=(1, 1, 1), transmission=1.0, roughness=.025,
            # The closed spheres have radius 1.12.  Keep the authored optical
            # thickness equal to their diameter so the raster two-interface
            # proxy and the GI medium describe the same physical boundary.
            ior=ior, thickness=2.24,
        ), f"refraction-{index}")
    return scene


def build_absorption_scene():
    scene = _base_scene()
    # Make the dielectric interface legible in the raster approximation as
    # well as the attenuation through the medium.  The GI target naturally
    # sees this same authored room through traced reflection rays.
    scene.add_reflection_probe(ReflectionProbe(
        _diorama_radiance_probe(), position=(0.0, 1.2, 0.0),
        radius=8.0, intensity=1.0,
    ))
    for index, distance in enumerate((.35, 1.0, 3.0)):
        _add_sphere(scene, ((index - 1) * 2.6, 1.2, 0), 1.12, Material(
            base_color=(1, 1, 1), transmission=1.0, roughness=.02,
            ior=1.5, attenuation_color=(.08, .55, .92),
            attenuation_distance=distance, thickness=2.0,
        ), f"absorption-{index}")
    return scene


def build_nested_dielectric_scene():
    scene = _base_scene()
    # Supply the portable raster target with the same room radiance that the
    # GI target reaches with reflection rays.  Screen-space transmission still
    # carries the nested interfaces; this probe only supplies reflection and
    # off-screen fallback energy.
    scene.add_reflection_probe(ReflectionProbe(
        _diorama_radiance_probe(), position=(0.0, 1.45, 0.0),
        radius=8.0, intensity=1.0,
    ))
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
    "build_automatic_probe_scene",
    "build_refraction_scene", "build_transparency_scene",
]
