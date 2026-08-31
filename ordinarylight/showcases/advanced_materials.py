"""Focused portable scenes for advanced material-lobe validation."""

from __future__ import annotations

import numpy as np

from ..lights import DirectionalLight, EnvironmentLight, PointLight, SpotLight
from ..scene import Material, Scene
from .materials import sphere


def _scene(materials, *, lights=None):
    scene = Scene()
    if lights is None:
        lights = (
            DirectionalLight(
                direction=(0.35, -1.0, -0.25), color=(0.8, 0.88, 1.0),
                intensity=2.2,
            ),
            PointLight(
                position=(-3.0, 4.5, 4.0), color=(1.0, 0.72, 0.48),
                intensity=85.0,
            ),
        )
    for light in lights:
        scene.add_light(light)
    scene.add_mesh(
        ((-8, 0, -5), (8, 0, -5), (8, 0, 5), (-8, 0, 5)),
        ((0, 2, 1), (0, 3, 2)),
        Material(base_color=(0.46, 0.5, 0.56), roughness=0.72),
    )
    count = len(materials)
    for index, material in enumerate(materials):
        x = (index - (count - 1) * 0.5) * 2.5
        vertices, indices = sphere((x, 1.15, 0.0), 1.08, rings=24, segments=48)
        scene.add_mesh(vertices, indices, material, name=f"material-{index}")
    return scene


def build_clearcoat_scene():
    materials = tuple(Material(
        base_color=(0.72, 0.08, 0.035), roughness=0.55,
        clearcoat=weight, clearcoat_roughness=roughness,
    ) for weight, roughness in ((0.0, 0.1), (0.5, 0.18), (1.0, 0.04)))
    # A point light requires a cube shadow map, which is not part of the first
    # strict-raster shadow tier.  Keep this material comparison on one broad,
    # shadow-capable key that both GI and raster evaluate identically.  It
    # preserves the intended warm studio light while making inter-object
    # occlusion visible through the shared 2D spot-shadow path.
    return _scene(materials, lights=(SpotLight(
        position=(-3.0, 4.5, 4.0), direction=(3.0, -3.35, -4.0),
        color=(1.0, 0.76, 0.56), intensity=105.0,
        inner_cone_angle=0.72, outer_cone_angle=1.08, range=20.0,
    ),))


def build_sheen_scene():
    return _scene(tuple(Material(
        base_color=(0.12, 0.08, 0.2), roughness=0.7,
        sheen_color=color, sheen_roughness=roughness,
    ) for color, roughness in (
        ((0.0, 0.0, 0.0), 0.5),
        ((0.75, 0.08, 0.3), 0.5),
        ((0.15, 0.45, 1.0), 0.12),
    )))


def build_anisotropy_scene():
    scene = _scene(tuple(Material(
        base_color=(0.78, 0.52, 0.16), metallic=1.0, roughness=0.28,
        anisotropy=value,
    ) for value in (-0.85, 0.0, 0.85)))
    # Metallic anisotropy is only fully characterized when there is broad
    # incident radiance for the elongated lobe to reflect. GI can obtain that
    # radiance from indirect scene paths, while strict rasterization cannot;
    # an explicit studio environment gives both targets the same input.
    width, height = 256, 128
    u = (np.arange(width, dtype=np.float32) + 0.5) / width
    v = (np.arange(height, dtype=np.float32) + 0.5) / height
    uu, vv = np.meshgrid(u, v)
    environment = np.empty((height, width, 3), np.float32)
    horizon = np.exp(-((vv - 0.48) / 0.18) ** 2)
    environment[..., 0] = 0.045 + 0.18 * horizon
    environment[..., 1] = 0.055 + 0.20 * horizon
    environment[..., 2] = 0.075 + 0.24 * horizon
    key = np.exp(-(
        ((uu - 0.20) / 0.055) ** 2 + ((vv - 0.30) / 0.12) ** 2
    ))
    rim = np.exp(-(
        ((uu - 0.72) / 0.14) ** 2 + ((vv - 0.42) / 0.055) ** 2
    ))
    environment += key[..., None] * np.asarray((7.0, 6.4, 5.4), np.float32)
    environment += rim[..., None] * np.asarray((1.2, 1.7, 2.5), np.float32)
    scene.set_environment(EnvironmentLight(
        image=environment, intensity=1.15, rotation=-0.18,
    ))
    return scene


def build_thin_transmission_scene():
    return _scene((
        Material(base_color=(0.75, 0.9, 1.0), transmission=1.0,
                 roughness=0.06, ior=1.5, thickness=2.16),
        Material(base_color=(0.75, 0.9, 1.0), transmission=1.0,
                 roughness=0.06, ior=1.5, thin_walled=True),
    ))


def build_subsurface_scene():
    return _scene(tuple(Material(
        base_color=(0.62, 0.12, 0.08), roughness=0.62,
        subsurface=weight, subsurface_color=(1.0, 0.22, 0.12),
        subsurface_radius=radius,
    ) for weight, radius in ((0.0, 0.2), (0.55, 0.45), (1.0, 0.8))))


__all__ = [
    "build_anisotropy_scene", "build_clearcoat_scene", "build_sheen_scene",
    "build_subsurface_scene", "build_thin_transmission_scene",
]
