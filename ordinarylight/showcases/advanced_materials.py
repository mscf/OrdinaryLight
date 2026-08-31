"""Focused portable scenes for advanced material-lobe validation."""

from __future__ import annotations

from ..lights import DirectionalLight, PointLight
from ..scene import Material, Scene
from .materials import sphere


def _scene(materials):
    scene = Scene()
    scene.add_light(DirectionalLight(
        direction=(0.35, -1.0, -0.25), color=(0.8, 0.88, 1.0),
        intensity=2.2,
    ))
    scene.add_light(PointLight(
        position=(-3.0, 4.5, 4.0), color=(1.0, 0.72, 0.48), intensity=85.0,
    ))
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
    return _scene(tuple(Material(
        base_color=(0.72, 0.08, 0.035), roughness=0.55,
        clearcoat=weight, clearcoat_roughness=roughness,
    ) for weight, roughness in ((0.0, 0.1), (0.5, 0.18), (1.0, 0.04))))


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
    return _scene(tuple(Material(
        base_color=(0.78, 0.52, 0.16), metallic=1.0, roughness=0.28,
        anisotropy=value,
    ) for value in (-0.85, 0.0, 0.85)))


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
