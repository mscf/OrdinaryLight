"""Emissive area-light and multiple-importance-sampling showcase."""

import ordinarylight as ol

from .materials import diffuse, fresnel_glass, mirror, quad, sphere


def build_area_light_showcase():
    scene = ol.Scene()
    surfaces = (
        (((-6, 0, -5), (6, 0, -5), (6, 0, 5), (-6, 0, 5)), (0.68, 0.70, 0.74)),
        (((-6, 0, 5), (6, 0, 5), (6, 7, 5), (-6, 7, 5)), (0.72, 0.73, 0.76)),
        (((-6, 0, -5), (-6, 0, 5), (-6, 7, 5), (-6, 7, -5)), (0.62, 0.16, 0.12)),
        (((6, 0, 5), (6, 0, -5), (6, 7, -5), (6, 7, 5)), (0.12, 0.25, 0.62)),
    )
    for corners, color in surfaces:
        vertices, indices = quad(*corners)
        scene.add_mesh(vertices, indices, ol.Material(base_color=color, program=diffuse))

    # Nonzero material emission makes these visible meshes sampleable lights.
    vertices, indices = quad(
        (-2.5, 6.7, -1.4), (0.2, 6.7, -1.4),
        (0.2, 6.7, 1.4), (-2.5, 6.7, 1.4),
    )
    scene.add_mesh(vertices, indices, ol.Material(
        base_color=(1.0, 0.86, 0.68), emission=(14.0, 8.0, 4.0), program=diffuse,
    ))
    vertices, indices = quad(
        (1.0, 4.8, 4.85), (1.0, 6.2, 4.85),
        (4.1, 6.2, 4.85), (4.1, 4.8, 4.85),
    )
    scene.add_mesh(vertices, indices, ol.Material(
        base_color=(0.65, 0.82, 1.0), emission=(3.0, 7.0, 14.0), program=diffuse,
    ))

    vertices, indices = sphere((-2.0, 1.35, 0.4), 1.35)
    scene.add_mesh(vertices, indices, ol.Material(
        base_color=(0.92, 0.72, 0.22), metallic=1.0, program=mirror,
    ))
    vertices, indices = sphere((2.0, 1.5, 0.0), 1.5)
    scene.add_mesh(vertices, indices, ol.Material(
        base_color=(0.88, 0.96, 1.0), transmission=1.0, ior=1.52,
        program=fresnel_glass,
    ))
    return scene
