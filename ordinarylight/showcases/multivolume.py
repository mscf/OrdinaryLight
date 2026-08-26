"""Reusable showcase scenes for overlapping structured volumes."""

import numpy as np

import ordinarylight as ol


def _cloud_density(resolution, phase, shell_radius, center):
    coordinates = np.linspace(-1.0, 1.0, resolution, dtype=np.float32)
    z, y, x = np.meshgrid(coordinates, coordinates, coordinates, indexing="ij")
    cx, cy, cz = center
    radial = np.sqrt(
        ((x - cx) * 0.92) ** 2
        + ((y - cy) * 1.08) ** 2
        + ((z - cz) * 0.96) ** 2
    )
    shell = np.exp(-((radial - shell_radius) / 0.16) ** 2)
    core = np.exp(-(radial / 0.52) ** 4)
    structure = 0.62 + 0.38 * (
        np.sin(8.0 * x + 5.0 * z + phase)
        * np.cos(7.0 * y - 3.0 * z + phase * 0.7)
    )
    density = np.clip((0.68 * shell + 0.55 * core) * structure, 0.0, 1.0)

    # Explicitly transparent support prevents the transformed proxy boundary
    # from becoming visible when the media overlap other geometry.
    boundary_distance = 1.0 - np.maximum.reduce((np.abs(x), np.abs(y), np.abs(z)))
    fade = np.clip(boundary_distance / 0.14, 0.0, 1.0)
    fade = fade * fade * (3.0 - 2.0 * fade)
    return np.asarray(density * fade, np.float32)


def build_multivolume_showcase(resolution=56):
    """Return three partially overlapping emissive--absorbing media."""
    scene = ol.Scene()
    media = (
        (
            "cyan-cloud", (0.0, 0.0, 0.0), 0.54,
            (-1.75, 0.15, -1.05), (2.7, 2.7, 2.7),
            ((0.00, 0.00, 0.00, 0.000),
             (0.02, 0.20, 0.65, 0.025),
             (0.08, 1.20, 2.80, 0.120),
             (0.55, 3.20, 4.20, 0.320)),
        ),
        (
            "magenta-cloud", (0.12, -0.08, 0.02), 0.61,
            (-0.45, 0.25, -0.85), (2.45, 2.45, 2.45),
            ((0.00, 0.00, 0.00, 0.000),
             (0.35, 0.02, 0.28, 0.025),
             (2.20, 0.12, 1.45, 0.130),
             (4.00, 0.75, 2.50, 0.340)),
        ),
        (
            "amber-cloud", (-0.10, 0.10, -0.08), 0.48,
            (-0.95, 0.05, -1.85), (2.6, 2.6, 2.6),
            ((0.00, 0.00, 0.00, 0.000),
             (0.45, 0.10, 0.01, 0.020),
             (2.60, 0.75, 0.06, 0.110),
             (4.80, 2.20, 0.35, 0.300)),
        ),
    )
    for index, (
        name, center, shell_radius, translation, scale, transfer,
    ) in enumerate(media):
        scene.add_volume(
            _cloud_density(resolution, index * 1.7, shell_radius, center),
            ol.VolumeMaterial(
                ol.Texture1D(np.asarray(transfer, np.float32)),
                density_scale=1.0,
                emission_scale=1.15,
                step_size=0.022,
            ),
            transform=(
                ol.Transform.translation(translation)
                @ ol.Transform.scale(scale)
            ),
            value_range=(0.0, 1.0),
            name=name,
            metadata={"role": "overlapping-volume-showcase"},
        )

    floor = np.asarray((
        (-5.0, 0.0, -5.0), (5.0, 0.0, -5.0),
        (5.0, 0.0, 5.0), (-5.0, 0.0, 5.0),
    ), np.float32)
    scene.add_mesh(
        floor, ((0, 1, 2), (0, 2, 3)),
        ol.Material(base_color=(0.12, 0.15, 0.22), roughness=0.72),
        name="floor",
    )
    scene.add_points(
        ((-2.35, 0.22, 1.15), (2.15, 0.30, 0.85)),
        radii=(0.22, 0.30),
        materials=(
            ol.Material(
                base_color=(0.08, 0.35, 0.95), metallic=0.75, roughness=0.16,
            ),
            ol.Material(
                base_color=(0.95, 0.22, 0.05), metallic=0.35, roughness=0.24,
            ),
        ),
        names=("blue-reference", "orange-reference"),
    )
    emitter = np.asarray((
        (-1.8, 5.0, -0.8), (1.8, 5.0, -0.8),
        (1.8, 5.0, 0.8), (-1.8, 5.0, 0.8),
    ), np.float32)
    scene.add_mesh(
        emitter, ((0, 1, 2), (0, 2, 3)),
        ol.Material(emission=(7.0, 8.0, 11.0), emission_two_sided=True),
        name="area-light",
    )
    scene.add_point_light(
        (0.0, 4.2, 2.0), color=(0.55, 0.72, 1.0), intensity=42.0,
    )
    return scene
