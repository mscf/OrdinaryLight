"""Reusable structured-volume and mesh-composition showcase scenes."""

import numpy as np

import ordinarylight as ol


def build_volume_showcase(resolution=80, *, reference_geometry=True):
    scene = ol.Scene()
    coordinates = np.linspace(-1.0, 1.0, resolution, dtype=np.float32)
    z, y, x = np.meshgrid(coordinates, coordinates, coordinates, indexing="ij")
    radial = np.sqrt((x * 0.92) ** 2 + (y * 1.15) ** 2 + z ** 2)
    shell = np.exp(-((radial - 0.58) / 0.12) ** 2)
    core = np.exp(-((x + 0.20) ** 2 + (y - 0.05) ** 2
                    + (z + 0.10) ** 2) / 0.10)
    detail = 0.5 + 0.5 * np.sin(11.0 * x + 7.0 * z) * np.cos(9.0 * y)
    density = np.clip(shell * (0.50 + 0.50 * detail) + 0.85 * core, 0.0, 1.0)
    # The analytic lobes have a small but nonzero tail at the finite grid
    # boundary.  Since the transfer function maps every positive value to
    # some opacity, abruptly truncating that tail reveals the volume's box.
    # Taper only the outer 12% so the synthetic field has transparent support
    # without changing its interior structure.
    boundary_distance = 1.0 - np.maximum.reduce((np.abs(x), np.abs(y), np.abs(z)))
    boundary_fade = np.clip(boundary_distance / 0.12, 0.0, 1.0)
    boundary_fade = boundary_fade * boundary_fade * (3.0 - 2.0 * boundary_fade)
    density *= boundary_fade

    transfer = ol.Texture1D(np.asarray((
        (0.00, 0.00, 0.00, 0.000),
        (0.03, 0.08, 0.25, 0.010),
        (0.08, 0.55, 1.20, 0.045),
        (0.65, 0.20, 1.40, 0.100),
        (2.80, 0.55, 0.08, 0.260),
        (5.00, 2.30, 0.65, 0.500),
    ), np.float32))
    scene.add_volume(
        density,
        ol.VolumeMaterial(
            transfer, density_scale=1.0, emission_scale=1.35,
            step_size=0.018,
        ),
        transform=(
            ol.Transform.translation((-1.55, 0.15, -1.55))
            @ ol.Transform.scale((3.1, 3.1, 3.1))
        ),
        value_range=(0.0, 1.0), name="emissive-nebula",
        metadata={"role": "structured-volume-showcase"},
    )

    if not reference_geometry:
        return scene

    floor = np.asarray((
        (-5, 0, -5), (5, 0, -5), (5, 0, 5), (-5, 0, 5),
    ), np.float32)
    scene.add_mesh(
        floor, ((0, 1, 2), (0, 2, 3)),
        ol.Material(base_color=(0.18, 0.21, 0.28), roughness=0.65),
        name="floor",
    )
    emitter = np.asarray((
        (-2.0, 5.0, -1.0), (2.0, 5.0, -1.0),
        (2.0, 5.0, 1.0), (-2.0, 5.0, 1.0),
    ), np.float32)
    scene.add_mesh(
        emitter, ((0, 1, 2), (0, 2, 3)),
        ol.Material(
            emission=(8.0, 9.0, 12.0), emission_two_sided=True,
        ), name="area-light",
    )
    scene.add_points(
        ((-2.0, 0.22, 1.25), (1.8, 0.32, 1.0)),
        radii=(0.22, 0.32),
        materials=(
            ol.Material(base_color=(0.12, 0.35, 0.8), metallic=0.65, roughness=0.18),
            ol.Material(base_color=(0.9, 0.25, 0.08), metallic=0.2, roughness=0.28),
        ),
        names=("blue-marker", "orange-marker"),
    )
    scene.add_point_light((0.0, 4.5, 1.5), color=(0.7, 0.8, 1.0), intensity=55.0)
    return scene
