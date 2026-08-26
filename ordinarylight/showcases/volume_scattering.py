"""Reusable light-dependent volume-scattering showcase scenes."""

import numpy as np

import ordinarylight as ol


def build_volume_scattering_showcase(
    resolution=64, *, scattering_orders=1,
    scattering_albedo=(0.9, 0.9, 0.9),
):
    scene = ol.Scene()
    coordinates = np.linspace(-1.0, 1.0, resolution, dtype=np.float32)
    z, y, x = np.meshgrid(coordinates, coordinates, coordinates, indexing="ij")
    radial = np.sqrt((0.9 * x) ** 2 + (1.1 * y) ** 2 + z ** 2)
    billows = 0.72 + 0.28 * np.sin(9.0 * x + 4.0 * z) * np.cos(7.0 * y)
    density = np.clip(np.exp(-((radial / 0.72) ** 4)) * billows, 0.0, 1.0)
    boundary = 1.0 - np.maximum.reduce((np.abs(x), np.abs(y), np.abs(z)))
    fade = np.clip(boundary / 0.16, 0.0, 1.0)
    density *= fade * fade * (3.0 - 2.0 * fade)

    transfer = ol.Texture1D(np.asarray((
        (0.0, 0.0, 0.0, 0.000),
        (0.0, 0.0, 0.0, 0.025),
        (0.0, 0.0, 0.0, 0.090),
        (0.0, 0.0, 0.0, 0.220),
    ), np.float32))
    scene.add_volume(
        density,
        ol.VolumeMaterial(
            transfer, density_scale=1.0, emission_scale=0.0,
            step_size=0.024, scattering_scale=0.92,
            scattering_color=(0.55, 0.72, 1.0),
            phase_function="henyey_greenstein", anisotropy=0.55,
            scattering_albedo=scattering_albedo,
            scattering_orders=scattering_orders,
        ),
        transform=(
            ol.Transform.translation((-1.65, 0.12, -1.45))
            @ ol.Transform.scale((3.3, 3.3, 3.3))
        ),
        value_range=(0.0, 1.0), name="forward-scattering-cloud",
        metadata={
            "role": (
                "single-scattering-showcase" if scattering_orders == 1
                else "bounded-multiple-scattering-showcase"
            ),
        },
    )

    floor = np.asarray((
        (-5.0, 0.0, -5.0), (5.0, 0.0, -5.0),
        (5.0, 0.0, 5.0), (-5.0, 0.0, 5.0),
    ), np.float32)
    scene.add_mesh(
        floor, ((0, 1, 2), (0, 2, 3)),
        ol.Material(base_color=(0.11, 0.14, 0.20), roughness=0.72),
        name="floor",
    )
    scene.add_points(
        ((-2.0, 0.25, 1.25), (1.9, 0.32, 0.9)), radii=(0.25, 0.32),
        materials=(
            ol.Material(base_color=(0.08, 0.28, 0.8), metallic=0.5),
            ol.Material(base_color=(0.9, 0.22, 0.05), roughness=0.3),
        ),
        names=("blue-reference", "orange-reference"),
    )
    # A warm side light reveals the phase response; a dim frontal fill keeps
    # the opaque references readable while orbiting.
    scene.add_point_light((-2.6, 2.7, -0.4), color=(1.0, 0.48, 0.16), intensity=52.0)
    scene.add_point_light((2.8, 3.4, 2.0), color=(0.25, 0.48, 1.0), intensity=28.0)
    return scene
