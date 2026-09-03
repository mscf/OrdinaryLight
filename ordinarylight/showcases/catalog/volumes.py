"""Volume rendering showcases."""

from ordinarylight.integrations.workbench import OrbitCamera, Showcase

from ordinarylight.showcases.multivolume import build_multivolume_showcase
from ordinarylight.showcases.volume_multiple_scattering import build_volume_multiple_scattering_showcase
from ordinarylight.showcases.volume_scattering import build_volume_scattering_showcase
from ordinarylight.showcases.volumes import build_volume_showcase
from ordinarylight.showcases.scientific import build_scientific_scalar_field_scene


VOLUME_CAMERA = OrbitCamera(
    target=(-0.1, 1.45, -0.7), radius=8.2, height=3.1,
    arc_radians=0.42,
)


SHOWCASES = (
    Showcase(
        "scientific-scalar-field", "Scientific scalar field",
        build_scientific_scalar_field_scene,
        description=(
            "One coordinate-aware field drives a clipped volume, three "
            "orthogonal slices, and an isosurface through a shared transfer "
            "function."
        ),
        camera=OrbitCamera(target=(0.0, 0.0, 0.0), radius=4.8, height=2.3),
        renderer={
            "volume_rendering": "ray-march", "volume_step_scale": 1.0,
            "volume_max_steps": 1024,
        },
        tags=("volumes", "scientific", "raster-feature"),
    ),
    Showcase(
        "volume", "Volumes: transfer function", build_volume_showcase,
        description=(
            "An emissive structured volume demonstrates transfer-function "
            "color, absorption, and composition with opaque geometry."
        ),
        camera=VOLUME_CAMERA,
        renderer={
            "volume_rendering": "ray-march",
            "volume_step_scale": 1.0,
            "volume_max_steps": 1024,
        },
        tags=("volumes", "gi-feature", "raster-feature"),
    ),
    Showcase(
        "multi-volume", "Volumes: multiple media", build_multivolume_showcase,
        description=(
            "Overlapping structured media demonstrate order-independent "
            "volume composition."
        ),
        camera=VOLUME_CAMERA,
        renderer={
            "volume_rendering": "ray-march",
            "volume_step_scale": 1.0,
            "volume_max_steps": 1024,
        },
        tags=("volumes", "composition", "gi-feature", "raster-feature"),
    ),
    Showcase(
        "volume-scattering", "Volumes: scattering",
        build_volume_scattering_showcase,
        description=(
            "A heterogeneous medium demonstrates light-dependent "
            "Henyey-Greenstein scattering and an embedded opaque object's "
            "volumetric shadow."
        ),
        camera=VOLUME_CAMERA,
        renderer={
            "volume_rendering": "ray-march", "volume_step_scale": 1.0,
            "volume_max_steps": 1024,
        },
        tags=("volumes", "scattering", "gi-feature", "raster-feature"),
    ),
    Showcase(
        "volume-multiple-scattering", "Volumes: multiple scattering",
        build_volume_multiple_scattering_showcase,
        description=(
            "Bounded higher-order in-scattering exercises volume transport "
            "and opaque volumetric shadows across the GI and raster targets."
        ),
        camera=VOLUME_CAMERA,
        renderer={
            "volume_rendering": "ray-march", "volume_step_scale": 1.0,
            "volume_max_steps": 1024,
        },
        tags=("volumes", "scattering", "stress", "gi-feature",
              "raster-feature"),
    ),
)
