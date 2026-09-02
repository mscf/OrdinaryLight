"""Volume rendering showcases."""

from ordinarylight.integrations.workbench import OrbitCamera, Showcase

from ordinarylight.showcases.multivolume import build_multivolume_showcase
from ordinarylight.showcases.volume_multiple_scattering import build_volume_multiple_scattering_showcase
from ordinarylight.showcases.volume_scattering import build_volume_scattering_showcase
from ordinarylight.showcases.volumes import build_volume_showcase


VOLUME_CAMERA = OrbitCamera(
    target=(-0.1, 1.45, -0.7), radius=8.2, height=3.1,
    arc_radians=0.42,
)


SHOWCASES = (
    Showcase(
        "volume", "Volumes: transfer function", build_volume_showcase,
        description=(
            "An emissive structured volume demonstrates transfer-function "
            "color, absorption, and composition with opaque geometry."
        ),
        camera=VOLUME_CAMERA,
        renderer={"volume_slices": 128},
        tags=("volumes", "gi-feature", "raster-feature"),
    ),
    Showcase(
        "multi-volume", "Volumes: multiple media", build_multivolume_showcase,
        description=(
            "Overlapping structured media demonstrate order-independent "
            "volume composition."
        ),
        camera=VOLUME_CAMERA,
        renderer={"volume_slices": 128},
        tags=("volumes", "composition", "gi-feature", "raster-feature"),
    ),
    Showcase(
        "volume-scattering", "Volumes: scattering",
        build_volume_scattering_showcase,
        description=(
            "A heterogeneous medium demonstrates light-dependent "
            "Henyey-Greenstein scattering."
        ),
        camera=VOLUME_CAMERA,
        renderer={"volume_slices": 128},
        tags=("volumes", "scattering", "gi-feature", "raster-feature"),
    ),
    Showcase(
        "volume-multiple-scattering", "Volumes: multiple scattering",
        build_volume_multiple_scattering_showcase,
        description=(
            "Bounded higher-order in-scattering exercises volume transport "
            "across the GI and raster targets."
        ),
        camera=VOLUME_CAMERA,
        renderer={"volume_slices": 128},
        tags=("volumes", "scattering", "stress", "gi-feature",
              "raster-feature"),
    ),
)
