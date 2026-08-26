"""Volume rendering showcases."""

from ordinarylight.integrations.workbench import Showcase

from ordinarylight.showcases.multivolume import build_multivolume_showcase
from ordinarylight.showcases.volume_multiple_scattering import build_volume_multiple_scattering_showcase
from ordinarylight.showcases.volume_scattering import build_volume_scattering_showcase
from ordinarylight.showcases.volumes import build_volume_showcase


SHOWCASES = (
    Showcase("volume", "Volume transfer function", build_volume_showcase,
             tags=("volumes",)),
    Showcase("multi-volume", "Multiple volumes", build_multivolume_showcase,
             tags=("volumes", "composition")),
    Showcase("volume-scattering", "Volume scattering", build_volume_scattering_showcase,
             tags=("volumes", "scattering")),
    Showcase(
        "volume-multiple-scattering", "Volume multiple scattering",
        build_volume_multiple_scattering_showcase,
        tags=("volumes", "scattering", "stress"),
    ),
)
