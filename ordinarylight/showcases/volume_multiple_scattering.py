"""Native Vulkan showcase for bounded higher-order volume scattering."""

from .volume_scattering import build_volume_scattering_showcase


def build_volume_multiple_scattering_showcase(resolution=64):
    return build_volume_scattering_showcase(
        resolution,
        scattering_orders=4,
        scattering_albedo=(0.92, 0.94, 0.98),
    )
