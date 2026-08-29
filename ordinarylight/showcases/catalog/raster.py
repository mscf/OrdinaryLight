"""Portable raster feature demonstrations."""

from ordinarylight.integrations.workbench import OrbitCamera, Showcase
from ordinarylight.showcases.raster_features import (
    build_directional_shadow_scene, build_spot_shadow_scene,
)


_CAMERA = OrbitCamera(target=(0.0, 0.9, 0.0), radius=8.5, height=4.2)

SHOWCASES = (
    Showcase(
        "raster-directional-shadows", "Raster: directional shadows",
        build_directional_shadow_scene,
        description="Native directional shadow-map generation and sampling.",
        camera=_CAMERA,
        renderer={"shadows": True, "shadow_map_size": 512},
        tags=("raster-feature", "lighting", "shadows"),
    ),
    Showcase(
        "raster-spot-shadows", "Raster: spot shadows",
        build_spot_shadow_scene,
        description="Native perspective shadow map for a cone light.",
        camera=_CAMERA,
        renderer={"shadows": True, "shadow_map_size": 512},
        tags=("raster-feature", "lighting", "shadows"),
    ),
)
