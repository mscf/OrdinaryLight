"""Optional Ordinary Shade material hooks used by raster showcases."""

import ordinaryshade as osh

from ..raster import raster_material_hook
from ..shaders.raster_programs import (
    RasterMaterialContext,
    RasterSurface,
    blend_raster_surfaces,
)


@raster_material_hook
def layered_raster_showcase_hook(
    surface: RasterSurface, context: RasterMaterialContext,
) -> RasterSurface:
    """Layer a procedural blue coating over the standard raster surface."""
    layer_weight = osh.maximum(
        0.0, osh.minimum(1.0, context.uv.x * context.uv.x),
    )
    coating = RasterSurface(
        osh.vec3(0.15, 0.75, 1.0),
        surface.emission,
        surface.normal,
        0.25,
        0.12,
        surface.transmission,
        surface.occlusion,
    )
    return blend_raster_surfaces(surface, coating, layer_weight)
