"""Optional Ordinary Shade material hooks used by raster showcases."""

import ordinaryshade as osh

from ..materials import material_modifier
from ..materials.gpu import (
    SurfaceContext,
    SurfaceParameters,
    blend_surface_parameters,
)


@material_modifier
def advanced_surface_showcase_modifier(
    surface: SurfaceParameters, context: SurfaceContext,
) -> SurfaceParameters:
    """Layer a procedural blue coating over the standard raster surface."""
    layer_weight = osh.maximum(
        0.0, osh.minimum(1.0, context.uv.x * context.uv.x),
    )
    coating = SurfaceParameters(
        osh.vec3(0.15, 0.75, 1.0),
        surface.emission,
        surface.normal,
        0.25,
        0.12,
        surface.transmission,
        surface.occlusion,
        0.85,
        0.08,
        osh.vec3(0.03, 0.12, 0.2),
        0.25,
        0.35,
        1.0,
        0.35,
        osh.vec3(1.0, 0.22, 0.12),
        0.45,
    )
    return blend_surface_parameters(surface, coating, layer_weight)


# Compatibility for the first raster-only showcase name.
layered_raster_showcase_hook = advanced_surface_showcase_modifier
