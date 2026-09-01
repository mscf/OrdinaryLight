"""Raster programs, resources, and portable processing."""
from . import _core
globals().update({name: value for name, value in vars(_core).items() if not name.startswith("_")})
from .lighting import evaluate_vertex_lighting, material_channels
from .resources import (
    CAMERA_DTYPE, GEOMETRY_PRODUCT_CAMERA_DTYPE, DRAW_DTYPE, LIGHT_DTYPE, MATERIAL_DTYPE, SHADOW_DTYPE,
    RasterGpuScene, pack_raster_gpu_scene,
)
from .shadows import ShadowMapRequest, plan_shadow_maps
