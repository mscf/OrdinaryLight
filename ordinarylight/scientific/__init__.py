"""Scientific data models and renderer adapters."""

from .clipping import ClipPlane, ClipRegion, RegionOfInterest
from .colormaps import available_colormaps, colormap, opacity_curve
from .export import (
    SCIENTIFIC_EXPORT_SCHEMA, export_scientific_image, scalar_field_sha256,
    verify_scientific_export,
)
from .scalar_field import ProbeResult, RayProbeResult, SampleResult, ScalarField3D
from .series import ScalarFieldSeries
from .isosurface import ScalarIsosurface
from .interaction import ScientificInspector
from .slice import ScalarSlice, scientific_slice_material
from .transfer import ScalarMapping, TransferFunction

__all__ = [
    "ClipPlane", "ClipRegion", "ProbeResult", "RayProbeResult",
    "RegionOfInterest", "SampleResult", "ScalarField3D",
    "ScalarFieldSeries", "ScalarIsosurface", "ScalarMapping", "ScalarSlice",
    "ScientificInspector",
    "SCIENTIFIC_EXPORT_SCHEMA", "TransferFunction", "export_scientific_image",
    "available_colormaps", "colormap", "opacity_curve", "scalar_field_sha256",
    "scientific_slice_material", "verify_scientific_export",
]
