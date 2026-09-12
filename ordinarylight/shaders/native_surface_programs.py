"""Typed application material contract for native procedural intersections.

The returned parameters use the same BSDF implementation as native triangles.
Intersection identity and texture coordinates belong to the application; they
must not be interpreted as addresses into native triangle attribute buffers.
"""
import ordinaryshade as osh

from .native_intersection_programs import NativeIntersection
from .transport_programs import MaterialData


@osh.external
def nativeEvaluateMaterial(hit: NativeIntersection, cone_width: osh.f32) -> MaterialData:
    """Reserved signature for an application-owned OrdinaryShade evaluator.

    ``cone_width`` is the world-space ray footprint at this hit. The callback
    samples its own resident material/texture data and returns evaluated native
    parameters. Normals and application identity come from the intersection.
    """
    pass
