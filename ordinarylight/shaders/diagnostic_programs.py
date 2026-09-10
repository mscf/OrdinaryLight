"""Nearest-hit diagnostic kernels sharing the transport intersection ABI."""
import ordinaryshade as osh
from .transport_intersections_programs import ordinarylightIntersect, OrdinaryLightHit

@osh.structure
class RayInput:
    origin: osh.vec4
    direction: osh.vec4

@osh.structure
class OrdinaryLightColorHit:
    color: osh.vec4
    identity: osh.uvec4

@osh.structure
class QueryConstants:
    count: osh.u32
    t_min: osh.f32
    t_max: osh.f32
    tolerance: osh.f32
    max_steps: osh.u32

@osh.function
def main() -> osh.void:
    i = gl_GlobalInvocationID.x
    if i >= pc.count:
        return
    hit = ordinarylightIntersect(rays[i].origin.xyz, rays[i].direction.xyz, pc.t_min, pc.t_max, pc.tolerance, pc.max_steps)
    if osh.specialization('OL_QUERY_COLORS'):
        status = hit.boundary.w
        color = osh.vec4(0, 0, 0, 1)
        if status == osh.u32(0) and hit.identity.x != osh.u32(0):
            if hit.identity.z >= osh.array_length(ordinarylightQueryColors):
                status = osh.u32(32)
            else:
                color = ordinarylightQueryColors[hit.identity.z]
                if osh.any_value(osh.is_nan(color)) or osh.any_value(osh.is_inf(color)):
                    status = osh.u32(16)
                    color = osh.vec4(0, 0, 0, 1)
        color_hits[i].color = color
        color_hits[i].identity = osh.uvec4(hit.identity.x, hit.identity.z, hit.identity.w, status)
        if osh.specialization('OL_QUERY_HDR'):
            width = osh.u32(ordinarylightQueryHdr.size().x)
            ordinarylightQueryHdr.store(osh.ivec2(i % width, i / width), color)
    else:
        hits[i] = hit
