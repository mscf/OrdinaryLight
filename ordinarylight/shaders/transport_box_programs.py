"""Typed transport helpers. Generated artifacts must be rebuilt from this source."""
import ordinaryshade as osh
from .transport_programs import OrdinaryLightCustomHit

@osh.function
def OL_BOX_ENTRY(origin: osh.vec3, direction: osh.vec3, t_min: osh.f32, t_max: osh.f32, parameters: osh.vec4, tolerance: osh.f32, max_steps: osh.u32, hit: osh.inout(OrdinaryLightCustomHit)) -> osh.u32:
    found = False
    closest = t_max
    if ((((osh.any_value(osh.is_nan(parameters.xy)) or osh.any_value(osh.is_inf(parameters.xy))) or parameters.x < 0.0) or parameters.y < 1.0) or osh.any_value(osh.floor(parameters.xy) != parameters.xy)) or osh.any_value(parameters.xy > osh.vec2(16777216.0)):
        return osh.u32(2)
    first = osh.u32(parameters.x)
    count = osh.u32(parameters.y)
    index_count = osh.array_length(OL_BOX_RESOURCE) / osh.u32(3)
    if osh.specialization("OL_BOX_PARTITIONED"):
        index_count = osh.array_length(OL_BOX_INDICES)
    if count > max_steps or first + count > index_count:
        return osh.u32(2)
    slot = first
    while slot < first + count:
        i = slot
        if osh.specialization("OL_BOX_PARTITIONED"):
            i = OL_BOX_INDICES[slot]
        if i >= osh.u32(osh.array_length(OL_BOX_RESOURCE)) / osh.u32(3):
            return osh.u32(2)
        lower = OL_BOX_RESOURCE[i * osh.u32(3)]
        upper = OL_BOX_RESOURCE[i * osh.u32(3) + osh.u32(1)]
        if lower.w == 0.0:
            slot = slot + 1
            continue
        if ((((lower.w != 1.0 or osh.any_value(osh.is_nan(lower.xyz))) or osh.any_value(osh.is_inf(lower.xyz))) or osh.any_value(osh.is_nan(upper.xyz))) or osh.any_value(osh.is_inf(upper.xyz))) or osh.any_value(lower.xyz >= upper.xyz):
            return osh.u32(2)
        near_t = -1e+30
        far_t = 1e+30
        near_n = osh.vec3(0)
        far_n = osh.vec3(0)
        missed = False
        axis = 0
        while axis < 3:
            if osh.absolute(direction[axis]) < 1e-20:
                if origin[axis] < lower[axis] or origin[axis] > upper[axis]:
                    missed = True
            else:
                a = (lower[axis] - origin[axis]) / direction[axis]
                b = (upper[axis] - origin[axis]) / direction[axis]
                n = osh.vec3(0)
                n[axis] = -1.0 if direction[axis] > 0.0 else 1.0
                if osh.minimum(a, b) > near_t:
                    near_t = osh.minimum(a, b)
                    near_n = n
                if osh.maximum(a, b) < far_t:
                    far_t = osh.maximum(a, b)
                    far_n = -n
            axis = axis + 1
        if missed or near_t > far_t:
            slot = slot + 1
            continue
        entry = near_t >= t_min - tolerance
        distance = osh.maximum(near_t, t_min) if entry else far_t
        if distance < t_min or distance > closest:
            slot = slot + 1
            continue
        metadata = osh.float_bits_to_uint(OL_BOX_RESOURCE[i * osh.u32(3) + osh.u32(2)])
        hit.distance = distance
        hit.geometric_normal = near_n if entry else far_n
        hit.material = metadata.x
        hit.boundary = metadata.y
        hit.identity = metadata.z
        hit.flags = OL_HIT_MATERIAL | OL_HIT_BOUNDARY | OL_HIT_IDENTITY
        closest = distance
        found = True
        slot = slot + 1
    return osh.u32(1) if found else osh.u32(0)
