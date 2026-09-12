"""Shared native ray-query contract, authored in OrdinaryShade.

The native triangle address and application identity are separate. Generated
intersection payloads must remain associated with the committed candidate;
traversal order is never an identity or nearest-hit guarantee.
"""

import ordinaryshade as osh


@osh.structure
class NativeIntersection:
    position_distance: osh.vec4
    geometric_normal: osh.vec4
    shading_normal: osh.vec4
    identity: osh.uvec4
    address: osh.uvec4
    texcoord: osh.vec4
    previous_position: osh.vec4


@osh.structure
class NativeOpticalBoundary:
    """Lossless smooth boundary; ior is (outside, inside), normal points outside."""
    ior: osh.vec2
    enabled: osh.boolean


@osh.external
def nativeEvaluateBoundary(hit: NativeIntersection) -> NativeOpticalBoundary:
    """Typed application optical boundary callback, independent of material IDs."""
    pass


@osh.function
def nativeBoundaryEnabled(boundary: NativeOpticalBoundary) -> osh.boolean:
    return (boundary.enabled and osh.all_value(boundary.ior > osh.vec2(0.0))
            and not osh.any_value(osh.is_nan(boundary.ior))
            and not osh.any_value(osh.is_inf(boundary.ior)))


@osh.function
def nativeIntersectionMiss() -> NativeIntersection:
    return NativeIntersection(
        osh.vec4(0.0, 0.0, 0.0, -1.0), osh.vec4(0.0), osh.vec4(0.0),
        osh.uvec4(4294967295), osh.uvec4(4294967295, 4294967295, 0, 0),
        osh.vec4(0.0), osh.vec4(0.0),
    )


@osh.external
def nativeIntersectCandidate(
    origin: osh.vec3, direction: osh.vec3, t_min: osh.f32, t_max: osh.f32,
    primitive: osh.u32, instance: osh.u32, instance_offset: osh.u32,
) -> NativeIntersection:
    """Reserved ABI for a typed application candidate implementation."""
    pass


@osh.function
def nativeTraceSurface(
    origin: osh.vec3, t_min: osh.f32,
    direction: osh.vec3, t_max: osh.f32, visibility: osh.boolean, mask: osh.u32,
) -> NativeIntersection:
    hit = nativeIntersectionMiss()
    selected = hit
    selected_primitive = osh.u32(4294967295)
    selected_instance = osh.u32(4294967295)
    flags = osh.u32(1)
    if visibility:
        flags = flags | osh.u32(4)
    query = osh.ray_query()
    query.initialize(scene_tlas, flags, mask, origin, t_min, direction, t_max)
    while query.proceed():
        if osh.specialization('WAVE_CUSTOM_GEOMETRY'):
            if query.intersection_type(False) != osh.u32(1):
                continue
            primitive = query.primitive_index(False)
            instance = query.instance_id(False)
            limit = t_max
            if query.intersection_type(True) != osh.u32(0):
                limit = osh.minimum(limit, query.intersection_t(True))
            candidate = nativeIntersectCandidate(
                origin, direction, t_min, limit, primitive, instance,
                query.instance_custom_index(False),
            )
            distance = candidate.position_distance.w
            if distance < t_min or distance > limit:
                continue
            if osh.is_nan(distance) or osh.is_inf(distance):
                continue
            if (osh.any_value(osh.is_nan(candidate.geometric_normal.xyz))
                    or osh.any_value(osh.is_inf(candidate.geometric_normal.xyz))
                    or osh.absolute(osh.dot(candidate.geometric_normal.xyz,
                                            candidate.geometric_normal.xyz) - 1.0) > 0.001):
                continue
            if (osh.any_value(osh.is_nan(candidate.shading_normal.xyz))
                    or osh.any_value(osh.is_inf(candidate.shading_normal.xyz))
                    or osh.absolute(osh.dot(candidate.shading_normal.xyz,
                                            candidate.shading_normal.xyz) - 1.0) > 0.001
                    or osh.dot(candidate.shading_normal.xyz, candidate.geometric_normal.xyz) <= 0.0):
                continue
            candidate.position_distance = osh.vec4(origin + distance * direction, distance)
            candidate.address = osh.uvec4(primitive + query.instance_custom_index(False),
                                         primitive, query.instance_custom_index(False), 2)
            if osh.specialization('WAVE_NATIVE_OPTICAL_BOUNDARIES'):
                if visibility:
                    boundary = nativeEvaluateBoundary(candidate)
                    if nativeBoundaryEnabled(boundary) and boundary.ior.x == boundary.ior.y:
                        continue
            query.generate_intersection(distance)
            if (query.intersection_type(True) == osh.u32(2)
                    and query.primitive_index(True) == primitive
                    and query.instance_id(True) == instance
                    and query.intersection_t(True) == distance):
                selected = candidate
                selected_primitive = primitive
                selected_instance = instance
    kind = query.intersection_type(True)
    if kind == osh.u32(0):
        return hit
    distance = query.intersection_t(True)
    primitive = query.primitive_index(True)
    instance = query.instance_id(True)
    instance_offset = query.instance_custom_index(True)
    if kind == osh.u32(2):
        if primitive != selected_primitive or instance != selected_instance:
            return hit
        hit = selected
    else:
        hit.identity = osh.uvec4(instance, primitive, primitive + instance_offset, 0)
        hit.texcoord.xy = query.barycentrics(True)
    hit.position_distance = osh.vec4(origin + distance * direction, distance)
    hit.address = osh.uvec4(primitive + instance_offset, primitive, instance_offset, kind)
    return hit


@osh.function
def nativeSurfaceMask() -> osh.u32:
    if osh.specialization('WAVE_CUSTOM_GEOMETRY'):
        return osh.u32(3)
    return osh.u32(1)
