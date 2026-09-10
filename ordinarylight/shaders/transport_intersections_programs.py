"""Typed transport helpers. Generated artifacts must be rebuilt from this source."""
import ordinaryshade as osh
from .transport_programs import CustomRecord, TransportMaterialRecord, OrdinaryLightCustomHit

@osh.structure
class OrdinaryLightHit:
    position_distance: osh.vec4
    geometric_normal: osh.vec4
    shading_normal: osh.vec4
    identity: osh.uvec4
    boundary: osh.uvec4

@osh.function
def ordinarylightBounds(origin: osh.vec3, direction: osh.vec3, lower: osh.vec3, upper: osh.vec3, near_t: osh.inout(osh.f32), far_t: osh.inout(osh.f32)) -> osh.boolean:
    axis = 0
    while axis < 3:
        if osh.absolute(direction[axis]) < 1e-20:
            if origin[axis] < lower[axis] or origin[axis] > upper[axis]:
                return False
        else:
            a = (lower[axis] - origin[axis]) / direction[axis]
            b = (upper[axis] - origin[axis]) / direction[axis]
            near_t = osh.maximum(near_t, osh.minimum(a, b))
            far_t = osh.minimum(far_t, osh.maximum(a, b))
        axis = axis + 1
    return near_t <= far_t

@osh.function
def ordinarylightIntersect(origin: osh.vec3, direction: osh.vec3, t_min: osh.f32, t_max: osh.f32, tolerance: osh.f32, max_steps: osh.u32) -> OrdinaryLightHit:
    hit = OrdinaryLightHit(osh.vec4(0), osh.vec4(0), osh.vec4(0), osh.uvec4(0), osh.uvec4(0))
    hit.position_distance = osh.vec4(0, 0, 0, t_max)
    hit.geometric_normal = osh.vec4(0)
    hit.shading_normal = osh.vec4(0)
    hit.identity = osh.uvec4(0)
    hit.boundary = osh.uvec4(osh.u32(4294967295), 0, 0, 0)
    query = osh.ray_query()
    query.initialize(transport_tlas, gl_RayFlagsOpaqueEXT, 255, origin, t_min, direction, t_max)
    generated_hit = hit
    while query.proceed():
        if query.intersection_type(False) != gl_RayQueryCandidateIntersectionAABBEXT:
            continue
        index = query.primitive_index(False)
        geometry = custom_geometry[index]
        if geometry.metadata.x == osh.u32(4294967295):
            continue
        near_t = t_min
        far_t = t_max
        if query.intersection_type(True) != gl_RayQueryCommittedIntersectionNoneEXT:
            far_t = osh.minimum(far_t, query.intersection_t(True))
        if not ordinarylightBounds(origin, direction, geometry.lower.xyz, geometry.upper.xyz, near_t, far_t):
            continue
        result = OrdinaryLightCustomHit(osh.f32(0), osh.vec3(0), osh.u32(0), osh.u32(0), osh.u32(0), osh.u32(0), osh.vec2(0), osh.vec3(0))
        result.distance = 0
        result.geometric_normal = osh.vec3(0)
        result.flags = osh.u32(0)
        result.material = osh.u32(0)
        result.boundary = OL_NO_BOUNDARY
        result.identity = osh.u32(0)
        result.uv = osh.vec2(0)
        result.shading_normal = osh.vec3(0)
        status = ordinarylightCustomIntersect(geometry.metadata.x, origin, direction, near_t, far_t, geometry.parameters, tolerance, max_steps, result)
        distance = result.distance
        normal = result.geometric_normal
        if status > osh.u32(1) or (status == osh.u32(1) and ((((((osh.is_nan(distance) or osh.is_inf(distance)) or osh.any_value(osh.is_nan(normal))) or osh.any_value(osh.is_inf(normal))) or osh.absolute(osh.dot(normal, normal) - 1.0) > 0.001) or distance < near_t - tolerance) or distance > far_t + tolerance)):
            hit.boundary.w = osh.u32(1)
            query.terminate()
            return hit
        if (status == osh.u32(1) and distance <= far_t) and distance >= t_min:
            material = geometry.metadata.y
            boundary = geometry.metadata.z
            identity = geometry.metadata.w
            if result.flags & ~osh.u32(31) != osh.u32(0):
                hit.boundary.w = osh.u32(32)
                query.terminate()
                return hit
            if result.flags & OL_HIT_MATERIAL != osh.u32(0):
                if result.material >= OL_MATERIAL_COUNT - OL_CUSTOM_MATERIAL_OFFSET:
                    hit.boundary.w = osh.u32(32)
                    query.terminate()
                    return hit
                material = OL_CUSTOM_MATERIAL_OFFSET + result.material
            if result.flags & OL_HIT_BOUNDARY != osh.u32(0):
                boundary = OL_NO_BOUNDARY
                if result.boundary != OL_NO_BOUNDARY:
                    i = osh.u32(0)
                    while i < OL_BOUNDARY_COUNT:
                        if medium_boundaries[i].z == result.boundary:
                            boundary = i
                            break
                        i = i + 1
                    if boundary == OL_NO_BOUNDARY:
                        hit.boundary.w = osh.u32(32)
                        query.terminate()
                        return hit
            if material >= OL_MATERIAL_COUNT or (boundary != OL_NO_BOUNDARY and boundary >= OL_BOUNDARY_COUNT):
                hit.boundary.w = osh.u32(32)
                query.terminate()
                return hit
            if (transport_materials[material].albedo_kind.w == 1.0) != (boundary != OL_NO_BOUNDARY):
                hit.boundary.w = osh.u32(32)
                query.terminate()
                return hit
            if result.flags & OL_HIT_IDENTITY != osh.u32(0):
                identity = result.identity
            uv = result.uv if result.flags & OL_HIT_UV != osh.u32(0) else osh.vec2(0)
            shading = result.shading_normal if result.flags & OL_HIT_SHADING_NORMAL != osh.u32(0) else normal
            if ((((osh.any_value(osh.is_nan(uv)) or osh.any_value(osh.is_inf(uv))) or osh.any_value(osh.is_nan(shading))) or osh.any_value(osh.is_inf(shading))) or osh.absolute(osh.dot(shading, shading) - 1.0) > 0.001) or osh.dot(shading, normal) <= 0.0:
                hit.boundary.w = osh.u32(1)
                query.terminate()
                return hit
            query.generate_intersection(distance)
            if (query.intersection_type(True) == gl_RayQueryCommittedIntersectionGeneratedEXT and query.primitive_index(True) == index) and query.intersection_t(True) == distance:
                generated_hit.identity = osh.uvec4(2, index, identity, material)
                generated_hit.boundary.x = boundary
                generated_hit.geometric_normal = osh.vec4(osh.normalize(normal), uv.x)
                generated_hit.shading_normal = osh.vec4(osh.normalize(shading), uv.y)
    kind = query.intersection_type(True)
    if kind == gl_RayQueryCommittedIntersectionNoneEXT:
        return hit
    distance = query.intersection_t(True)
    index = query.primitive_index(True)
    hit.position_distance = osh.vec4(origin + distance * direction, distance)
    if kind == gl_RayQueryCommittedIntersectionTriangleEXT:
        index = index + query.instance_custom_index(True)
        a = transport_vertices[index * osh.u32(3)].xyz
        b = transport_vertices[index * osh.u32(3) + osh.u32(1)].xyz
        c = transport_vertices[index * osh.u32(3) + osh.u32(2)].xyz
        geometric = osh.normalize(osh.cross(b - a, c - a))
        uv = query.barycentrics(True)
        shading = transport_attributes[index * osh.u32(9)].xyz * (1.0 - uv.x - uv.y) + transport_attributes[index * osh.u32(9) + osh.u32(3)].xyz * uv.x + transport_attributes[index * osh.u32(9) + osh.u32(6)].xyz * uv.y
        if osh.dot(shading, shading) < 1e-12:
            shading = geometric
        shading = osh.normalize(shading)
        if osh.dot(shading, geometric) < 0:
            shading = -shading
        metadata = triangle_records[index]
        hit.identity = osh.uvec4(1, index, metadata.z, metadata.x)
        hit.boundary.x = metadata.y
        texcoord = transport_attributes[index * osh.u32(9) + osh.u32(1)].xy * (1.0 - uv.x - uv.y) + transport_attributes[index * osh.u32(9) + osh.u32(4)].xy * uv.x + transport_attributes[index * osh.u32(9) + osh.u32(7)].xy * uv.y
        hit.geometric_normal = osh.vec4(geometric, texcoord.x)
        hit.shading_normal = osh.vec4(shading, texcoord.y)
    else:
        hit.identity = generated_hit.identity
        hit.boundary.x = generated_hit.boundary.x
        hit.geometric_normal = generated_hit.geometric_normal
        hit.shading_normal = generated_hit.shading_normal
    if hit.boundary.x != osh.u32(4294967295):
        hit.boundary.yz = medium_boundaries[hit.boundary.x].xy
    return hit
