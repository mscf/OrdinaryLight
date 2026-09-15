"""Typed experimental visibility layouts: scalar distance records and vector planes."""
import ordinaryshade as osh
from .native_intersection_programs import NativeIntersection


@osh.structure
class DistanceVisibility:
    distance: osh.u32
    geometric_normal_x: osh.u32
    geometric_normal_y: osh.u32
    geometric_normal_z: osh.u32
    geometric_normal_w: osh.u32
    shading_normal_x: osh.u32
    shading_normal_y: osh.u32
    shading_normal_z: osh.u32
    shading_normal_w: osh.u32
    identity_x: osh.u32
    identity_y: osh.u32
    identity_z: osh.u32
    identity_w: osh.u32
    address_x: osh.u32
    address_y: osh.u32
    address_z: osh.u32
    address_w: osh.u32
    texcoord_x: osh.u32
    texcoord_y: osh.u32
    texcoord_z: osh.u32
    texcoord_w: osh.u32
    previous_position_x: osh.u32
    previous_position_y: osh.u32
    previous_position_z: osh.u32
    previous_position_w: osh.u32


@osh.function
def packDistanceVisibility(hit: NativeIntersection) -> DistanceVisibility:
    return DistanceVisibility(
        osh.float_bits_to_uint(hit.position_distance.w),
        osh.float_bits_to_uint(hit.geometric_normal.x), osh.float_bits_to_uint(hit.geometric_normal.y), osh.float_bits_to_uint(hit.geometric_normal.z), osh.float_bits_to_uint(hit.geometric_normal.w),
        osh.float_bits_to_uint(hit.shading_normal.x), osh.float_bits_to_uint(hit.shading_normal.y), osh.float_bits_to_uint(hit.shading_normal.z), osh.float_bits_to_uint(hit.shading_normal.w),
        hit.identity.x, hit.identity.y, hit.identity.z, hit.identity.w,
        hit.address.x, hit.address.y, hit.address.z, hit.address.w,
        osh.float_bits_to_uint(hit.texcoord.x), osh.float_bits_to_uint(hit.texcoord.y), osh.float_bits_to_uint(hit.texcoord.z), osh.float_bits_to_uint(hit.texcoord.w),
        osh.float_bits_to_uint(hit.previous_position.x), osh.float_bits_to_uint(hit.previous_position.y), osh.float_bits_to_uint(hit.previous_position.z), osh.float_bits_to_uint(hit.previous_position.w),
    )


@osh.function
def unpackDistanceVisibility(value: DistanceVisibility, origin: osh.vec3, direction: osh.vec3) -> NativeIntersection:
    distance=osh.uint_bits_to_float(value.distance)
    position=osh.vec3(0.0)
    if value.address_w != osh.u32(0):
        position=origin+distance*direction
    return NativeIntersection(
        osh.vec4(position,distance),
        osh.uint_bits_to_float(osh.uvec4(value.geometric_normal_x, value.geometric_normal_y, value.geometric_normal_z, value.geometric_normal_w)),
        osh.uint_bits_to_float(osh.uvec4(value.shading_normal_x, value.shading_normal_y, value.shading_normal_z, value.shading_normal_w)),
        osh.uvec4(value.identity_x, value.identity_y, value.identity_z, value.identity_w),
        osh.uvec4(value.address_x, value.address_y, value.address_z, value.address_w),
        osh.uint_bits_to_float(osh.uvec4(value.texcoord_x, value.texcoord_y, value.texcoord_z, value.texcoord_w)),
        osh.uint_bits_to_float(osh.uvec4(value.previous_position_x, value.previous_position_y, value.previous_position_z, value.previous_position_w)),
    )


@osh.function
def storeVisibilityPlanes(hit: NativeIntersection, pixel: osh.u32, pixels: osh.u32, sample: osh.u32) -> osh.void:
    base=sample*pixels*osh.u32(7)+pixel
    visibility_planes[base]=osh.float_bits_to_uint(hit.position_distance)
    visibility_planes[base+pixels]=osh.float_bits_to_uint(hit.geometric_normal)
    visibility_planes[base+pixels*osh.u32(2)]=osh.float_bits_to_uint(hit.shading_normal)
    visibility_planes[base+pixels*osh.u32(3)]=hit.identity
    visibility_planes[base+pixels*osh.u32(4)]=hit.address
    visibility_planes[base+pixels*osh.u32(5)]=osh.float_bits_to_uint(hit.texcoord)
    visibility_planes[base+pixels*osh.u32(6)]=osh.float_bits_to_uint(hit.previous_position)


@osh.function
def loadVisibilityPlanes(pixel: osh.u32, pixels: osh.u32, sample: osh.u32) -> NativeIntersection:
    base=sample*pixels*osh.u32(7)+pixel
    return NativeIntersection(
        osh.uint_bits_to_float(visibility_planes[base]),
        osh.uint_bits_to_float(visibility_planes[base+pixels]),
        osh.uint_bits_to_float(visibility_planes[base+pixels*osh.u32(2)]),
        visibility_planes[base+pixels*osh.u32(3)],
        visibility_planes[base+pixels*osh.u32(4)],
        osh.uint_bits_to_float(visibility_planes[base+pixels*osh.u32(5)]),
        osh.uint_bits_to_float(visibility_planes[base+pixels*osh.u32(6)]))


@osh.function
def storeDistancePlanes(hit: NativeIntersection, pixel: osh.u32, pixels: osh.u32, sample: osh.u32) -> osh.void:
    plane_start=sample*(pixels*osh.u32(6)+(pixels+osh.u32(3))/osh.u32(4))
    base=plane_start+pixel
    visibility_planes[base]=osh.float_bits_to_uint(hit.geometric_normal)
    visibility_planes[base+pixels]=osh.float_bits_to_uint(hit.shading_normal)
    visibility_planes[base+pixels*osh.u32(2)]=hit.identity
    visibility_planes[base+pixels*osh.u32(3)]=hit.address
    visibility_planes[base+pixels*osh.u32(4)]=osh.float_bits_to_uint(hit.texcoord)
    visibility_planes[base+pixels*osh.u32(5)]=osh.float_bits_to_uint(hit.previous_position)
    visibility_planes[plane_start+pixels*osh.u32(6)+pixel/osh.u32(4)][pixel%osh.u32(4)]=osh.float_bits_to_uint(hit.position_distance.w)


@osh.function
def loadDistancePlanes(pixel: osh.u32, pixels: osh.u32, sample: osh.u32, origin: osh.vec3, direction: osh.vec3) -> NativeIntersection:
    plane_start=sample*(pixels*osh.u32(6)+(pixels+osh.u32(3))/osh.u32(4))
    base=plane_start+pixel
    address=visibility_planes[base+pixels*osh.u32(3)]
    distance=osh.uint_bits_to_float(visibility_planes[plane_start+pixels*osh.u32(6)+pixel/osh.u32(4)][pixel%osh.u32(4)])
    position=osh.vec3(0.0)
    if address.w!=osh.u32(0):
        position=origin+distance*direction
    return NativeIntersection(
        osh.vec4(position,distance),
        osh.uint_bits_to_float(visibility_planes[base]),
        osh.uint_bits_to_float(visibility_planes[base+pixels]),
        visibility_planes[base+pixels*osh.u32(2)],
        address,
        osh.uint_bits_to_float(visibility_planes[base+pixels*osh.u32(4)]),
        osh.uint_bits_to_float(visibility_planes[base+pixels*osh.u32(5)]))
