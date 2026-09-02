"""Generate shared fused-primary helpers with Ordinary Shade."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ORDINARYSHADE = ROOT.parent / "ordinaryshade"
if DEFAULT_ORDINARYSHADE.is_dir():
    sys.path.insert(0, str(DEFAULT_ORDINARYSHADE))

import ordinaryshade as osh


class OrdinarylightWaveRayABI:
    __annotations__ = {
        "origin_tmin": osh.vec4,
        "direction_tmax": osh.vec4,
        "path_index": osh.u32,
        "padding_a": osh.u32,
        "padding_b": osh.u32,
        "padding_c": osh.u32,
    }


OrdinarylightWaveRayABI = osh.structure(OrdinarylightWaveRayABI)


class OrdinarylightOutputQueueABI:
    __annotations__ = {
        "count": osh.u32,
        "capacity": osh.u32,
        "overflow": osh.u32,
        "queue_padding": osh.u32,
        "rays": osh.runtime_array(OrdinarylightWaveRayABI),
    }


OrdinarylightOutputQueueABI = osh.structure(OrdinarylightOutputQueueABI)


class MaterialData:
    __annotations__ = {
        "base_roughness": osh.vec4,
        "emission_metallic": osh.vec4,
        "attenuation_transmission": osh.vec4,
        "ior_distance": osh.vec4,
        "texture_indices": osh.vec4,
        "texture_parameters": osh.vec4,
        "advanced0": osh.vec4,
        "advanced1": osh.vec4,
        "sheen_color": osh.vec4,
        "subsurface_color": osh.vec4,
        "advanced_texture_indices": osh.vec4,
        "optical": osh.vec4,
    }


MaterialData = osh.structure(MaterialData)


class VertexAttributeData:
    __annotations__ = {
        "normal": osh.vec4,
        "texcoord": osh.vec4,
        "tangent": osh.vec4,
    }


VertexAttributeData = osh.structure(VertexAttributeData)


class WaveMediumStack:
    __annotations__ = {"ior": osh.array(osh.f32, 16)}


WaveMediumStack = osh.structure(WaveMediumStack)


class WavePathState:
    __annotations__ = {
        "throughput": osh.vec4,
        "radiance": osh.vec4,
        "metadata": osh.uvec4,
    }


WavePathState = osh.structure(WavePathState)


class SecondaryPathState:
    __annotations__ = {
        "position_valid": osh.vec4,
        "normal_pdf": osh.vec4,
        "primary_throughput": osh.vec4,
        "primary_radiance": osh.vec4,
        "diffuse_radiance_hit_distance": osh.vec4,
        "specular_radiance_hit_distance": osh.vec4,
        "primary_position": osh.vec4,
        "primary_geometry": osh.vec4,
    }


SecondaryPathState = osh.structure(SecondaryPathState)


@osh.external
def integrateVolumesBeforeSurface(
    origin: osh.vec3,
    direction: osh.vec3,
    distance: osh.f32,
    radiance: osh.inout(osh.vec3),
    throughput: osh.inout(osh.vec3),
) -> osh.void:
    pass


@osh.external
def profileWork(counter: osh.u32, amount: osh.u32) -> osh.void:
    pass


MaterialEvaluationABI = osh.opaque_type("MaterialEvaluation")


@osh.external
def waveApplyMaterialProgram(
    material: osh.inout(MaterialData),
    normal: osh.vec3,
    uv: osh.vec2,
    direction: osh.vec3,
    entering: osh.boolean,
    primitive: osh.u32,
    weights: osh.vec3,
    bounce_index: osh.f32,
) -> MaterialEvaluationABI:
    pass


@osh.external
def processPrimaryPixel(pixel: osh.uvec2) -> osh.void:
    pass


@osh.external
def ordinarylightSecondaryBounce(
    path: osh.inout(WavePathState),
    origin: osh.inout(osh.vec3),
    direction: osh.inout(osh.vec3),
    path_index: osh.u32,
    medium_depth: osh.inout(osh.u32),
    rng: osh.inout(osh.u32),
    cone_width: osh.inout(osh.f32),
    cone_spread: osh.inout(osh.f32),
) -> osh.boolean:
    pass


@osh.function
def ordinarylight_secondary_trace_query(
    origin: osh.vec3,
    direction: osh.vec3,
    surface_hit: osh.inout(osh.boolean),
    distance: osh.inout(osh.f32),
    primitive: osh.inout(osh.u32),
    barycentrics: osh.inout(osh.vec2),
) -> osh.void:
    query = osh.ray_query()
    query.initialize(
        scene_tlas, osh.u32(1), osh.u32(1), origin, 0.001,
        direction, 1.0e30,
    )
    while query.proceed():
        pass
    surface_hit = query.intersection_type(True) == osh.u32(1)
    if surface_hit:
        distance = query.intersection_t(True)
        primitive = query.primitive_index(True) + query.instance_custom_index(True)
        barycentrics = query.barycentrics(True)


@osh.function
def ordinarylight_secondary_reorder(hint: osh.u32) -> osh.void:
    osh.reorder_thread(hint, osh.u32(7))


@osh.function
def ordinarylight_primary_scheduled_group(
    group_id: osh.uvec2,
    group_count: osh.uvec2,
    swizzle_width: osh.u32,
) -> osh.uvec2:
    scheduled = group_id
    if swizzle_width > osh.u32(1):
        full_tiles = group_count.x / swizzle_width
        full_tile_groups = full_tiles * swizzle_width * group_count.y
        linear_group = group_id.y * group_count.x + group_id.x
        if linear_group < full_tile_groups:
            groups_per_tile = swizzle_width * group_count.y
            tile = linear_group / groups_per_tile
            local = linear_group - tile * groups_per_tile
            scheduled = osh.uvec2(
                tile * swizzle_width + local % swizzle_width,
                local / swizzle_width,
            )
        else:
            tail_width = group_count.x - full_tiles * swizzle_width
            local = linear_group - full_tile_groups
            scheduled = osh.uvec2(
                full_tiles * swizzle_width + local % tail_width,
                local / tail_width,
            )
    return scheduled


@osh.function
def ordinarylight_persistent_coarse_schedule(
    tile_extent: osh.uvec2,
) -> osh.void:
    persistent_tile_index = osh.shared(osh.u32)
    tile_count = (tile_extent + osh.uvec2(7)) / osh.uvec2(8)
    total_tiles = tile_count.x * tile_count.y
    while True:
        if osh.local_invocation_index == osh.u32(0):
            persistent_tile_index = osh.atomic_add(
                ordinarylight_output_queue.count, osh.u32(1)
            )
        osh.workgroup_barrier()
        if persistent_tile_index >= total_tiles:
            return
        tile = osh.uvec2(
            persistent_tile_index % tile_count.x,
            persistent_tile_index / tile_count.x,
        )
        processPrimaryPixel(
            tile * osh.uvec2(8) + osh.local_invocation_id.xy
        )
        osh.workgroup_barrier()


@osh.function
def ordinarylight_trace_remaining(
    path: osh.inout(WavePathState),
    origin: osh.inout(osh.vec3),
    direction: osh.inout(osh.vec3),
    path_index: osh.u32,
    medium_depth: osh.inout(osh.u32),
    rng: osh.inout(osh.u32),
    cone_width: osh.inout(osh.f32),
    cone_spread: osh.inout(osh.f32),
    stop_bounce: osh.u32,
) -> osh.void:
    while osh.u32(path.throughput.w) < stop_bounce:
        if not ordinarylightSecondaryBounce(
            path,
            origin,
            direction,
            path_index,
            medium_depth,
            rng,
            cone_width,
            cone_spread,
        ):
            break
    path.metadata.z = rng


@osh.function
def ordinarylight_reserve_output_index(
    subgroup_enqueue: osh.u32,
) -> osh.u32:
    if subgroup_enqueue == osh.u32(0):
        return osh.atomic_add(ordinarylight_output_queue_count, osh.u32(1))
    active_lanes = osh.subgroup_ballot(True)
    base = osh.u32(0)
    if osh.subgroup_elect():
        base = osh.atomic_add(
            ordinarylight_output_queue_count,
            osh.subgroup_ballot_bit_count(active_lanes),
        )
    base = osh.subgroup_broadcast_first(base)
    return base + osh.subgroup_ballot_exclusive_bit_count(active_lanes)


@osh.function
def ordinarylight_enqueue_continuation(
    output_index: osh.u32,
    origin_tmin: osh.vec4,
    direction_tmax: osh.vec4,
    path_index: osh.u32,
    cone_width: osh.f32,
    cone_spread: osh.f32,
) -> osh.boolean:
    if output_index >= ordinarylight_output_queue.capacity:
        osh.atomic_add(ordinarylight_output_queue.overflow, osh.u32(1))
        return False
    ordinarylight_output_queue.rays[output_index].origin_tmin = origin_tmin
    ordinarylight_output_queue.rays[output_index].direction_tmax = direction_tmax
    ordinarylight_output_queue.rays[output_index].path_index = path_index
    ordinarylight_output_queue.rays[output_index].padding_a = osh.float_bits_to_uint(
        cone_width
    )
    ordinarylight_output_queue.rays[output_index].padding_b = osh.float_bits_to_uint(
        cone_spread
    )
    ordinarylight_output_queue.rays[output_index].padding_c = osh.u32(0)
    return True


@osh.function
def ordinarylight_secondary_vertex_position(
    primitive: osh.u32, corner: osh.u32,
) -> osh.vec3:
    return ordinarylight_vertices[primitive * osh.u32(3) + corner].xyz


@osh.function
def ordinarylight_secondary_vertex_attribute(
    primitive: osh.u32, corner: osh.u32,
) -> VertexAttributeData:
    return ordinarylight_attributes[primitive * osh.u32(3) + corner]


@osh.function
def ordinarylight_secondary_material(primitive: osh.u32) -> MaterialData:
    return ordinarylight_materials[primitive]


@osh.function
def ordinarylight_medium_ior(path_index: osh.u32, depth: osh.u32) -> osh.f32:
    return ordinarylight_medium_stacks[path_index].ior[depth]


@osh.function
def ordinarylight_set_medium_ior(
    path_index: osh.u32, depth: osh.u32, value: osh.f32,
) -> osh.void:
    ordinarylight_medium_stacks[path_index].ior[depth] = value


@osh.function
def ordinarylight_store_path(path_index: osh.u32, path: WavePathState) -> osh.void:
    ordinarylight_paths[path_index] = path


@osh.function
def ordinarylight_deactivate_stored_path(path_index: osh.u32) -> osh.void:
    ordinarylight_paths[path_index].metadata.w = (
        ordinarylight_paths[path_index].metadata.w & ~osh.u32(1)
    )


@osh.function
def ordinarylight_clear_secondary_path(path_index: osh.u32) -> osh.void:
    ordinarylight_secondary_paths[path_index].position_valid = osh.vec4(0.0)
    ordinarylight_secondary_paths[path_index].normal_pdf = osh.vec4(0.0)
    ordinarylight_secondary_paths[path_index].primary_throughput = osh.vec4(0.0)
    ordinarylight_secondary_paths[path_index].primary_radiance = osh.vec4(0.0)
    ordinarylight_secondary_paths[
        path_index
    ].diffuse_radiance_hit_distance = osh.vec4(0.0)
    ordinarylight_secondary_paths[
        path_index
    ].specular_radiance_hit_distance = osh.vec4(0.0)
    ordinarylight_secondary_paths[path_index].primary_position = osh.vec4(0.0)
    ordinarylight_secondary_paths[path_index].primary_geometry = osh.vec4(0.0)


@osh.function
def ordinarylight_secondary_primary_valid(path_index: osh.u32) -> osh.f32:
    return ordinarylight_secondary_paths[path_index].primary_throughput.w


@osh.function
def ordinarylight_store_secondary_hit(
    path_index: osh.u32, position_valid: osh.vec4, normal: osh.vec3,
) -> osh.void:
    ordinarylight_secondary_paths[path_index].position_valid = position_valid
    ordinarylight_secondary_paths[path_index].normal_pdf.xyz = normal


@osh.function
def ordinarylight_store_secondary_primary(
    path_index: osh.u32,
    throughput: osh.vec3,
    radiance: osh.vec3,
    pdf: osh.f32,
    sampled_specular: osh.f32,
    specular_probability: osh.f32,
    position: osh.vec3,
    roughness: osh.f32,
    primitive: osh.u32,
    barycentrics: osh.vec2,
) -> osh.void:
    ordinarylight_secondary_paths[path_index].primary_throughput = osh.vec4(
        throughput, 1.0 + sampled_specular
    )
    ordinarylight_secondary_paths[path_index].primary_radiance = osh.vec4(
        radiance, specular_probability
    )
    ordinarylight_secondary_paths[path_index].normal_pdf.w = pdf
    ordinarylight_secondary_paths[path_index].primary_position = osh.vec4(
        position, 1.0 + osh.clamp(roughness, 0.0, 1.0)
    )
    ordinarylight_secondary_paths[path_index].primary_geometry = osh.vec4(
        osh.uint_bits_to_float(primitive), barycentrics, 0.0
    )


@osh.function
def ordinarylight_integrate_secondary_volumes(
    origin: osh.vec3,
    direction: osh.vec3,
    distance: osh.f32,
    radiance: osh.inout(osh.vec3),
    throughput: osh.inout(osh.vec3),
) -> osh.void:
    integrateVolumesBeforeSurface(
        origin, direction, distance, radiance, throughput
    )


@osh.function
def ordinarylight_profile_work(counter: osh.u32, amount: osh.u32) -> osh.void:
    profileWork(counter, amount)


@osh.function
def ordinarylight_apply_material_program(
    material: osh.inout(MaterialData),
    normal: osh.vec3,
    uv: osh.vec2,
    direction: osh.vec3,
    entering: osh.boolean,
    primitive: osh.u32,
    weights: osh.vec3,
    bounce_index: osh.f32,
) -> MaterialEvaluationABI:
    return waveApplyMaterialProgram(
        material,
        normal,
        uv,
        direction,
        entering,
        primitive,
        weights,
        bounce_index,
    )


@osh.function
def ordinarylight_primary_hash(value: osh.u32) -> osh.u32:
    value = value ^ (value >> osh.u32(16))
    value = value * osh.u32(0x7FEB352D)
    value = value ^ (value >> osh.u32(15))
    value = value * osh.u32(0x846CA68B)
    value = value ^ (value >> osh.u32(16))
    return value


@osh.function
def ordinarylight_primary_rng_seed(
    pixel_index: osh.u32,
    frame_index: osh.u32,
    sample_index: osh.u32,
) -> osh.u32:
    return ordinarylight_primary_hash(
        pixel_index
        ^ ordinarylight_primary_hash(frame_index)
        ^ ordinarylight_primary_hash(sample_index + osh.u32(1))
    )


@osh.function
def ordinarylight_primary_rng_step(value: osh.u32) -> osh.u32:
    return ordinarylight_primary_hash(value)


@osh.function
def ordinarylight_primary_rng_value(value: osh.u32) -> osh.f32:
    return osh.f32(value) / 4294967296.0


@osh.function
def ordinarylight_primary_path_identity(
    frame_index: osh.u32,
    sample_index: osh.u32,
) -> osh.u32:
    return (frame_index << osh.u32(8)) | (sample_index & osh.u32(255))


@osh.function
def ordinarylight_primary_path_flags(
    indirect_capture: osh.boolean,
) -> osh.u32:
    flags = osh.u32(257)
    if indirect_capture:
        flags = flags | osh.u32(8)
    return flags


@osh.function
def ordinarylight_primary_ray_origin(
    origin: osh.vec3,
    right: osh.vec3,
    up: osh.vec3,
    ndc: osh.vec2,
    aspect: osh.f32,
    projection: osh.i32,
) -> osh.vec3:
    ray_origin = origin
    if projection == 1:
        ray_origin = ray_origin + ndc.x * aspect * right - ndc.y * up
    return ray_origin


@osh.function
def ordinarylight_primary_ray_direction(
    forward: osh.vec3,
    right: osh.vec3,
    up: osh.vec3,
    ndc: osh.vec2,
    aspect: osh.f32,
    projection: osh.i32,
) -> osh.vec3:
    if projection == 1:
        return osh.normalize(forward)
    if projection == 2:
        yaw = ndc.x * osh.length(right)
        pitch = -ndc.y * osh.length(up)
        return osh.normalize(
            osh.normalize(forward) * osh.cosine(pitch) * osh.cosine(yaw)
            + osh.normalize(right) * osh.cosine(pitch) * osh.sine(yaw)
            + osh.normalize(up) * osh.sine(pitch)
        )
    return osh.normalize(forward + ndc.x * aspect * right - ndc.y * up)


@osh.function
def ordinarylight_primary_barycentric_weights(
    barycentrics: osh.vec2,
) -> osh.vec3:
    return osh.vec3(
        1.0 - barycentrics.x - barycentrics.y,
        barycentrics.x,
        barycentrics.y,
    )


@osh.function
def ordinarylight_primary_hit_position(
    ray_origin: osh.vec3,
    incoming: osh.vec3,
    distance: osh.f32,
) -> osh.vec3:
    return ray_origin + distance * incoming


@osh.function
def ordinarylight_primary_geometric_normal(
    a: osh.vec3,
    b: osh.vec3,
    c: osh.vec3,
) -> osh.vec3:
    return osh.normalize(osh.cross(b - a, c - a))


@osh.function
def ordinarylight_primary_shading_normal(
    normal_a: osh.vec3,
    normal_b: osh.vec3,
    normal_c: osh.vec3,
    weights: osh.vec3,
    geometric_normal: osh.vec3,
) -> osh.vec3:
    shading_normal = osh.normalize(
        normal_a * weights.x
        + normal_b * weights.y
        + normal_c * weights.z
    )
    if osh.dot(shading_normal, geometric_normal) < 0.0:
        shading_normal = -shading_normal
    return shading_normal


@osh.function
def ordinarylight_primary_is_entering(
    incoming: osh.vec3,
    geometric_normal: osh.vec3,
) -> osh.boolean:
    return osh.dot(incoming, geometric_normal) < 0.0


@osh.function
def ordinarylight_primary_oriented_normal(
    shading_normal: osh.vec3,
    entering: osh.boolean,
) -> osh.vec3:
    if entering:
        return shading_normal
    return -shading_normal


@osh.function
def ordinarylight_primary_cone_spread(
    camera_up: osh.vec3,
    image_height: osh.u32,
) -> osh.f32:
    return 2.0 * osh.length(camera_up) / osh.f32(osh.maximum(
        image_height, osh.u32(1)
    ))


@osh.function
def ordinarylight_primary_surface_class(
    transmission: osh.f32,
    emissive: osh.f32,
    opaque_scene: osh.boolean,
) -> osh.f32:
    if not opaque_scene and transmission > 0.001:
        return 2.0
    if emissive > 0.5:
        return 1.0
    return 0.0


@osh.function
def ordinarylight_primary_interpolate_vec4(
    value_a: osh.vec4,
    value_b: osh.vec4,
    value_c: osh.vec4,
    weights: osh.vec3,
) -> osh.vec4:
    return (
        value_a * weights.x
        + value_b * weights.y
        + value_c * weights.z
    )


@osh.function
def ordinarylight_primary_uv_density(
    vertex_a: osh.vec3,
    vertex_b: osh.vec3,
    vertex_c: osh.vec3,
    uv_a: osh.vec2,
    uv_b: osh.vec2,
    uv_c: osh.vec2,
) -> osh.f32:
    first_uv = uv_b - uv_a
    second_uv = uv_c - uv_a
    uv_area = osh.absolute(
        first_uv.x * second_uv.y - first_uv.y * second_uv.x
    )
    world_area = osh.length(osh.cross(
        vertex_b - vertex_a, vertex_c - vertex_a
    ))
    return osh.sqrt(uv_area / osh.maximum(world_area, 0.00000001))


@osh.function
def ordinarylight_primary_triangle_tangent(
    vertex_a: osh.vec3,
    vertex_b: osh.vec3,
    vertex_c: osh.vec3,
    uv_a: osh.vec2,
    uv_b: osh.vec2,
    uv_c: osh.vec2,
    shading_normal: osh.vec3,
) -> osh.vec4:
    edge_a = vertex_b - vertex_a
    edge_b = vertex_c - vertex_a
    delta_a = uv_b - uv_a
    delta_b = uv_c - uv_a
    determinant = delta_a.x * delta_b.y - delta_a.y * delta_b.x
    if osh.absolute(determinant) < 0.00000001:
        fallback = osh.cross(shading_normal, osh.vec3(0.0, 1.0, 0.0))
        if osh.absolute(shading_normal.z) < 0.999:
            fallback = osh.cross(shading_normal, osh.vec3(0.0, 0.0, 1.0))
        return osh.vec4(osh.normalize(fallback), 1.0)
    inverse = 1.0 / determinant
    tangent = osh.normalize(
        (edge_a * delta_b.y - edge_b * delta_a.y) * inverse
    )
    bitangent = osh.normalize(
        (edge_b * delta_a.x - edge_a * delta_b.x) * inverse
    )
    handedness = 1.0
    if osh.dot(osh.cross(shading_normal, tangent), bitangent) < 0.0:
        handedness = -1.0
    return osh.vec4(tangent, handedness)


@osh.function
def ordinarylight_primary_correct_mapped_normal(
    mapped_normal: osh.vec3,
    geometric_normal: osh.vec3,
) -> osh.vec3:
    if osh.dot(mapped_normal, geometric_normal) < 0.0:
        return -mapped_normal
    return mapped_normal


@osh.function
def ordinarylight_texture_apply_rgb(
    base_value: osh.vec3,
    texture_value: osh.vec3,
) -> osh.vec3:
    return base_value * texture_value


@osh.function
def ordinarylight_texture_apply_scalar(
    base_value: osh.f32,
    texture_value: osh.f32,
) -> osh.f32:
    return base_value * texture_value


@osh.function
def ordinarylight_texture_apply_occlusion(
    occlusion: osh.f32,
    strength: osh.f32,
) -> osh.f32:
    return osh.mix(1.0, occlusion, strength)


@osh.function
def ordinarylight_texture_apply_normal(
    sampled_normal: osh.vec3,
    normal_scale: osh.f32,
    shading_normal: osh.vec3,
    tangent_data: osh.vec4,
    rotation: osh.vec2,
    texture_scale: osh.vec2,
) -> osh.vec3:
    tangent_normal = sampled_normal * 2.0 - 1.0
    tangent_normal.xy = tangent_normal.xy * normal_scale
    tangent_normal = osh.normalize(tangent_normal)
    tangent = osh.normalize(
        tangent_data.xyz
        - shading_normal * osh.dot(shading_normal, tangent_data.xyz)
    )
    bitangent = osh.cross(shading_normal, tangent) * tangent_data.w
    inverse_x = 1.0 / texture_scale.x
    inverse_y = 1.0 / texture_scale.y
    transformed_tangent = osh.normalize(
        tangent * (rotation.x * inverse_x)
        + bitangent * (-rotation.y * inverse_y)
    )
    transformed_bitangent = osh.normalize(
        tangent * (rotation.y * inverse_x)
        + bitangent * (rotation.x * inverse_y)
    )
    return osh.normalize(
        transformed_tangent * tangent_normal.x
        + transformed_bitangent * tangent_normal.y
        + shading_normal * tangent_normal.z
    )


@osh.function
def ordinarylight_primary_invalid_position() -> osh.vec4:
    return osh.vec4(-1.0)


@osh.function
def ordinarylight_primary_invalid_material() -> osh.uvec4:
    return osh.uvec4(osh.u32(0xFFFFFFFF))


@osh.function
def ordinarylight_primary_hit_position_payload(
    distance: osh.f32,
) -> osh.vec4:
    return osh.vec4(distance)


@osh.function
def ordinarylight_primary_packed_payload(value: osh.u32) -> osh.uvec4:
    return osh.uvec4(value)


@osh.function
def ordinarylight_primary_emission(
    emission: osh.vec3,
    entering: osh.boolean,
    two_sided: osh.boolean,
) -> osh.vec3:
    if entering or two_sided:
        return emission
    return osh.vec3(0.0)


@osh.function
def ordinarylight_primary_should_terminate(
    max_bounces: osh.u32,
) -> osh.boolean:
    return max_bounces <= osh.u32(1)


@osh.function
def ordinarylight_primary_deactivate(flags: osh.u32) -> osh.u32:
    return flags & ~osh.u32(1)


@osh.function
def ordinarylight_primary_transmission(
    material_transmission: osh.f32,
    opaque_scene: osh.boolean,
) -> osh.f32:
    if opaque_scene:
        return 0.0
    return material_transmission


@osh.function
def ordinarylight_primary_target_ior(material_ior: osh.f32) -> osh.f32:
    return osh.maximum(material_ior, 1.0001)


@osh.function
def ordinarylight_primary_refracted_direction(
    incoming: osh.vec3,
    normal: osh.vec3,
    target_ior: osh.f32,
) -> osh.vec3:
    return osh.refract(incoming, normal, 1.0 / target_ior)


@osh.function
def ordinarylight_primary_resolve_transmission_direction(
    refracted: osh.vec3,
    incoming: osh.vec3,
    normal: osh.vec3,
) -> osh.vec3:
    if osh.dot(refracted, refracted) < 0.01:
        return incoming - 2.0 * osh.dot(incoming, normal) * normal
    return refracted


@osh.function
def ordinarylight_primary_enters_medium(
    refracted: osh.vec3,
    entering: osh.boolean,
) -> osh.boolean:
    return entering and osh.dot(refracted, refracted) >= 0.01


@osh.function
def ordinarylight_primary_medium_depth(
    entered_medium: osh.boolean,
) -> osh.u32:
    if entered_medium:
        return osh.u32(2)
    return osh.u32(1)


@osh.function
def ordinarylight_primary_transmission_weight(
    base_color: osh.vec3,
    transmission: osh.f32,
) -> osh.vec3:
    return osh.mix(osh.vec3(1.0), base_color, 0.2) * transmission


@osh.function
def ordinarylight_primary_initial_medium_ior() -> osh.f32:
    return 1.0


@osh.function
def ordinarylight_primary_apply_bsdf_weight(
    throughput: osh.vec3,
    bsdf_weight: osh.vec3,
) -> osh.vec3:
    return throughput * bsdf_weight


@osh.function
def ordinarylight_primary_scattered_cone_spread(
    cone_spread: osh.f32,
    roughness: osh.f32,
) -> osh.f32:
    return cone_spread + roughness * 0.25


@osh.function
def ordinarylight_primary_continuation_direction(
    direction: osh.vec3,
) -> osh.vec3:
    return osh.normalize(direction)


@osh.function
def ordinarylight_primary_continuation_flags(
    previous_flags: osh.u32,
    medium_depth: osh.u32,
    transmission: osh.f32,
    unified_nee: osh.boolean,
) -> osh.u32:
    flags = osh.u32(1) | (medium_depth << osh.u32(8))
    flags = flags | (previous_flags & osh.u32(8))
    if transmission <= 0.001:
        flags = flags | osh.u32(2)
        if unified_nee:
            flags = flags | osh.u32(4)
    return flags


@osh.function
def ordinarylight_primary_previous_pdf(
    previous_pdf: osh.f32,
    bsdf_pdf: osh.f32,
    transmission: osh.f32,
) -> osh.f32:
    if transmission <= 0.001:
        return bsdf_pdf
    return previous_pdf


@osh.function
def ordinarylight_primary_capture_secondary(
    flags: osh.u32,
    transmission: osh.f32,
) -> osh.boolean:
    return (flags & osh.u32(8)) != osh.u32(0) and transmission <= 0.001


@osh.function
def ordinarylight_primary_continuation_origin(
    position: osh.vec3,
    direction: osh.vec3,
) -> osh.vec3:
    return position + direction * 0.002


@osh.function
def ordinarylight_primary_ray_origin_payload(origin: osh.vec3) -> osh.vec4:
    return osh.vec4(origin, 0.001)


@osh.function
def ordinarylight_primary_ray_direction_payload(
    direction: osh.vec3,
) -> osh.vec4:
    return osh.vec4(direction, 1.0e30)


@osh.function
def ordinarylight_pbr_cosine_hemisphere(
    normal: osh.vec3,
    random_u: osh.f32,
    random_v: osh.f32,
) -> osh.vec3:
    radius = osh.sqrt(random_u)
    phi = 6.28318530718 * random_v
    tangent = osh.cross(normal, osh.vec3(0.0, 1.0, 0.0))
    if osh.absolute(normal.z) < 0.999:
        tangent = osh.cross(normal, osh.vec3(0.0, 0.0, 1.0))
    tangent = osh.normalize(tangent)
    bitangent = osh.cross(normal, tangent)
    return osh.normalize(
        tangent * radius * osh.cosine(phi)
        + bitangent * radius * osh.sine(phi)
        + normal * osh.sqrt(osh.maximum(0.0, 1.0 - random_u))
    )


@osh.function
def ordinarylight_pbr_fresnel(f0: osh.vec3, cosine: osh.f32) -> osh.vec3:
    return f0 + (osh.vec3(1.0) - f0) * osh.power(
        1.0 - osh.clamp(cosine, 0.0, 1.0), 5.0
    )


@osh.function
def ordinarylight_pbr_ggx_distribution(
    normal_half: osh.f32,
    roughness: osh.f32,
) -> osh.f32:
    alpha = osh.maximum(roughness * roughness, 0.0009)
    alpha_squared = alpha * alpha
    denominator = normal_half * normal_half * (alpha_squared - 1.0) + 1.0
    return alpha_squared / osh.maximum(
        3.14159265359 * denominator * denominator, 0.000001
    )


@osh.function
def ordinarylight_pbr_ggx_smith(
    normal_direction: osh.f32,
    roughness: osh.f32,
) -> osh.f32:
    alpha = osh.maximum(roughness * roughness, 0.0009)
    alpha_squared = alpha * alpha
    return 2.0 * normal_direction / osh.maximum(
        normal_direction + osh.sqrt(
            alpha_squared
            + (1.0 - alpha_squared) * normal_direction * normal_direction
        ),
        0.000001,
    )


@osh.function
def ordinarylight_pbr_specular_probability(
    base_color: osh.vec3,
    metallic: osh.f32,
) -> osh.f32:
    f0 = osh.mix(osh.vec3(0.04), base_color, metallic)
    return osh.clamp(
        osh.maximum(f0.r, osh.maximum(f0.g, f0.b)), 0.1, 0.9
    )


@osh.function
def ordinarylight_pbr_evaluate(
    base_color: osh.vec3,
    roughness: osh.f32,
    metallic: osh.f32,
    normal: osh.vec3,
    view: osh.vec3,
    outgoing: osh.vec3,
) -> osh.vec3:
    normal_view = osh.maximum(osh.dot(normal, view), 0.0)
    normal_light = osh.maximum(osh.dot(normal, outgoing), 0.0)
    if normal_view <= 0.0 or normal_light <= 0.0:
        return osh.vec3(0.0)
    half_vector = osh.normalize(view + outgoing)
    normal_half = osh.maximum(osh.dot(normal, half_vector), 0.0)
    view_half = osh.maximum(osh.dot(view, half_vector), 0.0)
    f0 = osh.mix(osh.vec3(0.04), base_color, metallic)
    fresnel = f0 + (osh.vec3(1.0) - f0) * osh.power(
        1.0 - osh.clamp(view_half, 0.0, 1.0), 5.0
    )
    alpha = osh.maximum(roughness * roughness, 0.0009)
    alpha_squared = alpha * alpha
    distribution_denominator = (
        normal_half * normal_half * (alpha_squared - 1.0) + 1.0
    )
    distribution = alpha_squared / osh.maximum(
        3.14159265359 * distribution_denominator
        * distribution_denominator,
        0.000001,
    )
    view_geometry = 2.0 * normal_view / osh.maximum(
        normal_view + osh.sqrt(
            alpha_squared
            + (1.0 - alpha_squared) * normal_view * normal_view
        ),
        0.000001,
    )
    light_geometry = 2.0 * normal_light / osh.maximum(
        normal_light + osh.sqrt(
            alpha_squared
            + (1.0 - alpha_squared) * normal_light * normal_light
        ),
        0.000001,
    )
    geometry = view_geometry * light_geometry
    specular = fresnel * distribution * geometry / osh.maximum(
        4.0 * normal_view * normal_light, 0.000001
    )
    diffuse = (
        (osh.vec3(1.0) - fresnel) * (1.0 - metallic)
        * base_color / 3.14159265359
    )
    return diffuse + specular


@osh.function
def ordinarylight_pbr_pdf(
    base_color: osh.vec3,
    roughness: osh.f32,
    metallic: osh.f32,
    normal: osh.vec3,
    view: osh.vec3,
    outgoing: osh.vec3,
) -> osh.f32:
    normal_light = osh.maximum(osh.dot(normal, outgoing), 0.0)
    if normal_light <= 0.0:
        return 0.0
    half_vector = osh.normalize(view + outgoing)
    normal_half = osh.maximum(osh.dot(normal, half_vector), 0.0)
    view_half = osh.maximum(osh.dot(view, half_vector), 0.000001)
    alpha = osh.maximum(roughness * roughness, 0.0009)
    alpha_squared = alpha * alpha
    denominator = normal_half * normal_half * (alpha_squared - 1.0) + 1.0
    distribution = alpha_squared / osh.maximum(
        3.14159265359 * denominator * denominator, 0.000001
    )
    specular_pdf = distribution * normal_half / (4.0 * view_half)
    diffuse_pdf = normal_light / 3.14159265359
    f0 = osh.mix(osh.vec3(0.04), base_color, metallic)
    probability = osh.clamp(
        osh.maximum(f0.r, osh.maximum(f0.g, f0.b)), 0.1, 0.9
    )
    return osh.mix(
        diffuse_pdf, specular_pdf, probability,
    )


@osh.function
def ordinarylight_pbr_sample_half_vector(
    normal: osh.vec3,
    roughness: osh.f32,
    random_u: osh.f32,
    random_v: osh.f32,
) -> osh.vec3:
    alpha = osh.maximum(roughness * roughness, 0.0009)
    alpha_squared = alpha * alpha
    phi = 6.28318530718 * random_v
    cosine = osh.sqrt(
        (1.0 - random_u) / osh.maximum(
            1.0 + (alpha_squared - 1.0) * random_u, 0.000001
        )
    )
    sine = osh.sqrt(osh.maximum(0.0, 1.0 - cosine * cosine))
    tangent = osh.cross(normal, osh.vec3(0.0, 1.0, 0.0))
    if osh.absolute(normal.z) < 0.999:
        tangent = osh.cross(normal, osh.vec3(0.0, 0.0, 1.0))
    tangent = osh.normalize(tangent)
    bitangent = osh.cross(normal, tangent)
    return osh.normalize(
        tangent * sine * osh.cosine(phi)
        + bitangent * sine * osh.sine(phi)
        + normal * cosine
    )


@osh.function
def ordinarylight_pbr_reflect(
    incoming: osh.vec3,
    half_vector: osh.vec3,
) -> osh.vec3:
    return incoming - 2.0 * osh.dot(half_vector, incoming) * half_vector


@osh.function
def ordinarylight_pbr_weight(
    evaluated: osh.vec3,
    metallic: osh.f32,
    occlusion: osh.f32,
    normal: osh.vec3,
    view: osh.vec3,
    outgoing: osh.vec3,
    pdf: osh.f32,
) -> osh.vec3:
    weight = evaluated * osh.maximum(
        osh.dot(normal, outgoing), 0.0
    ) / pdf
    return weight * osh.mix(occlusion, 1.0, metallic)


@osh.function
def ordinarylight_analytic_light_direction(
    light_type: osh.i32,
    light_position: osh.vec3,
    light_direction: osh.vec3,
    hit: osh.vec3,
) -> osh.vec3:
    if light_type == 1:
        return -osh.normalize(light_direction)
    offset = light_position - hit
    distance_squared = osh.maximum(osh.dot(offset, offset), 0.000001)
    return offset / osh.sqrt(distance_squared)


@osh.function
def ordinarylight_analytic_light_distance_squared(
    light_type: osh.i32,
    light_position: osh.vec3,
    hit: osh.vec3,
) -> osh.f32:
    if light_type == 1:
        return 1.0
    offset = light_position - hit
    return osh.maximum(osh.dot(offset, offset), 0.000001)


@osh.function
def ordinarylight_analytic_light_attenuation(
    light_type: osh.i32,
    distance_squared: osh.f32,
    direction: osh.vec3,
    light_direction: osh.vec3,
    spot_inner: osh.f32,
    spot_outer: osh.f32,
) -> osh.f32:
    if light_type == 1:
        return 1.0
    attenuation = 1.0 / distance_squared
    if light_type == 2:
        cone = osh.dot(osh.normalize(light_direction), -direction)
        interpolation = osh.clamp(
            (cone - spot_outer) / (spot_inner - spot_outer), 0.0, 1.0
        )
        spot = interpolation * interpolation * (3.0 - 2.0 * interpolation)
        attenuation = attenuation * spot
    return attenuation


@osh.function
def ordinarylight_analytic_light_cosine(
    normal: osh.vec3,
    direction: osh.vec3,
) -> osh.f32:
    return osh.maximum(osh.dot(normal, direction), 0.0)


@osh.function
def ordinarylight_analytic_light_shadow_distance(
    light_type: osh.i32,
    distance_to_light: osh.f32,
) -> osh.f32:
    if light_type == 1:
        return 10000.0
    return osh.maximum(distance_to_light - 0.004, 0.001)


@osh.function
def ordinarylight_analytic_light_incident(
    color: osh.vec3,
    intensity: osh.f32,
    attenuation: osh.f32,
    transmittance: osh.f32,
) -> osh.vec3:
    return color * intensity * attenuation * transmittance


@osh.function
def ordinarylight_analytic_light_contribution(
    evaluated_pbr: osh.vec3,
    incident: osh.vec3,
    cosine: osh.f32,
) -> osh.vec3:
    return evaluated_pbr * incident * cosine


@osh.function
def ordinarylight_area_light_position(
    vertex_a: osh.vec3,
    vertex_b: osh.vec3,
    vertex_c: osh.vec3,
    root_u: osh.f32,
    value_v: osh.f32,
) -> osh.vec3:
    return (
        (1.0 - root_u) * vertex_a
        + root_u * (1.0 - value_v) * vertex_b
        + root_u * value_v * vertex_c
    )


@osh.function
def ordinarylight_area_light_barycentric_position(
    vertex_a: osh.vec3,
    vertex_b: osh.vec3,
    vertex_c: osh.vec3,
    barycentrics: osh.vec2,
) -> osh.vec3:
    a_weight = osh.maximum(
        1.0 - barycentrics.x - barycentrics.y, 0.0
    )
    return (
        a_weight * vertex_a
        + barycentrics.x * vertex_b
        + barycentrics.y * vertex_c
    )


@osh.function
def ordinarylight_area_light_cosine(
    vertex_a: osh.vec3,
    vertex_b: osh.vec3,
    vertex_c: osh.vec3,
    direction: osh.vec3,
    two_sided: osh.boolean,
) -> osh.f32:
    light_normal = osh.normalize(osh.cross(
        vertex_b - vertex_a, vertex_c - vertex_a
    ))
    raw_cosine = osh.dot(light_normal, -direction)
    if two_sided:
        return osh.absolute(raw_cosine)
    return osh.maximum(raw_cosine, 0.0)


@osh.function
def ordinarylight_area_light_pdf(
    selection_pdf: osh.f32,
    distance_squared: osh.f32,
    light_cosine: osh.f32,
    area: osh.f32,
    technique_probability: osh.f32,
) -> osh.f32:
    light_pdf = selection_pdf * distance_squared / osh.maximum(
        light_cosine * area, 0.000001
    )
    return light_pdf * osh.maximum(technique_probability, 0.000001)


@osh.function
def ordinarylight_area_light_mis(
    effective_pdf: osh.f32,
    sample_count: osh.f32,
    bsdf_pdf: osh.f32,
) -> osh.f32:
    first_pdf = effective_pdf * sample_count
    first_squared = first_pdf * first_pdf
    second_squared = bsdf_pdf * bsdf_pdf
    return first_squared / osh.maximum(
        first_squared + second_squared, 0.000001
    )


@osh.function
def ordinarylight_area_light_contribution(
    evaluated_pbr: osh.vec3,
    emission: osh.vec3,
    surface_cosine: osh.f32,
    mis: osh.f32,
    transmittance: osh.f32,
    effective_pdf: osh.f32,
) -> osh.vec3:
    return (
        evaluated_pbr * emission * surface_cosine * mis * transmittance
        / osh.maximum(effective_pdf, 0.000001)
    )


@osh.function
def ordinarylight_environment_analytic(direction: osh.vec3) -> osh.vec3:
    sky = osh.maximum(direction.y, 0.0)
    return osh.mix(
        osh.vec3(0.018, 0.022, 0.032),
        osh.vec3(0.32, 0.46, 0.72), sky,
    )


@osh.function
def ordinarylight_environment_uv(
    direction: osh.vec3,
    rotation: osh.f32,
) -> osh.vec2:
    longitude = osh.arctangent2(direction.z, direction.x) + rotation
    return osh.vec2(
        osh.fraction(longitude * 0.15915494309189535 + 0.5),
        osh.arccosine(osh.clamp(direction.y, -1.0, 1.0))
        * 0.3183098861837907,
    )


@osh.function
def ordinarylight_environment_radiance(
    encoded: osh.vec3,
    exposure: osh.f32,
    color: osh.vec3,
    intensity: osh.f32,
    textured: osh.boolean,
) -> osh.vec3:
    radiance = osh.vec3(1.0)
    if textured:
        radiance = osh.exp2(encoded * exposure) - osh.vec3(1.0)
    return radiance * color * intensity


@osh.function
def ordinarylight_environment_effective_pdf(
    cosine: osh.f32,
    technique_probability: osh.f32,
) -> osh.f32:
    pdf = cosine / 3.14159265359
    return pdf * osh.maximum(technique_probability, 0.000001)


@osh.function
def ordinarylight_environment_mis(
    cosine: osh.f32,
    effective_pdf: osh.f32,
    sample_count: osh.f32,
) -> osh.f32:
    first_pdf = effective_pdf * sample_count
    second_pdf = cosine / 3.14159265359
    first_squared = first_pdf * first_pdf
    second_squared = second_pdf * second_pdf
    return first_squared / osh.maximum(
        first_squared + second_squared, 0.000001
    )


@osh.function
def ordinarylight_environment_contribution(
    evaluated_pbr: osh.vec3,
    radiance: osh.vec3,
    cosine: osh.f32,
    mis: osh.f32,
    transmittance: osh.f32,
    effective_pdf: osh.f32,
    occlusion: osh.f32,
    metallic: osh.f32,
) -> osh.vec3:
    material_weight = osh.mix(occlusion, 1.0, metallic)
    return (
        evaluated_pbr * radiance * cosine * mis * transmittance
        / osh.maximum(effective_pdf, 0.000001) * material_weight
    )


@osh.function
def ordinarylight_environment_encode_direction(
    direction_input: osh.vec3,
) -> osh.vec2:
    direction = direction_input / (
        osh.absolute(direction_input.x) + osh.absolute(direction_input.y)
        + osh.absolute(direction_input.z)
    )
    encoded = direction.xy
    if direction.z < 0.0:
        encoded = (
            osh.vec2(1.0) - osh.absolute(encoded.yx)
        ) * osh.sign(encoded.xy)
    return encoded * 0.5 + osh.vec2(0.5)


@osh.function
def ordinarylight_environment_decode_direction(
    encoded: osh.vec2,
) -> osh.vec3:
    value = encoded * 2.0 - osh.vec2(1.0)
    direction = osh.vec3(
        value, 1.0 - osh.absolute(value.x) - osh.absolute(value.y)
    )
    if direction.z < 0.0:
        direction.xy = (
            osh.vec2(1.0) - osh.absolute(direction.yx)
        ) * osh.sign(direction.xy)
    return osh.normalize(direction)


@osh.function
def ordinarylight_unified_area_probability(
    area_light_count: osh.u32,
    environment_samples: osh.u32,
    area_light_weight: osh.f32,
) -> osh.f32:
    if area_light_count == osh.u32(0):
        return 0.0
    if environment_samples == osh.u32(0):
        return 1.0
    area_weight = osh.sqrt(osh.maximum(area_light_weight, 0.000001))
    return area_weight / (area_weight + 1.0)


@osh.function
def ordinarylight_unified_secondary_area_probability(
    area_enabled: osh.boolean,
    environment_enabled: osh.boolean,
    area_light_weight: osh.f32,
    secondary_area_samples: osh.u32,
    environment_samples: osh.u32,
) -> osh.f32:
    area_weight = 0.0
    if area_enabled:
        area_weight = (
            osh.sqrt(osh.maximum(area_light_weight, 0.000001))
            * osh.f32(osh.maximum(secondary_area_samples, osh.u32(1)))
        )
    environment_weight = 0.0
    if environment_enabled:
        environment_weight = osh.f32(
            osh.minimum(environment_samples, osh.u32(4))
        )
    return area_weight / osh.maximum(
        area_weight + environment_weight, 0.000001
    )


@osh.function
def ordinarylight_secondary_nee_hash(value: osh.u32) -> osh.u32:
    value = value ^ (value >> osh.u32(16))
    value = value * osh.u32(0x7FEB352D)
    value = value ^ (value >> osh.u32(15))
    value = value * osh.u32(0x846CA68B)
    return value ^ (value >> osh.u32(16))


@osh.function
def ordinarylight_secondary_nee_select(
    probability: osh.f32,
    pixel_index: osh.u32,
    frame_sample: osh.u32,
    bounce: osh.u32,
) -> osh.boolean:
    if probability >= 0.999999:
        return True
    frame_index = frame_sample >> osh.u32(8)
    sample_index = frame_sample & osh.u32(255)
    bounce_hash = bounce + osh.u32(1)
    bounce_hash = bounce_hash ^ (bounce_hash >> osh.u32(16))
    bounce_hash = bounce_hash * osh.u32(0x7FEB352D)
    bounce_hash = bounce_hash ^ (bounce_hash >> osh.u32(15))
    bounce_hash = bounce_hash * osh.u32(0x846CA68B)
    bounce_hash = bounce_hash ^ (bounce_hash >> osh.u32(16))
    sample_hash = sample_index + osh.u32(1)
    sample_hash = sample_hash ^ (sample_hash >> osh.u32(16))
    sample_hash = sample_hash * osh.u32(0x7FEB352D)
    sample_hash = sample_hash ^ (sample_hash >> osh.u32(15))
    sample_hash = sample_hash * osh.u32(0x846CA68B)
    sample_hash = sample_hash ^ (sample_hash >> osh.u32(16))
    scramble = pixel_index ^ bounce_hash ^ sample_hash
    scramble = scramble ^ (scramble >> osh.u32(16))
    scramble = scramble * osh.u32(0x7FEB352D)
    scramble = scramble ^ (scramble >> osh.u32(15))
    scramble = scramble * osh.u32(0x846CA68B)
    scramble = scramble ^ (scramble >> osh.u32(16))
    sequence = osh.bitfield_reverse(frame_index) ^ scramble
    selector = (osh.f32(sequence) + 0.5) * (1.0 / 4294967296.0)
    return selector < probability


@osh.function
def ordinarylight_light_candidate_target(contribution: osh.vec3) -> osh.f32:
    return osh.maximum(
        osh.dot(contribution, osh.vec3(0.2126, 0.7152, 0.0722)), 0.0
    )


@osh.function
def ordinarylight_environment_miss_mis(
    previous_diffuse: osh.boolean,
    environment_samples: osh.u32,
    previous_pdf: osh.f32,
    unified_nee: osh.boolean,
    area_probability: osh.f32,
) -> osh.f32:
    if not previous_diffuse or environment_samples == osh.u32(0):
        return 1.0
    light_pdf = previous_pdf * osh.f32(environment_samples)
    if unified_nee:
        light_pdf = previous_pdf * (1.0 - area_probability)
    path_squared = previous_pdf * previous_pdf
    light_squared = light_pdf * light_pdf
    return path_squared / osh.maximum(
        path_squared + light_squared, 0.000001
    )


@osh.function
def ordinarylight_emissive_hit_mis(
    previous_diffuse: osh.boolean,
    emission: osh.vec3,
    vertex_a: osh.vec3,
    vertex_b: osh.vec3,
    vertex_c: osh.vec3,
    geometric_normal: osh.vec3,
    incoming: osh.vec3,
    hit_distance: osh.f32,
    two_sided: osh.boolean,
    area_light_weight: osh.f32,
    secondary_area_samples: osh.u32,
    unified_nee: osh.boolean,
    area_probability: osh.f32,
    previous_pdf: osh.f32,
) -> osh.f32:
    if not previous_diffuse or osh.dot(emission, emission) <= 0.0:
        return 1.0
    area = 0.5 * osh.length(osh.cross(
        vertex_b - vertex_a, vertex_c - vertex_a
    ))
    raw_cosine = osh.dot(geometric_normal, -incoming)
    light_cosine = osh.maximum(raw_cosine, 0.0)
    if two_sided:
        light_cosine = osh.absolute(raw_cosine)
    luminance = osh.dot(emission, osh.vec3(0.2126, 0.7152, 0.0722))
    selection_pdf = (
        area * luminance / osh.maximum(area_light_weight, 0.000001)
    )
    light_pdf = (
        selection_pdf * hit_distance * hit_distance
        / osh.maximum(light_cosine * area, 0.000001)
    )
    sampled_light_pdf = light_pdf * osh.f32(
        osh.maximum(secondary_area_samples, osh.u32(1))
    )
    if unified_nee:
        sampled_light_pdf = light_pdf * area_probability
    path_squared = previous_pdf * previous_pdf
    light_squared = sampled_light_pdf * sampled_light_pdf
    return path_squared / osh.maximum(
        path_squared + light_squared, 0.000001
    )


@osh.function
def ordinarylight_emission_contribution(
    throughput: osh.vec3,
    emission: osh.vec3,
    mis: osh.f32,
) -> osh.vec3:
    return throughput * emission * mis


@osh.function
def ordinarylight_secondary_direct_contribution(
    throughput: osh.vec3,
    direct: osh.vec3,
    nee_probability: osh.f32,
) -> osh.vec3:
    return throughput * direct / osh.maximum(nee_probability, 0.000001)


@osh.function
def ordinarylight_secondary_scatter_throughput(
    throughput: osh.vec3,
    bsdf_weight: osh.vec3,
) -> osh.vec3:
    return throughput * bsdf_weight


@osh.function
def ordinarylight_secondary_roulette_enabled(
    roulette_start: osh.u32,
    next_bounce: osh.u32,
    transmission: osh.f32,
) -> osh.boolean:
    return (
        roulette_start > osh.u32(0)
        and next_bounce >= roulette_start
        and transmission <= 0.001
    )


@osh.function
def ordinarylight_secondary_survival_probability(
    throughput: osh.vec3,
    minimum_survival: osh.f32,
) -> osh.f32:
    return osh.clamp(
        osh.maximum(throughput.r, osh.maximum(throughput.g, throughput.b)),
        minimum_survival, 0.95,
    )


@osh.function
def ordinarylight_secondary_survives(
    random_value: osh.f32,
    survival_probability: osh.f32,
) -> osh.boolean:
    return random_value < survival_probability


@osh.function
def ordinarylight_secondary_survival_throughput(
    throughput: osh.vec3,
    survival_probability: osh.f32,
) -> osh.vec3:
    return throughput / survival_probability


@osh.function
def ordinarylight_secondary_target_ior(
    entering: osh.boolean,
    material_ior: osh.f32,
    previous_medium_ior: osh.f32,
    medium_depth: osh.u32,
) -> osh.f32:
    if entering:
        return osh.maximum(material_ior, 1.0001)
    if medium_depth > osh.u32(1):
        return previous_medium_ior
    return 1.0


@osh.function
def ordinarylight_secondary_refracted_direction(
    incoming: osh.vec3,
    normal: osh.vec3,
    current_ior: osh.f32,
    target_ior: osh.f32,
) -> osh.vec3:
    return osh.refract(incoming, normal, current_ior / target_ior)


@osh.function
def ordinarylight_secondary_enters_medium(
    refracted: osh.vec3,
    entering: osh.boolean,
    medium_depth: osh.u32,
    maximum_depth: osh.u32,
) -> osh.boolean:
    return (
        osh.dot(refracted, refracted) >= 0.01
        and entering
        and medium_depth < maximum_depth
    )


@osh.function
def ordinarylight_secondary_medium_depth(
    refracted: osh.vec3,
    entering: osh.boolean,
    medium_depth: osh.u32,
    maximum_depth: osh.u32,
) -> osh.u32:
    if osh.dot(refracted, refracted) < 0.01:
        return medium_depth
    if entering and medium_depth < maximum_depth:
        return medium_depth + osh.u32(1)
    if not entering and medium_depth > osh.u32(1):
        return medium_depth - osh.u32(1)
    return medium_depth


@osh.function
def ordinarylight_secondary_transmission_throughput(
    throughput: osh.vec3,
    base_color: osh.vec3,
    transmission: osh.f32,
) -> osh.vec3:
    tint = osh.mix(osh.vec3(1.0), base_color, 0.2)
    return throughput * tint * transmission


@osh.function
def ordinarylight_secondary_cone_width(
    cone_width: osh.f32,
    distance: osh.f32,
    cone_spread: osh.f32,
) -> osh.f32:
    return cone_width + distance * cone_spread


@osh.function
def ordinarylight_secondary_texture_footprint(
    cone_width: osh.f32,
    uv_density: osh.f32,
    textured: osh.boolean,
) -> osh.f32:
    if not textured:
        return 0.0
    return cone_width * uv_density


@osh.function
def ordinarylight_secondary_correct_shading_normal(
    shading_normal: osh.vec3,
    geometric_normal: osh.vec3,
) -> osh.vec3:
    if osh.dot(shading_normal, geometric_normal) < 0.0:
        return -shading_normal
    return shading_normal


@osh.function
def ordinarylight_secondary_stop_bounce(
    hybrid: osh.boolean,
    inline_bounces: osh.u32,
    maximum_bounces: osh.u32,
) -> osh.u32:
    if hybrid:
        return osh.minimum(inline_bounces, maximum_bounces)
    return maximum_bounces


@osh.function
def ordinarylight_secondary_throughput_visible(
    throughput: osh.vec3,
) -> osh.boolean:
    return osh.maximum(
        throughput.r, osh.maximum(throughput.g, throughput.b)
    ) >= 0.0001


@osh.function
def ordinarylight_secondary_miss_contribution(
    throughput: osh.vec3,
    environment: osh.vec3,
    mis: osh.f32,
) -> osh.vec3:
    return throughput * environment * mis


@osh.function
def ordinarylight_secondary_next_bounce(bounce: osh.u32) -> osh.u32:
    return bounce + osh.u32(1)


@osh.function
def ordinarylight_secondary_bounce_terminates(
    next_bounce: osh.u32,
    maximum_bounces: osh.u32,
) -> osh.boolean:
    return next_bounce >= maximum_bounces


@osh.function
def ordinarylight_secondary_capture_hit(
    flags: osh.u32,
    bounce: osh.u32,
    primary_capture_valid: osh.f32,
) -> osh.boolean:
    return (
        (flags & osh.u32(8)) != osh.u32(0)
        and bounce == osh.u32(1)
        and primary_capture_valid > 0.5
    )


@osh.function
def ordinarylight_secondary_ser_hint(
    transmission: osh.f32,
    metallic: osh.f32,
    two_sided: osh.f32,
    roughness: osh.f32,
    textured: osh.boolean,
) -> osh.u32:
    hint = osh.u32(transmission > 0.001)
    hint = hint | (osh.u32(metallic > 0.5) << osh.u32(1))
    hint = hint | (osh.u32(two_sided > 0.5) << osh.u32(2))
    roughness_bucket = osh.u32(osh.clamp(roughness * 7.0, 0.0, 7.0))
    hint = hint | (roughness_bucket << osh.u32(3))
    hint = hint | (osh.u32(textured) << osh.u32(6))
    return hint


@osh.function
def ordinarylight_secondary_nee_probability(value: osh.f32) -> osh.f32:
    return osh.clamp(value, 0.000001, 1.0)


@osh.function
def ordinarylight_secondary_area_sample_count(value: osh.u32) -> osh.u32:
    return osh.clamp(value, osh.u32(1), osh.u32(16))


@osh.function
def ordinarylight_secondary_environment_sample_count(
    value: osh.u32,
) -> osh.u32:
    return osh.minimum(value, osh.u32(4))


@osh.function
def ordinarylight_secondary_average_contribution(
    contribution: osh.vec3,
    sample_count: osh.u32,
) -> osh.vec3:
    return contribution / osh.f32(sample_count)


@osh.function
def ordinarylight_secondary_emission_visible(
    entering: osh.boolean,
    two_sided: osh.f32,
) -> osh.boolean:
    return entering or two_sided > 0.5


@osh.function
def ordinarylight_secondary_capture_position(position: osh.vec3) -> osh.vec4:
    return osh.vec4(position, 1.0)


def generated_source():
    generated = "\n".join((
        osh.compile_function(
            ordinarylight_secondary_trace_query,
            external_values={
                "scene_tlas": osh.opaque_type("accelerationStructureEXT")
            },
        ).source.rstrip(),
        "#if WAVE_SER\n"
        + osh.compile_function(
            ordinarylight_secondary_reorder,
            capabilities=("shader_reorder",),
        ).source.rstrip()
        + "\n#endif",
        "#if WAVE_ORDINARYSHADE_SECONDARY_ORCHESTRATION\n"
        + osh.compile_function(
            ordinarylight_integrate_secondary_volumes,
            externals=(integrateVolumesBeforeSurface,),
        ).source.rstrip()
        + "\n#endif",
        "#if WAVE_ORDINARYSHADE_SECONDARY_ORCHESTRATION && WAVE_WORK_COUNTERS\n"
        + osh.compile_function(
            ordinarylight_profile_work,
            externals=(profileWork,),
        ).source.rstrip()
        + "\n#endif",
        "#if WAVE_CUSTOM_MATERIAL_PROGRAM\n"
        + osh.compile_function(
            ordinarylight_apply_material_program,
            externals=(waveApplyMaterialProgram,),
        ).source.rstrip()
        + "\n#endif",
        "#if WAVE_ORDINARYSHADE_SECONDARY_ORCHESTRATION && WAVE_PERSISTENT_COARSE\n"
        + osh.compile_function(
            ordinarylight_persistent_coarse_schedule,
            externals=(processPrimaryPixel,),
            external_values={
                "ordinarylight_output_queue": OrdinarylightOutputQueueABI,
            },
        ).source.rstrip()
        + "\n#endif",
        "#if WAVE_ORDINARYSHADE_SECONDARY_ORCHESTRATION\n"
        + osh.compile_function(
            ordinarylight_trace_remaining,
            externals=(ordinarylightSecondaryBounce,),
        ).source.rstrip()
        + "\n#endif",
        "#if WAVE_ORDINARYSHADE_SECONDARY_ORCHESTRATION\n"
        + "\n".join(
            osh.compile_function(
                helper,
                external_values={
                    "ordinarylight_paths": osh.runtime_array(WavePathState),
                    "ordinarylight_secondary_paths": osh.runtime_array(
                        SecondaryPathState
                    ),
                },
            ).source.rstrip()
            for helper in (
                ordinarylight_store_path,
                ordinarylight_deactivate_stored_path,
                ordinarylight_clear_secondary_path,
                ordinarylight_secondary_primary_valid,
                ordinarylight_store_secondary_hit,
                ordinarylight_store_secondary_primary,
            )
        )
        + "\n#endif",
        "#if WAVE_ORDINARYSHADE_SECONDARY_ORCHESTRATION\n"
        + osh.compile_function(
            ordinarylight_medium_ior,
            external_values={
                "ordinarylight_medium_stacks": osh.runtime_array(WaveMediumStack)
            },
        ).source.rstrip()
        + "\n"
        + osh.compile_function(
            ordinarylight_set_medium_ior,
            external_values={
                "ordinarylight_medium_stacks": osh.runtime_array(WaveMediumStack)
            },
        ).source.rstrip()
        + "\n#endif",
        osh.compile_function(
            ordinarylight_primary_scheduled_group
        ).source.rstrip(),
        "#if WAVE_ORDINARYSHADE_SECONDARY_ORCHESTRATION\n"
        + osh.compile_function(
            ordinarylight_reserve_output_index,
            external_values={"ordinarylight_output_queue_count": osh.u32},
            capabilities=("subgroup_ballot",),
        ).source.rstrip()
        + "\n#endif",
        "#if WAVE_ORDINARYSHADE_SECONDARY_ORCHESTRATION\n"
        + osh.compile_function(
            ordinarylight_secondary_vertex_position,
            external_values={
                "ordinarylight_vertices": osh.runtime_array(osh.vec4)
            },
        ).source.rstrip()
        + "\n"
        + osh.compile_function(
            ordinarylight_secondary_vertex_attribute,
            external_values={
                "ordinarylight_attributes": osh.runtime_array(VertexAttributeData)
            },
        ).source.rstrip()
        + "\n"
        + osh.compile_function(
            ordinarylight_secondary_material,
            external_values={
                "ordinarylight_materials": osh.runtime_array(MaterialData)
            },
        ).source.rstrip()
        + "\n#endif",
        "#if WAVE_ORDINARYSHADE_SECONDARY_ORCHESTRATION\n"
        + osh.compile_function(
            ordinarylight_enqueue_continuation,
            external_values={
                "ordinarylight_output_queue": OrdinarylightOutputQueueABI
            },
        ).source.rstrip()
        + "\n#endif",
        osh.compile_function(ordinarylight_primary_hash).source.rstrip(),
        osh.compile_function(ordinarylight_primary_rng_seed).source.rstrip(),
        osh.compile_function(ordinarylight_primary_rng_step).source.rstrip(),
        osh.compile_function(ordinarylight_primary_rng_value).source.rstrip(),
        osh.compile_function(ordinarylight_primary_path_identity).source.rstrip(),
        osh.compile_function(ordinarylight_primary_path_flags).source.rstrip(),
        osh.compile_function(ordinarylight_primary_ray_origin).source.rstrip(),
        osh.compile_function(ordinarylight_primary_ray_direction).source.rstrip(),
        osh.compile_function(
            ordinarylight_primary_barycentric_weights
        ).source.rstrip(),
        osh.compile_function(ordinarylight_primary_hit_position).source.rstrip(),
        osh.compile_function(
            ordinarylight_primary_geometric_normal
        ).source.rstrip(),
        osh.compile_function(
            ordinarylight_primary_shading_normal
        ).source.rstrip(),
        osh.compile_function(ordinarylight_primary_is_entering).source.rstrip(),
        osh.compile_function(
            ordinarylight_primary_oriented_normal
        ).source.rstrip(),
        osh.compile_function(ordinarylight_primary_cone_spread).source.rstrip(),
        osh.compile_function(
            ordinarylight_primary_surface_class
        ).source.rstrip(),
        osh.compile_function(
            ordinarylight_primary_interpolate_vec4
        ).source.rstrip(),
        osh.compile_function(
            ordinarylight_primary_uv_density
        ).source.rstrip(),
        osh.compile_function(
            ordinarylight_primary_triangle_tangent
        ).source.rstrip(),
        osh.compile_function(
            ordinarylight_primary_correct_mapped_normal
        ).source.rstrip(),
        osh.compile_function(
            ordinarylight_texture_apply_rgb
        ).source.rstrip(),
        osh.compile_function(
            ordinarylight_texture_apply_scalar
        ).source.rstrip(),
        osh.compile_function(
            ordinarylight_texture_apply_occlusion
        ).source.rstrip(),
        osh.compile_function(
            ordinarylight_texture_apply_normal
        ).source.rstrip(),
        osh.compile_function(
            ordinarylight_primary_invalid_position
        ).source.rstrip(),
        osh.compile_function(
            ordinarylight_primary_invalid_material
        ).source.rstrip(),
        osh.compile_function(
            ordinarylight_primary_hit_position_payload
        ).source.rstrip(),
        osh.compile_function(
            ordinarylight_primary_packed_payload
        ).source.rstrip(),
        osh.compile_function(
            ordinarylight_primary_emission
        ).source.rstrip(),
        osh.compile_function(
            ordinarylight_primary_should_terminate
        ).source.rstrip(),
        osh.compile_function(
            ordinarylight_primary_deactivate
        ).source.rstrip(),
        osh.compile_function(
            ordinarylight_primary_transmission
        ).source.rstrip(),
        osh.compile_function(
            ordinarylight_primary_target_ior
        ).source.rstrip(),
        osh.compile_function(
            ordinarylight_primary_refracted_direction
        ).source.rstrip(),
        osh.compile_function(
            ordinarylight_primary_resolve_transmission_direction
        ).source.rstrip(),
        osh.compile_function(
            ordinarylight_primary_enters_medium
        ).source.rstrip(),
        osh.compile_function(
            ordinarylight_primary_medium_depth
        ).source.rstrip(),
        osh.compile_function(
            ordinarylight_primary_transmission_weight
        ).source.rstrip(),
        osh.compile_function(
            ordinarylight_primary_initial_medium_ior
        ).source.rstrip(),
        osh.compile_function(
            ordinarylight_primary_apply_bsdf_weight
        ).source.rstrip(),
        osh.compile_function(
            ordinarylight_primary_scattered_cone_spread
        ).source.rstrip(),
        osh.compile_function(
            ordinarylight_primary_continuation_direction
        ).source.rstrip(),
        osh.compile_function(
            ordinarylight_primary_continuation_flags
        ).source.rstrip(),
        osh.compile_function(
            ordinarylight_primary_previous_pdf
        ).source.rstrip(),
        osh.compile_function(
            ordinarylight_primary_capture_secondary
        ).source.rstrip(),
        osh.compile_function(
            ordinarylight_primary_continuation_origin
        ).source.rstrip(),
        osh.compile_function(
            ordinarylight_primary_ray_origin_payload
        ).source.rstrip(),
        osh.compile_function(
            ordinarylight_primary_ray_direction_payload
        ).source.rstrip(),
        osh.compile_function(
            ordinarylight_pbr_cosine_hemisphere
        ).source.rstrip(),
        osh.compile_function(ordinarylight_pbr_fresnel).source.rstrip(),
        osh.compile_function(
            ordinarylight_pbr_ggx_distribution
        ).source.rstrip(),
        osh.compile_function(ordinarylight_pbr_ggx_smith).source.rstrip(),
        osh.compile_function(
            ordinarylight_pbr_specular_probability
        ).source.rstrip(),
        osh.compile_function(ordinarylight_pbr_evaluate).source.rstrip(),
        osh.compile_function(ordinarylight_pbr_pdf).source.rstrip(),
        osh.compile_function(
            ordinarylight_pbr_sample_half_vector
        ).source.rstrip(),
        osh.compile_function(ordinarylight_pbr_reflect).source.rstrip(),
        osh.compile_function(ordinarylight_pbr_weight).source.rstrip(),
        osh.compile_function(
            ordinarylight_analytic_light_direction
        ).source.rstrip(),
        osh.compile_function(
            ordinarylight_analytic_light_distance_squared
        ).source.rstrip(),
        osh.compile_function(
            ordinarylight_analytic_light_attenuation
        ).source.rstrip(),
        osh.compile_function(
            ordinarylight_analytic_light_cosine
        ).source.rstrip(),
        osh.compile_function(
            ordinarylight_analytic_light_shadow_distance
        ).source.rstrip(),
        osh.compile_function(
            ordinarylight_analytic_light_incident
        ).source.rstrip(),
        osh.compile_function(
            ordinarylight_analytic_light_contribution
        ).source.rstrip(),
        osh.compile_function(
            ordinarylight_area_light_position
        ).source.rstrip(),
        osh.compile_function(
            ordinarylight_area_light_barycentric_position
        ).source.rstrip(),
        osh.compile_function(
            ordinarylight_area_light_cosine
        ).source.rstrip(),
        osh.compile_function(
            ordinarylight_area_light_pdf
        ).source.rstrip(),
        osh.compile_function(
            ordinarylight_area_light_mis
        ).source.rstrip(),
        osh.compile_function(
            ordinarylight_area_light_contribution
        ).source.rstrip(),
        osh.compile_function(
            ordinarylight_environment_analytic
        ).source.rstrip(),
        osh.compile_function(
            ordinarylight_environment_uv
        ).source.rstrip(),
        osh.compile_function(
            ordinarylight_environment_radiance
        ).source.rstrip(),
        osh.compile_function(
            ordinarylight_environment_effective_pdf
        ).source.rstrip(),
        osh.compile_function(
            ordinarylight_environment_mis
        ).source.rstrip(),
        osh.compile_function(
            ordinarylight_environment_contribution
        ).source.rstrip(),
        osh.compile_function(
            ordinarylight_environment_encode_direction
        ).source.rstrip(),
        osh.compile_function(
            ordinarylight_environment_decode_direction
        ).source.rstrip(),
        osh.compile_function(
            ordinarylight_unified_area_probability
        ).source.rstrip(),
        osh.compile_function(
            ordinarylight_unified_secondary_area_probability
        ).source.rstrip(),
        osh.compile_function(
            ordinarylight_secondary_nee_hash
        ).source.rstrip(),
        osh.compile_function(
            ordinarylight_secondary_nee_select
        ).source.rstrip(),
        osh.compile_function(
            ordinarylight_light_candidate_target
        ).source.rstrip(),
        osh.compile_function(
            ordinarylight_environment_miss_mis
        ).source.rstrip(),
        osh.compile_function(
            ordinarylight_emissive_hit_mis
        ).source.rstrip(),
        osh.compile_function(
            ordinarylight_emission_contribution
        ).source.rstrip(),
        osh.compile_function(
            ordinarylight_secondary_direct_contribution
        ).source.rstrip(),
        osh.compile_function(
            ordinarylight_secondary_scatter_throughput
        ).source.rstrip(),
        osh.compile_function(
            ordinarylight_secondary_roulette_enabled
        ).source.rstrip(),
        osh.compile_function(
            ordinarylight_secondary_survival_probability
        ).source.rstrip(),
        osh.compile_function(
            ordinarylight_secondary_survives
        ).source.rstrip(),
        osh.compile_function(
            ordinarylight_secondary_survival_throughput
        ).source.rstrip(),
        osh.compile_function(
            ordinarylight_secondary_target_ior
        ).source.rstrip(),
        osh.compile_function(
            ordinarylight_secondary_refracted_direction
        ).source.rstrip(),
        osh.compile_function(
            ordinarylight_secondary_enters_medium
        ).source.rstrip(),
        osh.compile_function(
            ordinarylight_secondary_medium_depth
        ).source.rstrip(),
        osh.compile_function(
            ordinarylight_secondary_transmission_throughput
        ).source.rstrip(),
        osh.compile_function(
            ordinarylight_secondary_cone_width
        ).source.rstrip(),
        osh.compile_function(
            ordinarylight_secondary_texture_footprint
        ).source.rstrip(),
        osh.compile_function(
            ordinarylight_secondary_correct_shading_normal
        ).source.rstrip(),
        osh.compile_function(
            ordinarylight_secondary_stop_bounce
        ).source.rstrip(),
        osh.compile_function(
            ordinarylight_secondary_throughput_visible
        ).source.rstrip(),
        osh.compile_function(
            ordinarylight_secondary_miss_contribution
        ).source.rstrip(),
        osh.compile_function(
            ordinarylight_secondary_next_bounce
        ).source.rstrip(),
        osh.compile_function(
            ordinarylight_secondary_bounce_terminates
        ).source.rstrip(),
        osh.compile_function(
            ordinarylight_secondary_capture_hit
        ).source.rstrip(),
        osh.compile_function(
            ordinarylight_secondary_ser_hint
        ).source.rstrip(),
        osh.compile_function(
            ordinarylight_secondary_nee_probability
        ).source.rstrip(),
        osh.compile_function(
            ordinarylight_secondary_area_sample_count
        ).source.rstrip(),
        osh.compile_function(
            ordinarylight_secondary_environment_sample_count
        ).source.rstrip(),
        osh.compile_function(
            ordinarylight_secondary_average_contribution
        ).source.rstrip(),
        osh.compile_function(
            ordinarylight_secondary_emission_visible
        ).source.rstrip(),
        osh.compile_function(
            ordinarylight_secondary_capture_position
        ).source.rstrip(),
    ))
    return (
        "// Generated by scripts/generate_primary_shaders.py using Ordinary Shade.\n"
        "// Edit the typed Python source, not this generated GLSL.\n"
        "#ifndef ORDINARYLIGHT_ORDINARYSHADE_PRIMARY_GLSL\n"
        "#define ORDINARYLIGHT_ORDINARYSHADE_PRIMARY_GLSL 1\n"
        f"{generated.rstrip()}\n"
        "#endif\n"
    )


def main():
    output = ROOT / "ordinarylight/shaders/ordinaryshade_primary.glsl"
    source = generated_source()
    if "--check" in sys.argv:
        if output.read_text() != source:
            raise SystemExit(f"{output} is stale; regenerate it")
        print(f"Verified {output.relative_to(ROOT)}")
        return
    output.write_text(source)
    print(f"Wrote {output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
