"""Ordinary Shade source for the portable spatiotemporal denoiser.

These functions are the GPU source of truth.  Build tooling compiles them to
SPIR-V and WGSL; :mod:`ordinarylight.denoising.portable` is the deterministic
CPU oracle used to verify their behavior.
"""

import ordinaryshade as osh


@osh.structure
class WavePathState:
    throughput: osh.vec4
    radiance: osh.vec4
    metadata: osh.uvec4


@osh.structure
class SecondaryPathState:
    position_valid: osh.vec4
    normal_pdf: osh.vec4
    primary_throughput: osh.vec4
    primary_radiance: osh.vec4
    diffuse_radiance_hit_distance: osh.vec4
    specular_radiance_hit_distance: osh.vec4
    primary_position: osh.vec4
    primary_geometry: osh.vec4


@osh.structure
class PrepareCamera:
    origin: osh.vec4
    forward: osh.vec4
    right: osh.vec4
    up: osh.vec4


@osh.structure
class PrepareConstants:
    extent_paths: osh.uvec4


@osh.function
def prepare_decode_normal(encoded: osh.vec2) -> osh.vec3:
    normal = osh.vec3(
        encoded, 1.0 - osh.absolute(encoded.x) - osh.absolute(encoded.y)
    )
    if normal.z < 0.0:
        folded = (1.0 - osh.absolute(normal.yx)) * osh.sign(normal.xy)
        normal.x = folded.x
        normal.y = folded.y
    return osh.normalize(normal)


@osh.function
def prepare_unpack_normal(packed: osh.u32) -> osh.vec3:
    unit = osh.vec2(
        osh.f32(packed & osh.u32(0x7FFF)),
        osh.f32((packed >> osh.u32(15)) & osh.u32(0x7FFF)),
    ) / 32767.0
    return prepare_decode_normal(unit * 2.0 - 1.0)


@osh.function
def prepare_previous_pixel(
    world_position: osh.vec3, extent: osh.ivec2,
) -> osh.vec3:
    offset = world_position - previous_camera.origin.xyz
    depth = osh.dot(offset, previous_camera.forward.xyz)
    vertical_scale = osh.length(previous_camera.up.xyz)
    aspect = osh.f32(extent.x) / osh.f32(extent.y)
    if depth <= 0.0001 or vertical_scale <= 0.0001:
        return osh.vec3(-1.0, -1.0, depth)
    ndc = osh.vec2(
        osh.dot(offset, osh.normalize(previous_camera.right.xyz))
        / (depth * aspect * vertical_scale),
        -osh.dot(offset, osh.normalize(previous_camera.up.xyz))
        / (depth * vertical_scale),
    )
    pixel = (ndc * 0.5 + 0.5) * osh.vec2(extent) - 0.5
    return osh.vec3(pixel, depth)


@osh.compute(workgroup_size=(64, 1, 1))
def prepare_relax_signals(
    paths: osh.storage_buffer(WavePathState, access="read", binding=0),
    secondary_paths: osh.storage_buffer(
        SecondaryPathState, access="read", binding=1,
    ),
    packed_normal: osh.storage_image("r32ui", access="read", binding=2),
    packed_material: osh.storage_image("r32ui", access="read", binding=3),
    diffuse_output: osh.storage_image("rgba16f", access="write", binding=4),
    specular_output: osh.storage_image("rgba16f", access="write", binding=5),
    normal_roughness_output: osh.storage_image(
        "rgba16f", access="write", binding=6,
    ),
    view_z_output: osh.storage_image("r32f", access="write", binding=7),
    motion_output: osh.storage_image("rgba16f", access="write", binding=8),
    current_camera: osh.storage_record(
        PrepareCamera, access="read", binding=9,
    ),
    previous_camera: osh.storage_record(
        PrepareCamera, access="read", binding=10,
    ),
    previous_vertices: osh.storage_buffer(
        osh.vec4, access="read", binding=11,
    ),
    identity_output: osh.storage_image(
        "r32ui", access="write", binding=12,
    ),
    constants: osh.push_constants(PrepareConstants, wgsl_binding=13),
):
    path_index = osh.global_invocation_id.x
    path_count = constants.extent_paths.z
    if path_index >= path_count:
        return
    extent = osh.ivec2(constants.extent_paths.xy)
    pixel_index = paths[path_index].metadata.x
    if pixel_index >= osh.u32(extent.x * extent.y):
        return
    pixel = osh.ivec2(
        osh.i32(pixel_index % osh.u32(extent.x)),
        osh.i32(pixel_index / osh.u32(extent.x)),
    )
    secondary = secondary_paths[path_index]
    valid = secondary.primary_position.w > 0.5
    if not valid:
        diffuse_output.store(pixel, osh.vec4(0.0))
        specular_output.store(pixel, osh.vec4(0.0))
        normal_roughness_output.store(pixel, osh.vec4(0.0))
        view_z_output.store(pixel, osh.vec4(0.0))
        motion_output.store(pixel, osh.vec4(0.0))
        return
    world_position = secondary.primary_position.xyz
    normal = prepare_unpack_normal(packed_normal.load(pixel).x)
    roughness = osh.clamp(secondary.primary_position.w - 1.0, 0.0, 1.0)
    view_z = osh.dot(
        world_position - current_camera.origin.xyz,
        current_camera.forward.xyz,
    )
    primitive = osh.float_bits_to_uint(secondary.primary_geometry.x)
    barycentrics = secondary.primary_geometry.yz
    weights = osh.vec3(
        1.0 - barycentrics.x - barycentrics.y,
        barycentrics.x, barycentrics.y,
    )
    previous_world_position = (
        previous_vertices[primitive * osh.u32(3)].xyz * weights.x
        + previous_vertices[primitive * osh.u32(3) + osh.u32(1)].xyz * weights.y
        + previous_vertices[primitive * osh.u32(3) + osh.u32(2)].xyz * weights.z
    )
    old = prepare_previous_pixel(previous_world_position, extent)
    motion = old.xy - osh.vec2(pixel)
    previous_view_z = old.z
    if (
        old.z <= 0.0001 or osh.any_value(old.xy < osh.vec2(-0.5))
        or osh.any_value(old.xy >= osh.vec2(extent) - 0.5)
    ):
        motion = osh.vec2(0.0)
        previous_view_z = 0.0
    diffuse_output.store(
        pixel, secondary.diffuse_radiance_hit_distance,
    )
    specular_output.store(
        pixel, secondary.specular_radiance_hit_distance,
    )
    normal_roughness_output.store(pixel, osh.vec4(normal, roughness))
    view_z_output.store(pixel, osh.vec4(view_z))
    # Carry the expected previous-camera depth alongside the screen-space
    # motion.  Temporal validation must compare values in the same camera
    # space; current ``view_z`` and previous-frame ``view_z`` are not directly
    # comparable while the camera moves.
    motion_output.store(pixel, osh.vec4(motion, previous_view_z, 0.0))
    identity_output.store(pixel, osh.uvec4(primitive, 0, 0, 0))


@osh.structure
class TemporalConstants:
    extent_history: osh.vec4
    rejection: osh.vec4


@osh.compute(workgroup_size=(8, 8, 1))
def relax_temporal(
    current_radiance_hit_distance: osh.storage_image(
        "rgba16f", access="read", binding=0,
    ),
    normal_roughness: osh.storage_image(
        "rgba16f", access="read", binding=1,
    ),
    view_z: osh.storage_image("r32f", access="read", binding=2),
    motion: osh.storage_image("rgba16f", access="read", binding=3),
    material_id: osh.storage_image("r32ui", access="read", binding=4),
    previous_radiance: osh.storage_image(
        "rgba16f", access="read", binding=5,
    ),
    previous_normal_roughness: osh.storage_image(
        "rgba16f", access="read", binding=6,
    ),
    previous_view_z: osh.storage_image("r32f", access="read", binding=7),
    previous_material_id: osh.storage_image(
        "r32ui", access="read", binding=8,
    ),
    previous_history_length: osh.storage_image(
        "r32f", access="read", binding=9,
    ),
    output_radiance: osh.storage_image(
        "rgba16f", access="write", binding=10,
    ),
    output_history_length: osh.storage_image(
        "r32f", access="write", binding=11,
    ),
    constants: osh.uniform_buffer(TemporalConstants, binding=12),
    identity: osh.storage_image(
        "r32ui", access="read", binding=13,
    ),
    previous_identity: osh.storage_image(
        "r32ui", access="read", binding=14,
    ),
):
    pixel = osh.ivec2(osh.global_invocation_id.xy)
    extent = osh.ivec2(constants.extent_history.xy)
    if pixel.x >= extent.x or pixel.y >= extent.y:
        return
    current = current_radiance_hit_distance.load(pixel)
    current_normal = normal_roughness.load(pixel).xyz
    current_depth = view_z.load(pixel).r
    motion_sample = motion.load(pixel)
    motion_vector = motion_sample.xy
    expected_old_depth = motion_sample.z
    previous_pixel = osh.ivec2(osh.vec2(pixel) + motion_vector + osh.vec2(0.5))
    in_bounds = (
        previous_pixel.x >= 0 and previous_pixel.y >= 0
        and previous_pixel.x < extent.x and previous_pixel.y < extent.y
    )
    # ``extent_history.w`` is an explicit validity bit.  Newly allocated or
    # invalidated history images contain unspecified device memory, so bounds
    # and geometry tests alone must never make their contents eligible.
    accepted = (
        constants.extent_history.w > 0.5
        and in_bounds and current_depth != 0.0 and expected_old_depth > 0.0
    )
    history = current
    history_length = 1.0
    if accepted:
        old_depth = previous_view_z.load(previous_pixel).r
        old_normal = previous_normal_roughness.load(previous_pixel).xyz
        old_material = previous_material_id.load(previous_pixel).r
        old_primitive = previous_identity.load(previous_pixel).r
        current_primitive = identity.load(pixel).r
        current_material = material_id.load(pixel).r
        depth_tolerance = osh.maximum(
            osh.absolute(expected_old_depth) * constants.rejection.y, 0.001,
        )
        accepted = (
            old_depth != 0.0
            and osh.dot(current_normal, old_normal) >= constants.rejection.x
            and osh.absolute(expected_old_depth - old_depth) <= depth_tolerance
            and old_material == current_material
            and old_primitive == current_primitive
        )
        if accepted:
            history = previous_radiance.load(previous_pixel)
            neighborhood_sum = osh.vec3(0.0)
            neighborhood_square_sum = osh.vec3(0.0)
            neighborhood_count = 0.0
            for y in range(-1, 2):
                for x in range(-1, 2):
                    neighbor_pixel = pixel + osh.ivec2(x, y)
                    if neighbor_pixel.x < 0 or neighbor_pixel.y < 0:
                        continue
                    if neighbor_pixel.x >= extent.x or neighbor_pixel.y >= extent.y:
                        continue
                    neighbor = current_radiance_hit_distance.load(
                        neighbor_pixel
                    ).rgb
                    neighborhood_sum = neighborhood_sum + neighbor
                    neighborhood_square_sum = (
                        neighborhood_square_sum + neighbor * neighbor
                    )
                    neighborhood_count = neighborhood_count + 1.0
            neighborhood_mean = neighborhood_sum / osh.maximum(
                neighborhood_count, 1.0
            )
            neighborhood_variance = osh.maximum(
                neighborhood_square_sum / osh.maximum(
                    neighborhood_count, 1.0
                ) - neighborhood_mean * neighborhood_mean,
                osh.vec3(0.0),
            )
            neighborhood_deviation = osh.sqrt(neighborhood_variance)
            clamp_radius = neighborhood_deviation * constants.rejection.z
            if constants.rejection.w > 0.0:
                history_luma = osh.dot(
                    history.rgb, osh.vec3(0.2126, 0.7152, 0.0722)
                )
                mean_luma = osh.dot(
                    neighborhood_mean, osh.vec3(0.2126, 0.7152, 0.0722)
                )
                deviation_luma = osh.dot(
                    neighborhood_deviation,
                    osh.vec3(0.2126, 0.7152, 0.0722),
                )
                reactive_limit = osh.maximum(
                    deviation_luma * constants.rejection.w,
                    osh.absolute(mean_luma) * 0.1 + 0.01,
                )
                accepted = (
                    osh.absolute(history_luma - mean_luma) <= reactive_limit
                )
            history = osh.vec4(
                osh.clamp(
                    history.rgb,
                    neighborhood_mean - clamp_radius,
                    neighborhood_mean + clamp_radius,
                ),
                history.a,
            )
            if accepted:
                history_length = osh.minimum(
                    previous_history_length.load(previous_pixel).r + 1.0,
                    constants.extent_history.z,
                )
    if accepted:
        alpha = 1.0 / osh.maximum(history_length, 1.0)
        current = osh.vec4(
            osh.mix(history.rgb, current.rgb, alpha), current.a,
        )
    output_radiance.store(pixel, current)
    output_history_length.store(pixel, osh.vec4(history_length))


@osh.structure
class ComposeConstants:
    extent: osh.vec4


@osh.compute(workgroup_size=(8, 8, 1))
def relax_compose(
    diffuse: osh.storage_image("rgba16f", access="read", binding=0),
    specular: osh.storage_image("rgba16f", access="read", binding=1),
    view_z: osh.storage_image("r32f", access="read", binding=2),
    output_hdr: osh.storage_image("rgba16f", access="write", binding=3),
    constants: osh.push_constants(ComposeConstants, wgsl_binding=4),
):
    pixel = osh.ivec2(osh.global_invocation_id.xy)
    extent = osh.ivec2(constants.extent.xy)
    if pixel.x >= extent.x or pixel.y >= extent.y:
        return
    # Preserve the path tracer's environment/background.  Prepared surface
    # signals are defined only where the primary guide depth is non-zero.
    if view_z.load(pixel).r == 0.0:
        return
    diffuse_value = diffuse.load(pixel)
    specular_value = specular.load(pixel)
    output_hdr.store(pixel, osh.vec4(
        diffuse_value.rgb + specular_value.rgb, 1.0,
    ))


@osh.structure
class AtrousConstants:
    extent_step: osh.vec4
    weights: osh.vec4


@osh.compute(workgroup_size=(8, 8, 1))
def relax_atrous(
    input_radiance: osh.storage_image(
        "rgba16f", access="read", binding=0,
    ),
    normal_roughness: osh.storage_image(
        "rgba16f", access="read", binding=1,
    ),
    view_z: osh.storage_image("r32f", access="read", binding=2),
    material_id: osh.storage_image("r32ui", access="read", binding=3),
    output_radiance: osh.storage_image(
        "rgba16f", access="write", binding=4,
    ),
    constants: osh.push_constants(AtrousConstants, wgsl_binding=5),
):
    pixel = osh.ivec2(osh.global_invocation_id.xy)
    extent = osh.ivec2(constants.extent_step.xy)
    if pixel.x >= extent.x or pixel.y >= extent.y:
        return
    step_width = osh.i32(constants.extent_step.z)
    center = input_radiance.load(pixel)
    center_normal = normal_roughness.load(pixel).xyz
    center_depth = view_z.load(pixel).r
    center_material = material_id.load(pixel).r
    center_luma = osh.dot(center.rgb, osh.vec3(0.2126, 0.7152, 0.0722))
    total = center.rgb
    weight_sum = 1.0
    for y in range(-1, 2):
        for x in range(-1, 2):
            if x == 0 and y == 0:
                continue
            sample_pixel = pixel + osh.ivec2(x, y) * step_width
            if sample_pixel.x < 0 or sample_pixel.y < 0:
                continue
            if sample_pixel.x >= extent.x or sample_pixel.y >= extent.y:
                continue
            sample_depth = view_z.load(sample_pixel).r
            sample_material = material_id.load(sample_pixel).r
            if sample_material != center_material:
                continue
            if (sample_depth == 0.0) != (center_depth == 0.0):
                continue
            sample_normal = normal_roughness.load(sample_pixel).xyz
            sample = input_radiance.load(sample_pixel)
            normal_weight = osh.power(
                osh.maximum(osh.dot(center_normal, sample_normal), 0.0),
                constants.weights.x,
            )
            depth_scale = osh.maximum(
                osh.absolute(center_depth) * constants.weights.y, 0.001,
            )
            depth_weight = osh.exp(
                -osh.absolute(sample_depth - center_depth) / depth_scale,
            )
            sample_luma = osh.dot(
                sample.rgb, osh.vec3(0.2126, 0.7152, 0.0722),
            )
            color_scale = osh.maximum(
                osh.absolute(center_luma) / constants.weights.z, 0.02,
            )
            color_weight = osh.exp(
                -osh.absolute(sample_luma - center_luma) / color_scale,
            )
            kernel = 0.25
            if x == 0 or y == 0:
                kernel = 0.5
            weight = kernel * normal_weight * depth_weight * color_weight
            total = total + sample.rgb * weight
            weight_sum = weight_sum + weight
    output_radiance.store(
        pixel, osh.vec4(total / osh.maximum(weight_sum, 0.000001), center.a),
    )


__all__ = [
    "AtrousConstants", "ComposeConstants", "PrepareCamera", "PrepareConstants",
    "SecondaryPathState", "TemporalConstants", "WavePathState",
    "prepare_relax_signals", "relax_atrous", "relax_compose", "relax_temporal",
]
