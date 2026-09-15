"""Combined HDR resolve and sampled ReLAX preparation in OrdinaryShade.

Opt-in entry point; no reservoir seeding or legacy signal resolve. Keep signal
and guide behavior aligned with kernels.prepare_relax_signals (parity tested).
"""
import ordinaryshade as osh
from .kernels import (WavePathState, SecondaryPathState, PrepareCamera,
    PrepareConstants, prepare_unpack_normal, prepare_surface_history,
    prepare_previous_pixel)

@osh.compute(workgroup_size=(64, 1, 1))
def prepare_relax_signals_with_hdr(
    paths: osh.storage_buffer(WavePathState, access="read", binding=0),
    secondary_paths: osh.storage_buffer(
        SecondaryPathState, access="read", binding=1,
    ),
    packed_normal: osh.storage_image("r32ui", access="read", binding=2),
    packed_material: osh.storage_image("r32ui", access="read", binding=3),
    diffuse_output: osh.storage_image("rgba16f", binding=4),
    specular_output: osh.storage_image("rgba16f", binding=5),
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
    hdr_output: osh.storage_image("rgba16f", binding=14),
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
    # Same accumulation and rounding as the standalone path-to-HDR resolve.
    contribution = paths[path_index].radiance.rgb / osh.f32(constants.samples.y)
    accumulated = osh.vec3(0.0)
    if constants.samples.x != osh.u32(0):
        accumulated = hdr_output.load(pixel).rgb
    hdr_output.store(pixel, osh.vec4(accumulated + contribution, 1.0))
    secondary = secondary_paths[path_index]
    if constants.samples.z != osh.u32(0):
        # Classify this sample before averaging; secondary records are reset
        # by primary generation and cannot represent the whole sample batch.
        resolved = osh.maximum(paths[path_index].radiance.rgb, osh.vec3(0.0))
        primary = osh.minimum(
            osh.maximum(secondary.primary_radiance.rgb, osh.vec3(0.0)), resolved,
        )
        indirect = resolved - primary
        probability = osh.clamp(secondary.primary_radiance.w, 0.0, 1.0)
        diffuse = primary * (1.0 - probability)
        specular = primary * probability
        if secondary.specular_radiance_hit_distance.w < 0.0:
            specular = osh.minimum(
                osh.maximum(secondary.specular_radiance_hit_distance.rgb, osh.vec3(0.0)),
                primary,
            )
            diffuse = primary - specular
        indirect_fraction = osh.vec3(0.0)
        if secondary.primary_throughput.w >= 1.5:
            indirect_fraction = osh.vec3(1.0)
        if secondary.diffuse_radiance_hit_distance.w < 0.0:
            indirect_fraction = osh.clamp(
                secondary.diffuse_radiance_hit_distance.rgb,
                osh.vec3(0.0), osh.vec3(1.0),
            )
        specular = specular + indirect * indirect_fraction
        diffuse = diffuse + indirect * (osh.vec3(1.0) - indirect_fraction)
        distance = 0.0
        if secondary.primary_position.w > 0.5 and secondary.position_valid.w > 0.5:
            distance = osh.length(
                secondary.position_valid.xyz - secondary.primary_position.xyz
            )
        # Distance is still a last-sample guide, not an averaged path length.
        diffuse_distance = 0.0
        specular_distance = 0.0
        if osh.any_value(indirect_fraction > osh.vec3(0.0)):
            specular_distance = distance
        if osh.any_value(indirect_fraction < osh.vec3(1.0)):
            diffuse_distance = distance
        scale = 1.0 / osh.f32(osh.maximum(constants.samples.y, osh.u32(1)))
        diffuse = diffuse * scale
        specular = specular * scale
        if constants.samples.x != osh.u32(0):
            diffuse = diffuse + diffuse_output.load(pixel).rgb
            specular = specular + specular_output.load(pixel).rgb
        diffuse_output.store(pixel, osh.vec4(diffuse, diffuse_distance))
        specular_output.store(pixel, osh.vec4(specular, specular_distance))
    # Radiance accumulates across samples above, but geometry guides are
    # consumed only after the sample batch and represent its last sample.
    # Avoid reprojecting and overwriting those images for earlier samples.
    if constants.samples.x + osh.u32(1) < osh.maximum(constants.samples.y, osh.u32(1)):
        return
    valid = secondary.primary_position.w > 0.5
    if not valid:
        if constants.samples.z == osh.u32(0):
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
    surface_history = prepare_surface_history(secondary, pixel_index)
    instance_key = surface_history.identity
    previous_world_position = surface_history.previous_position
    transmissive = (osh.float_bits_to_uint(secondary.primary_geometry.y) & osh.u32(0x80000000)) != osh.u32(0)
    # Explicit static z=0 planar-mirror experiment. The first secondary hit
    # is reflected into virtual world space; no moving-object transform is
    # available for that hit yet. Preserve primary identity/material.
    if (
        constants.samples.w != osh.u32(0)
        and osh.absolute(world_position.z) < 0.0001
        and osh.absolute(normal.z) > 0.9999
        and roughness <= 0.0011
    ):
        if secondary.position_valid.w > 0.5:
            reflected = secondary.position_valid.xyz
            world_position = osh.vec3(reflected.x, reflected.y, -reflected.z)
            reflected_normal = secondary.normal_pdf.xyz
            normal = osh.vec3(
                reflected_normal.x, reflected_normal.y, -reflected_normal.z,
            )
            previous_world_position = world_position
            view_z = osh.dot(
                world_position - current_camera.origin.xyz,
                current_camera.forward.xyz,
            )
        else:
            # No finite secondary hit: do not reuse a prior reflected surface.
            view_z = 0.0
    old = prepare_previous_pixel(previous_world_position, extent)
    motion = old.xy - osh.vec2(pixel)
    previous_view_z = old.z
    if (
        not surface_history.valid or old.z <= 0.0001 or osh.any_value(old.xy < osh.vec2(-0.5))
        or osh.any_value(old.xy >= osh.vec2(extent) - 0.5)
    ):
        motion = osh.vec2(0.0)
        previous_view_z = 0.0
    if constants.samples.z == osh.u32(0):
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
    transmission_cap = 0.0
    if constants.extent_paths.w != osh.u32(0) and transmissive:
        transmission_cap = 1.0
    motion_output.store(pixel, osh.vec4(motion, previous_view_z, transmission_cap))
    # Native triangles retain instance continuity across tessellation edges.
    # Custom surfaces use the application identity fingerprint supplied by
    # prepare_surface_history, independently of the acceleration primitive.
    identity_output.store(pixel, osh.uvec4(instance_key, 0, 0, 0))
