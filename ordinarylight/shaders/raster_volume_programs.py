"""Portable fullscreen dense-volume pass authored with Ordinary Shade."""

import ordinaryshade as osh


@osh.structure
class VolumeVertexOutput:
    position: osh.invariant(osh.builtin(osh.vec4, "position"))
    uv: osh.location(osh.vec2, 0)


@osh.structure
class RasterVolumeCamera:
    inverse_view_projection: osh.mat4
    camera_position: osh.vec4
    viewport_steps: osh.vec4
    volume_count: osh.uvec4


@osh.structure
class RasterVolumeHeader:
    world_to_local: osh.mat4
    dimensions_offset: osh.uvec4
    value_parameters: osh.vec4
    render_parameters: osh.vec4
    scattering_parameters: osh.vec4
    phase_parameters: osh.vec4
    multiple_scattering_parameters: osh.vec4
    acceleration_parameters: osh.uvec4


@osh.structure
class RasterVolumeLight:
    position_type: osh.vec4
    direction_range: osh.vec4
    color_intensity: osh.vec4
    spot: osh.vec4


@osh.vertex
def volume_vertex(
    clip_position: osh.location(osh.vec2, 0),
    texture_coordinate: osh.location(osh.vec2, 1),
) -> VolumeVertexOutput:
    return VolumeVertexOutput(
        osh.vec4(clip_position, 0.0, 1.0),
        texture_coordinate,
    )


@osh.fragment
def volume_fragment(
    uv: osh.location(osh.vec2, 0),
    camera: osh.uniform_buffer(RasterVolumeCamera, binding=0),
    headers: osh.storage_buffer(RasterVolumeHeader, access="read", binding=1),
    transfers: osh.storage_buffer(osh.vec4, access="read", binding=2),
    scene_color: osh.sampled_texture_2d(binding=3),
    scene_depth: osh.sampled_depth_texture_2d(binding=4),
    volume_0: osh.sampled_texture_3d(binding=5),
    volume_1: osh.sampled_texture_3d(binding=6),
    volume_2: osh.sampled_texture_3d(binding=7),
    volume_3: osh.sampled_texture_3d(binding=8),
    linear_sampler: osh.sampler(binding=9),
    depth_sampler: osh.sampler(binding=10),
    lights: osh.storage_buffer(RasterVolumeLight, access="read", binding=11),
) -> osh.location(osh.vec4, 0):
    background = scene_color.sample_level_with(linear_sampler, uv, 0.0)
    opaque_depth = scene_depth.sample_depth_with(depth_sampler, uv)
    # Fullscreen vertices provide top-left-oriented texture coordinates on
    # both targets. The camera matrix follows OpenGL clip coordinates where
    # +Y is the top of the image, so only ray reconstruction flips UV Y.
    clip_xy = osh.vec2(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0)
    near_clip = osh.vec4(clip_xy, 0.0, 1.0)
    far_clip = osh.vec4(clip_xy, 1.0, 1.0)
    near_world_h = camera.inverse_view_projection * near_clip
    far_world_h = camera.inverse_view_projection * far_clip
    near_world = near_world_h.xyz / osh.maximum(osh.absolute(near_world_h.w), 1.0e-6)
    far_world = far_world_h.xyz / osh.maximum(osh.absolute(far_world_h.w), 1.0e-6)
    ray_origin = camera.camera_position.xyz
    ray_direction = osh.normalize(far_world - near_world)
    opaque_world_h = camera.inverse_view_projection * osh.vec4(
        clip_xy, opaque_depth, 1.0,
    )
    opaque_world = opaque_world_h.xyz / osh.maximum(
        osh.absolute(opaque_world_h.w), 1.0e-6,
    )
    ray_limit = osh.length(opaque_world - ray_origin)
    if opaque_depth >= 0.999999:
        ray_limit = osh.length(far_world - ray_origin)

    entry = ray_limit
    exit_distance = 0.0
    step_size = 1.0e30
    volume_count = osh.minimum(camera.volume_count.x, osh.u32(4))
    for volume_index in range(4):
        if osh.u32(volume_index) >= volume_count:
            break
        header = headers[osh.u32(volume_index)]
        local_origin = (header.world_to_local * osh.vec4(ray_origin, 1.0)).xyz
        local_direction = (header.world_to_local * osh.vec4(ray_direction, 0.0)).xyz
        inverse_direction = osh.vec3(1.0) / (local_direction + osh.sign(local_direction) * 1.0e-8)
        # Volume transforms map the canonical voxel domain [0, 1]^3 into
        # world space.  This must match the GI volume traversal and the
        # public Volume transform contract; treating the bounds as a
        # centered cube displaces the medium by half of its transformed
        # extent.
        first = (osh.vec3(0.0) - local_origin) * inverse_direction
        second = (osh.vec3(1.0) - local_origin) * inverse_direction
        lower = osh.minimum(first, second)
        upper = osh.maximum(first, second)
        volume_entry = osh.maximum(osh.maximum(lower.x, lower.y), lower.z)
        volume_exit = osh.minimum(osh.minimum(upper.x, upper.y), upper.z)
        if volume_exit > osh.maximum(volume_entry, 0.0):
            entry = osh.minimum(entry, osh.maximum(volume_entry, 0.0))
            exit_distance = osh.maximum(exit_distance, osh.minimum(volume_exit, ray_limit))
            step_size = osh.minimum(step_size, header.render_parameters.x)

    if exit_distance <= entry:
        return background
    step_size = osh.maximum(step_size * camera.viewport_steps.z, 1.0e-5)
    transmittance = 1.0
    radiance = osh.vec3(0.0)
    distance = entry + step_size * 0.5
    max_steps = osh.minimum(osh.u32(camera.viewport_steps.w), osh.u32(8192))
    for step in range(8192):
        if osh.u32(step) >= max_steps or distance >= exit_distance or transmittance <= 0.001:
            break
        combined_extinction = 0.0
        combined_emission = osh.vec3(0.0)
        world_position = ray_origin + ray_direction * distance
        for volume_index in range(4):
            if osh.u32(volume_index) >= volume_count:
                break
            header = headers[osh.u32(volume_index)]
            local = (header.world_to_local * osh.vec4(world_position, 1.0)).xyz
            if (
                local.x < 0.0 or local.x > 1.0
                or local.y < 0.0 or local.y > 1.0
                or local.z < 0.0 or local.z > 1.0
            ):
                continue
            scalar = 0.0
            if volume_index == 0:
                scalar = volume_0.sample_level_with(linear_sampler, local, 0.0).x
            elif volume_index == 1:
                scalar = volume_1.sample_level_with(linear_sampler, local, 0.0).x
            elif volume_index == 2:
                scalar = volume_2.sample_level_with(linear_sampler, local, 0.0).x
            else:
                scalar = volume_3.sample_level_with(linear_sampler, local, 0.0).x
            transfer_count = osh.maximum(osh.u32(header.value_parameters.y), osh.u32(1))
            transfer_coordinate = osh.clamp(scalar, 0.0, 1.0) * osh.f32(transfer_count - osh.u32(1))
            transfer_index = osh.minimum(osh.u32(transfer_coordinate + 0.5), transfer_count - osh.u32(1))
            sample_value = transfers[
                osh.u32(header.value_parameters.x) + transfer_index
            ]
            reference_alpha = osh.clamp(
                sample_value.a * header.value_parameters.z, 0.0, 0.999999,
            )
            reference_step = osh.maximum(header.render_parameters.x, 1.0e-5)
            extinction = -osh.logarithm(1.0 - reference_alpha) / reference_step
            combined_extinction = combined_extinction + extinction
            combined_emission = combined_emission + (
                sample_value.rgb * header.value_parameters.w
            )
            if header.scattering_parameters.w > 0.0 and extinction > 0.0:
                incoming_radiance = osh.vec3(0.0)
                isotropic_radiance = osh.vec3(0.0)
                outgoing = -ray_direction
                light_count = osh.minimum(camera.volume_count.y, osh.u32(8))
                for light_index in range(8):
                    if osh.u32(light_index) >= light_count:
                        break
                    light = lights[osh.u32(light_index)]
                    light_type = osh.i32(light.position_type.w + 0.5)
                    incoming = osh.vec3(0.0)
                    attenuation = 1.0
                    enabled = light_type != 3
                    if light_type == 1:
                        incoming = -osh.normalize(light.direction_range.xyz)
                    else:
                        offset = light.position_type.xyz - world_position
                        distance_squared = osh.maximum(osh.dot(offset, offset), 1.0e-6)
                        distance_to_light = osh.sqrt(distance_squared)
                        incoming = offset / distance_to_light
                        attenuation = 1.0 / distance_squared
                        if light.direction_range.w > 0.0 and distance_to_light > light.direction_range.w:
                            enabled = False
                        if light_type == 2:
                            cone = osh.dot(osh.normalize(light.direction_range.xyz), -incoming)
                            spot_outer = osh.cosine(light.spot.y)
                            spot_inner = osh.cosine(light.spot.x)
                            spot = osh.clamp(
                                (cone - spot_outer)
                                / osh.maximum(spot_inner - spot_outer, 1.0e-6),
                                0.0, 1.0,
                            )
                            spot = spot * spot * (3.0 - 2.0 * spot)
                            attenuation = attenuation * spot
                            if spot <= 0.0:
                                enabled = False
                    if enabled:
                        incident = light.color_intensity.xyz * light.color_intensity.w * attenuation
                        cosine = osh.dot(-incoming, outgoing)
                        phase = 0.0795774715459
                        if header.phase_parameters.y > 0.5:
                            anisotropy = osh.clamp(header.phase_parameters.x, -0.95, 0.95)
                            denominator = osh.maximum(
                                1.0 + anisotropy * anisotropy - 2.0 * anisotropy * cosine,
                                1.0e-4,
                            )
                            phase = (1.0 - anisotropy * anisotropy) / (
                                12.5663706144 * denominator * osh.sqrt(denominator)
                            )
                        incoming_radiance = incoming_radiance + incident * phase
                        isotropic_radiance = isotropic_radiance + incident * 0.0795774715459
                scattered = incoming_radiance
                scattering_orders = osh.minimum(
                    osh.u32(header.multiple_scattering_parameters.w + 0.5),
                    osh.u32(8),
                )
                ratio = osh.clamp(
                    header.multiple_scattering_parameters.xyz
                    * (1.0 - osh.exp(
                        -extinction * (exit_distance - entry)
                    )),
                    osh.vec3(0.0), osh.vec3(0.999),
                )
                order_weight = ratio
                for order in range(2, 9):
                    if osh.u32(order) > scattering_orders:
                        break
                    scattered = scattered + isotropic_radiance * order_weight
                    order_weight = order_weight * ratio
                combined_emission = combined_emission + (
                    scattered * header.scattering_parameters.xyz
                    * header.scattering_parameters.w
                )
        opacity = 1.0 - osh.exp(-combined_extinction * step_size)
        radiance = radiance + transmittance * combined_emission * opacity
        transmittance = transmittance * (1.0 - opacity)
        distance = distance + step_size
    return osh.vec4(
        radiance + background.rgb * transmittance,
        background.a,
    )


__all__ = [
    "RasterVolumeCamera", "RasterVolumeHeader", "VolumeVertexOutput",
    "RasterVolumeLight", "volume_fragment", "volume_vertex",
]
