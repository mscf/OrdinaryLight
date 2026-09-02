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


@osh.structure
class RasterVolumeShadow:
    view_projection: osh.mat4
    atlas: osh.vec4
    parameters: osh.vec4


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
    occupancy_0: osh.sampled_texture_3d(binding=12),
    occupancy_1: osh.sampled_texture_3d(binding=13),
    occupancy_2: osh.sampled_texture_3d(binding=14),
    occupancy_3: osh.sampled_texture_3d(binding=15),
    shadow_map: osh.sampled_depth_texture_2d(binding=16),
    shadow_sampler: osh.comparison_sampler(binding=17),
    shadows: osh.storage_buffer(RasterVolumeShadow, access="read", binding=18),
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
        # Traverse the same conservative occupancy hierarchy as the GI path.
        # A jump ends at the first brick boundary (or the next volume entry),
        # so it cannot cross unseen density in an overlapping medium.
        inside_any = False
        occupied_any = False
        empty_exit = 1.0e30
        if camera.volume_count.z > osh.u32(0):
            for occupancy_index in range(4):
                if osh.u32(occupancy_index) >= volume_count:
                    break
                occupancy_header = headers[osh.u32(occupancy_index)]
                occupancy_local = (
                    occupancy_header.world_to_local
                    * osh.vec4(world_position, 1.0)
                ).xyz
                occupancy_direction = (
                    occupancy_header.world_to_local
                    * osh.vec4(ray_direction, 0.0)
                ).xyz
                # An empty brick in one medium must not jump across the entry
                # surface of another medium.  Clamp the candidate jump to the
                # nearest positive box entry for every currently-outside
                # volume before considering the containing brick.
                occupancy_safe_direction = occupancy_direction + (
                    osh.sign(occupancy_direction) * 1.0e-8
                )
                occupancy_first = -occupancy_local / occupancy_safe_direction
                occupancy_second = (
                    osh.vec3(1.0) - occupancy_local
                ) / occupancy_safe_direction
                occupancy_lower = osh.minimum(
                    occupancy_first, occupancy_second,
                )
                occupancy_upper = osh.maximum(
                    occupancy_first, occupancy_second,
                )
                occupancy_entry = osh.maximum(
                    occupancy_lower.x,
                    osh.maximum(occupancy_lower.y, occupancy_lower.z),
                )
                occupancy_box_exit = osh.minimum(
                    occupancy_upper.x,
                    osh.minimum(occupancy_upper.y, occupancy_upper.z),
                )
                if (
                    occupancy_local.x >= 0.0 and occupancy_local.x <= 1.0
                    and occupancy_local.y >= 0.0 and occupancy_local.y <= 1.0
                    and occupancy_local.z >= 0.0 and occupancy_local.z <= 1.0
                ):
                    inside_any = True
                    brick_grid = occupancy_header.acceleration_parameters.yzw
                    if brick_grid.x == osh.u32(0):
                        occupied_any = True
                        continue
                    brick_coordinate = osh.minimum(
                        osh.uvec3(occupancy_local * osh.vec3(brick_grid)),
                        brick_grid - osh.uvec3(1),
                    )
                    occupancy_uv = (
                        osh.vec3(brick_coordinate) + osh.vec3(0.5)
                    ) / osh.vec3(brick_grid)
                    occupied = 0.0
                    if occupancy_index == 0:
                        occupied = occupancy_0.sample_level_with(
                            depth_sampler, occupancy_uv, 0.0,
                        ).x
                    elif occupancy_index == 1:
                        occupied = occupancy_1.sample_level_with(
                            depth_sampler, occupancy_uv, 0.0,
                        ).x
                    elif occupancy_index == 2:
                        occupied = occupancy_2.sample_level_with(
                            depth_sampler, occupancy_uv, 0.0,
                        ).x
                    else:
                        occupied = occupancy_3.sample_level_with(
                            depth_sampler, occupancy_uv, 0.0,
                        ).x
                    if occupied > 0.5:
                        occupied_any = True
                    else:
                        lower_brick = osh.vec3(brick_coordinate) / osh.vec3(brick_grid)
                        upper_brick = (
                            osh.vec3(brick_coordinate) + osh.vec3(1.0)
                        ) / osh.vec3(brick_grid)
                        axis_exit = osh.vec3(1.0e30)
                        if osh.absolute(occupancy_direction.x) > 1.0e-10:
                            boundary = (
                                upper_brick.x if occupancy_direction.x > 0.0
                                else lower_brick.x
                            )
                            axis_exit.x = osh.maximum(
                                (boundary - occupancy_local.x)
                                / occupancy_direction.x, 0.0,
                            )
                        if osh.absolute(occupancy_direction.y) > 1.0e-10:
                            boundary = (
                                upper_brick.y if occupancy_direction.y > 0.0
                                else lower_brick.y
                            )
                            axis_exit.y = osh.maximum(
                                (boundary - occupancy_local.y)
                                / occupancy_direction.y, 0.0,
                            )
                        if osh.absolute(occupancy_direction.z) > 1.0e-10:
                            boundary = (
                                upper_brick.z if occupancy_direction.z > 0.0
                                else lower_brick.z
                            )
                            axis_exit.z = osh.maximum(
                                (boundary - occupancy_local.z)
                                / occupancy_direction.z, 0.0,
                            )
                        empty_exit = osh.minimum(
                            empty_exit,
                            distance + osh.minimum(
                                axis_exit.x,
                                osh.minimum(axis_exit.y, axis_exit.z),
                            ),
                        )
                elif occupancy_box_exit > osh.maximum(occupancy_entry, 0.0):
                    if occupancy_entry > 0.0:
                        empty_exit = osh.minimum(
                            empty_exit, distance + occupancy_entry,
                        )
            if inside_any and not occupied_any:
                distance = osh.maximum(
                    distance + step_size, empty_exit + step_size * 0.001,
                )
                continue
            if not inside_any and empty_exit < 1.0e29:
                distance = osh.maximum(
                    distance + step_size, empty_exit + step_size * 0.001,
                )
                continue
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
                    distance_to_light = 1.0e6
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
                        # Integrate extinction through every overlapping volume
                        # on the light ray. This makes volume-on-volume shadows
                        # additive instead of whichever-medium-was-last wins.
                        light_limit = (
                            1.0e6 if light_type == 1 else distance_to_light
                        )
                        light_optical_depth = 0.0
                        for shadow_volume_index in range(4):
                            if osh.u32(shadow_volume_index) >= volume_count:
                                break
                            shadow_header = headers[osh.u32(shadow_volume_index)]
                            shadow_origin = (
                                shadow_header.world_to_local
                                * osh.vec4(world_position, 1.0)
                            ).xyz
                            shadow_direction = (
                                shadow_header.world_to_local
                                * osh.vec4(incoming, 0.0)
                            ).xyz
                            safe_shadow_direction = shadow_direction + (
                                osh.sign(shadow_direction) * 1.0e-8
                            )
                            shadow_first = -shadow_origin / safe_shadow_direction
                            shadow_second = (
                                osh.vec3(1.0) - shadow_origin
                            ) / safe_shadow_direction
                            shadow_lower = osh.minimum(shadow_first, shadow_second)
                            shadow_upper = osh.maximum(shadow_first, shadow_second)
                            shadow_entry = osh.maximum(
                                0.002,
                                osh.maximum(
                                    shadow_lower.x,
                                    osh.maximum(shadow_lower.y, shadow_lower.z),
                                ),
                            )
                            shadow_exit = osh.minimum(
                                light_limit,
                                osh.minimum(
                                    shadow_upper.x,
                                    osh.minimum(shadow_upper.y, shadow_upper.z),
                                ),
                            )
                            if shadow_exit > shadow_entry:
                                shadow_midpoint = world_position + incoming * (
                                    0.5 * (shadow_entry + shadow_exit)
                                )
                                shadow_local = (
                                    shadow_header.world_to_local
                                    * osh.vec4(shadow_midpoint, 1.0)
                                ).xyz
                                shadow_occupied = 1.0
                                if camera.volume_count.z > osh.u32(0):
                                    shadow_brick_grid = (
                                        shadow_header.acceleration_parameters.yzw
                                    )
                                    if shadow_brick_grid.x > osh.u32(0):
                                        shadow_brick_coordinate = osh.minimum(
                                            osh.uvec3(
                                                shadow_local
                                                * osh.vec3(shadow_brick_grid)
                                            ),
                                            shadow_brick_grid - osh.uvec3(1),
                                        )
                                        shadow_occupancy_uv = (
                                            osh.vec3(shadow_brick_coordinate)
                                            + osh.vec3(0.5)
                                        ) / osh.vec3(shadow_brick_grid)
                                        if shadow_volume_index == 0:
                                            shadow_occupied = occupancy_0.sample_level_with(
                                                depth_sampler, shadow_occupancy_uv, 0.0,
                                            ).x
                                        elif shadow_volume_index == 1:
                                            shadow_occupied = occupancy_1.sample_level_with(
                                                depth_sampler, shadow_occupancy_uv, 0.0,
                                            ).x
                                        elif shadow_volume_index == 2:
                                            shadow_occupied = occupancy_2.sample_level_with(
                                                depth_sampler, shadow_occupancy_uv, 0.0,
                                            ).x
                                        else:
                                            shadow_occupied = occupancy_3.sample_level_with(
                                                depth_sampler, shadow_occupancy_uv, 0.0,
                                            ).x
                                shadow_scalar = 0.0
                                if shadow_volume_index == 0:
                                    shadow_scalar = volume_0.sample_level_with(
                                        linear_sampler, shadow_local, 0.0,
                                    ).x
                                elif shadow_volume_index == 1:
                                    shadow_scalar = volume_1.sample_level_with(
                                        linear_sampler, shadow_local, 0.0,
                                    ).x
                                elif shadow_volume_index == 2:
                                    shadow_scalar = volume_2.sample_level_with(
                                        linear_sampler, shadow_local, 0.0,
                                    ).x
                                else:
                                    shadow_scalar = volume_3.sample_level_with(
                                        linear_sampler, shadow_local, 0.0,
                                    ).x
                                shadow_transfer_count = osh.maximum(
                                    osh.u32(shadow_header.value_parameters.y),
                                    osh.u32(1),
                                )
                                shadow_transfer_coordinate = osh.clamp(
                                    shadow_scalar, 0.0, 1.0,
                                ) * osh.f32(shadow_transfer_count - osh.u32(1))
                                shadow_transfer_index = osh.minimum(
                                    osh.u32(shadow_transfer_coordinate + 0.5),
                                    shadow_transfer_count - osh.u32(1),
                                )
                                shadow_sample = transfers[
                                    osh.u32(shadow_header.value_parameters.x)
                                    + shadow_transfer_index
                                ]
                                shadow_alpha = osh.clamp(
                                    shadow_sample.a
                                    * shadow_header.value_parameters.z,
                                    0.0, 0.999999,
                                )
                                shadow_reference_step = osh.maximum(
                                    shadow_header.render_parameters.x, 1.0e-5,
                                )
                                shadow_extinction = -osh.logarithm(
                                    1.0 - shadow_alpha
                                ) / shadow_reference_step
                                light_optical_depth = light_optical_depth + (
                                    shadow_extinction
                                    * (shadow_exit - shadow_entry)
                                    * (1.0 if shadow_occupied > 0.5 else 0.0)
                                )
                        incident = incident * osh.exp(-light_optical_depth)

                        # Match the opaque raster path's native shadow atlas.
                        opaque_visibility = 1.0
                        shadow_count = osh.minimum(
                            camera.volume_count.w, osh.u32(24),
                        )
                        for shadow_index in range(24):
                            if osh.u32(shadow_index) >= shadow_count:
                                break
                            shadow_record = shadows[osh.u32(shadow_index)]
                            if osh.absolute(
                                shadow_record.parameters.x
                                - osh.f32(light_index)
                            ) > 0.25:
                                continue
                            shadow_face_matches = True
                            if shadow_record.parameters.w > 1.5:
                                point_shadow_delta = (
                                    world_position - light.position_type.xyz
                                )
                                point_shadow_axis = osh.absolute(
                                    point_shadow_delta
                                )
                                point_shadow_face = 0.0
                                if (
                                    point_shadow_axis.y >= point_shadow_axis.x
                                    and point_shadow_axis.y >= point_shadow_axis.z
                                ):
                                    point_shadow_face = (
                                        2.0 if point_shadow_delta.y >= 0.0
                                        else 3.0
                                    )
                                elif point_shadow_axis.z >= point_shadow_axis.x:
                                    point_shadow_face = (
                                        4.0 if point_shadow_delta.z >= 0.0
                                        else 5.0
                                    )
                                else:
                                    point_shadow_face = (
                                        0.0 if point_shadow_delta.x >= 0.0
                                        else 1.0
                                    )
                                shadow_face_matches = osh.absolute(
                                    shadow_record.parameters.y - point_shadow_face
                                ) < 0.25
                            if not shadow_face_matches:
                                continue
                            shadow_clip = shadow_record.view_projection * osh.vec4(
                                world_position + incoming * shadow_record.parameters.z,
                                1.0,
                            )
                            shadow_w = osh.maximum(
                                osh.absolute(shadow_clip.w), 1.0e-6,
                            )
                            shadow_ndc = shadow_clip.xyz / shadow_w
                            if (
                                shadow_clip.w > 0.0
                                and osh.absolute(shadow_ndc.x) <= 1.0001
                                and osh.absolute(shadow_ndc.y) <= 1.0001
                                and shadow_ndc.z >= 0.0 and shadow_ndc.z <= 1.0
                            ):
                                shadow_uv = osh.vec2(
                                    shadow_record.atlas.x
                                    + shadow_record.atlas.y * shadow_ndc.x,
                                    shadow_record.atlas.z
                                    - shadow_record.atlas.w * shadow_ndc.y,
                                )
                                opaque_visibility = shadow_map.sample_compare_with(
                                    shadow_sampler, shadow_uv,
                                    shadow_ndc.z - 0.00002,
                                )
                                break
                        incident = incident * opaque_visibility
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
    "RasterVolumeLight", "RasterVolumeShadow", "volume_fragment", "volume_vertex",
]
