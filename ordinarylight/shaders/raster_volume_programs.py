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


@osh.vertex
def volume_vertex(
    clip_position: osh.location(osh.vec2, 0),
) -> VolumeVertexOutput:
    return VolumeVertexOutput(
        osh.vec4(clip_position, 0.0, 1.0),
        clip_position * 0.5 + osh.vec2(0.5),
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
) -> osh.location(osh.vec4, 0):
    background = scene_color.sample_level_with(linear_sampler, uv, 0.0)
    opaque_depth = scene_depth.sample_depth_with(depth_sampler, uv)
    # Sampled textures use a top-left screen origin, while the camera matrix
    # follows OpenGL clip coordinates where +Y is the top of the image.
    # Keep texture UVs unchanged and flip only the Y coordinate used for ray
    # reconstruction; conflating the two vertically displaced the volume
    # relative to otherwise correctly oriented raster geometry.
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
            extinction = sample_value.a * header.value_parameters.z
            combined_extinction = combined_extinction + extinction
            combined_emission = combined_emission + sample_value.rgb * (
                extinction + header.value_parameters.w
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
    "volume_fragment", "volume_vertex",
]
