struct RasterVolumeCamera {
    inverse_view_projection: mat4x4<f32>,
    camera_position: vec4<f32>,
    viewport_steps: vec4<f32>,
    volume_count: vec4<u32>,
}

struct RasterVolumeHeader {
    world_to_local: mat4x4<f32>,
    dimensions_offset: vec4<u32>,
    value_parameters: vec4<f32>,
    render_parameters: vec4<f32>,
    scattering_parameters: vec4<f32>,
    phase_parameters: vec4<f32>,
    multiple_scattering_parameters: vec4<f32>,
    acceleration_parameters: vec4<u32>,
}

@group(0) @binding(0) var<uniform> camera: RasterVolumeCamera;
@group(0) @binding(1) var<storage, read> headers: array<RasterVolumeHeader>;
@group(0) @binding(2) var<storage, read> transfers: array<vec4<f32>>;
@group(0) @binding(3) var scene_color: texture_2d<f32>;
@group(0) @binding(4) var scene_depth: texture_depth_2d;
@group(0) @binding(5) var volume_0: texture_3d<f32>;
@group(0) @binding(6) var volume_1: texture_3d<f32>;
@group(0) @binding(7) var volume_2: texture_3d<f32>;
@group(0) @binding(8) var volume_3: texture_3d<f32>;
@group(0) @binding(9) var linear_sampler: sampler;
@group(0) @binding(10) var depth_sampler: sampler;

@fragment
fn main(
    @location(0) uv: vec2<f32>
) -> @location(0) vec4<f32> {
    let background: vec4<f32> = textureSampleLevel(scene_color, linear_sampler, uv, 0.0);
    let opaque_depth: f32 = textureSample(scene_depth, depth_sampler, uv);
    let clip_xy: vec2<f32> = vec2<f32>(((uv.x * 2.0) - 1.0), (1.0 - (uv.y * 2.0)));
    let near_clip: vec4<f32> = vec4<f32>(clip_xy, 0.0, 1.0);
    let far_clip: vec4<f32> = vec4<f32>(clip_xy, 1.0, 1.0);
    let near_world_h: vec4<f32> = (camera.inverse_view_projection * near_clip);
    let far_world_h: vec4<f32> = (camera.inverse_view_projection * far_clip);
    let near_world: vec3<f32> = (near_world_h.xyz / max(abs(near_world_h.w), 1e-06));
    let far_world: vec3<f32> = (far_world_h.xyz / max(abs(far_world_h.w), 1e-06));
    let ray_origin: vec3<f32> = camera.camera_position.xyz;
    let ray_direction: vec3<f32> = normalize((far_world - near_world));
    let opaque_world_h: vec4<f32> = (camera.inverse_view_projection * vec4<f32>(clip_xy, opaque_depth, 1.0));
    let opaque_world: vec3<f32> = (opaque_world_h.xyz / max(abs(opaque_world_h.w), 1e-06));
    var ray_limit: f32 = length((opaque_world - ray_origin));
    if ((opaque_depth >= 0.999999)) {
        ray_limit = length((far_world - ray_origin));
    }
    var entry: f32 = ray_limit;
    var exit_distance: f32 = 0.0;
    var step_size: f32 = 1e+30;
    let volume_count: u32 = min(camera.volume_count.x, u32(4));
    for (var volume_index: i32 = 0; volume_index < 4; volume_index += 1) {
        if ((u32(volume_index) >= volume_count)) {
            break;
        }
        let header: RasterVolumeHeader = headers[u32(volume_index)];
        let local_origin: vec3<f32> = (header.world_to_local * vec4<f32>(ray_origin, 1.0)).xyz;
        let local_direction: vec3<f32> = (header.world_to_local * vec4<f32>(ray_direction, 0.0)).xyz;
        let inverse_direction: vec3<f32> = (vec3<f32>(1.0) / (local_direction + (sign(local_direction) * 1e-08)));
        let first: vec3<f32> = ((vec3<f32>(0.0) - local_origin) * inverse_direction);
        let second: vec3<f32> = ((vec3<f32>(1.0) - local_origin) * inverse_direction);
        let lower: vec3<f32> = min(first, second);
        let upper: vec3<f32> = max(first, second);
        let volume_entry: f32 = max(max(lower.x, lower.y), lower.z);
        let volume_exit: f32 = min(min(upper.x, upper.y), upper.z);
        if ((volume_exit > max(volume_entry, 0.0))) {
            entry = min(entry, max(volume_entry, 0.0));
            exit_distance = max(exit_distance, min(volume_exit, ray_limit));
            step_size = min(step_size, header.render_parameters.x);
        }
    }
    if ((exit_distance <= entry)) {
        return background;
    }
    step_size = max((step_size * camera.viewport_steps.z), 1e-05);
    var transmittance: f32 = 1.0;
    var radiance: vec3<f32> = vec3<f32>(0.0);
    var distance: f32 = (entry + (step_size * 0.5));
    let max_steps: u32 = min(u32(camera.viewport_steps.w), u32(8192));
    for (var step: i32 = 0; step < 8192; step += 1) {
        if ((((u32(step) >= max_steps) || (distance >= exit_distance)) || (transmittance <= 0.001))) {
            break;
        }
        var combined_extinction: f32 = 0.0;
        var combined_emission: vec3<f32> = vec3<f32>(0.0);
        let world_position: vec3<f32> = (ray_origin + (ray_direction * distance));
        for (var volume_index: i32 = 0; volume_index < 4; volume_index += 1) {
            if ((u32(volume_index) >= volume_count)) {
                break;
            }
            let header: RasterVolumeHeader = headers[u32(volume_index)];
            let local: vec3<f32> = (header.world_to_local * vec4<f32>(world_position, 1.0)).xyz;
            if (((((((local.x < 0.0) || (local.x > 1.0)) || (local.y < 0.0)) || (local.y > 1.0)) || (local.z < 0.0)) || (local.z > 1.0))) {
                continue;
            }
            var scalar: f32 = 0.0;
            if ((volume_index == 0)) {
                scalar = textureSampleLevel(volume_0, linear_sampler, local, 0.0).x;
            } else {
                if ((volume_index == 1)) {
                    scalar = textureSampleLevel(volume_1, linear_sampler, local, 0.0).x;
                } else {
                    if ((volume_index == 2)) {
                        scalar = textureSampleLevel(volume_2, linear_sampler, local, 0.0).x;
                    } else {
                        scalar = textureSampleLevel(volume_3, linear_sampler, local, 0.0).x;
                    }
                }
            }
            let transfer_count: u32 = max(u32(header.value_parameters.y), u32(1));
            let transfer_coordinate: f32 = (clamp(scalar, 0.0, 1.0) * f32((transfer_count - u32(1))));
            let transfer_index: u32 = min(u32((transfer_coordinate + 0.5)), (transfer_count - u32(1)));
            let sample_value: vec4<f32> = transfers[(u32(header.value_parameters.x) + transfer_index)];
            let extinction: f32 = (sample_value.a * header.value_parameters.z);
            combined_extinction = (combined_extinction + extinction);
            combined_emission = (combined_emission + (sample_value.rgb * (extinction + header.value_parameters.w)));
        }
        let opacity: f32 = (1.0 - exp(((-combined_extinction) * step_size)));
        radiance = (radiance + ((transmittance * combined_emission) * opacity));
        transmittance = (transmittance * (1.0 - opacity));
        distance = (distance + step_size);
    }
    return vec4<f32>((radiance + (background.rgb * transmittance)), background.a);
}
