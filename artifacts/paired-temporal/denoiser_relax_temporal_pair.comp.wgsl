struct TemporalConstants {
    extent_history: vec4<f32>,
    rejection: vec4<f32>,
}

@group(0) @binding(0) var current_radiance_hit_distance: texture_storage_2d<rgba16float, read>;
@group(0) @binding(1) var normal_roughness: texture_storage_2d<rgba16float, read>;
@group(0) @binding(2) var view_z: texture_storage_2d<r32float, read>;
@group(0) @binding(3) var motion: texture_storage_2d<rgba16float, read>;
@group(0) @binding(4) var material_id: texture_storage_2d<r32uint, read>;
@group(0) @binding(5) var previous_radiance: texture_storage_2d<rgba16float, read>;
@group(0) @binding(6) var previous_normal_roughness: texture_storage_2d<rgba16float, read>;
@group(0) @binding(7) var previous_view_z: texture_storage_2d<r32float, read>;
@group(0) @binding(8) var previous_material_id: texture_storage_2d<r32uint, read>;
@group(0) @binding(9) var previous_history_length: texture_storage_2d<r32float, read>;
@group(0) @binding(10) var output_radiance: texture_storage_2d<rgba16float, write>;
@group(0) @binding(11) var output_history_length: texture_storage_2d<r32float, write>;
@group(0) @binding(12) var<uniform> constants: TemporalConstants;
@group(0) @binding(13) var identity: texture_storage_2d<r32uint, read>;
@group(0) @binding(14) var previous_identity: texture_storage_2d<r32uint, read>;
@group(0) @binding(15) var current_specular: texture_storage_2d<rgba16float, read>;
@group(0) @binding(16) var previous_specular: texture_storage_2d<rgba16float, read>;
@group(0) @binding(17) var previous_specular_length: texture_storage_2d<r32float, read>;
@group(0) @binding(18) var output_specular: texture_storage_2d<rgba16float, write>;
@group(0) @binding(19) var output_specular_length: texture_storage_2d<r32float, write>;

@compute @workgroup_size(8, 8, 1)
fn main(
    @builtin(global_invocation_id) global_invocation_id: vec3<u32>,
    @builtin(local_invocation_id) local_invocation_id: vec3<u32>,
    @builtin(local_invocation_index) local_invocation_index: u32,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>,
) {
    let pixel: vec2<i32> = vec2<i32>(global_invocation_id.xy);
    let extent: vec2<i32> = vec2<i32>(constants.extent_history.xy);
    if (((pixel.x >= extent.x) || (pixel.y >= extent.y))) {
        return;
    }
    var current: vec4<f32> = textureLoad(current_radiance_hit_distance, pixel);
    var specular_current: vec4<f32> = textureLoad(current_specular, pixel);
    if ((!(constants.extent_history.w > 0.5))) {
        textureStore(output_radiance, pixel, current);
        textureStore(output_history_length, pixel, vec4<f32>(1.0));
        textureStore(output_specular, pixel, specular_current);
        textureStore(output_specular_length, pixel, vec4<f32>(1.0));
        return;
    }
    let current_normal: vec3<f32> = textureLoad(normal_roughness, pixel).xyz;
    let current_depth: f32 = textureLoad(view_z, pixel).r;
    let motion_sample: vec4<f32> = textureLoad(motion, pixel);
    let motion_vector: vec2<f32> = motion_sample.xy;
    let expected_old_depth: f32 = motion_sample.z;
    let previous_pixel: vec2<i32> = vec2<i32>(((vec2<f32>(pixel) + motion_vector) + vec2<f32>(0.5)));
    let in_bounds: bool = ((((previous_pixel.x >= 0) && (previous_pixel.y >= 0)) && (previous_pixel.x < extent.x)) && (previous_pixel.y < extent.y));
    var accepted: bool = ((((constants.extent_history.w > 0.5) && in_bounds) && (current_depth != 0.0)) && (expected_old_depth > 0.0));
    var history: vec4<f32> = current;
    var history_length: f32 = 1.0;
    var specular_history: vec4<f32> = specular_current;
    var specular_history_length: f32 = 1.0;
    var specular_accepted: bool = accepted;
    if (accepted) {
        let old_depth: f32 = textureLoad(previous_view_z, previous_pixel).r;
        let old_normal: vec3<f32> = textureLoad(previous_normal_roughness, previous_pixel).xyz;
        let old_material: u32 = textureLoad(previous_material_id, previous_pixel).r;
        let old_primitive: u32 = textureLoad(previous_identity, previous_pixel).r;
        let current_primitive: u32 = textureLoad(identity, pixel).r;
        let current_material: u32 = textureLoad(material_id, pixel).r;
        let depth_tolerance: f32 = max((abs(expected_old_depth) * constants.rejection.y), 0.001);
        accepted = (((((old_depth != 0.0) && (dot(current_normal, old_normal) >= constants.rejection.x)) && (abs((expected_old_depth - old_depth)) <= depth_tolerance)) && (old_material == current_material)) && (old_primitive == current_primitive));
        specular_accepted = accepted;
        if (accepted) {
            history = textureLoad(previous_radiance, previous_pixel);
            specular_history = textureLoad(previous_specular, previous_pixel);
            var neighborhood_sum: vec3<f32> = vec3<f32>(0.0);
            var neighborhood_square_sum: vec3<f32> = vec3<f32>(0.0);
            var neighborhood_count: f32 = 0.0;
            var specular_sum: vec3<f32> = vec3<f32>(0.0);
            var specular_square_sum: vec3<f32> = vec3<f32>(0.0);
            for (var y: i32 = (-1); y < 2; y += 1) {
                for (var x: i32 = (-1); x < 2; x += 1) {
                    let neighbor_pixel: vec2<i32> = (pixel + vec2<i32>(x, y));
                    if (((neighbor_pixel.x < 0) || (neighbor_pixel.y < 0))) {
                        continue;
                    }
                    if (((neighbor_pixel.x >= extent.x) || (neighbor_pixel.y >= extent.y))) {
                        continue;
                    }
                    if ((constants.rejection.w > 0.0)) {
                        if ((textureLoad(material_id, neighbor_pixel).r != current_material)) {
                            continue;
                        }
                        if ((textureLoad(identity, neighbor_pixel).r != current_primitive)) {
                            continue;
                        }
                        let neighbor_normal: vec3<f32> = textureLoad(normal_roughness, neighbor_pixel).xyz;
                        if ((dot(current_normal, neighbor_normal) < constants.rejection.x)) {
                            continue;
                        }
                    }
                    let neighbor: vec3<f32> = textureLoad(current_radiance_hit_distance, neighbor_pixel).rgb;
                    neighborhood_sum = (neighborhood_sum + neighbor);
                    neighborhood_square_sum = (neighborhood_square_sum + (neighbor * neighbor));
                    let specular_neighbor: vec3<f32> = textureLoad(current_specular, neighbor_pixel).rgb;
                    specular_sum = (specular_sum + specular_neighbor);
                    specular_square_sum = (specular_square_sum + (specular_neighbor * specular_neighbor));
                    neighborhood_count = (neighborhood_count + 1.0);
                }
            }
            let neighborhood_mean: vec3<f32> = (neighborhood_sum / max(neighborhood_count, 1.0));
            let neighborhood_variance: vec3<f32> = max(((neighborhood_square_sum / max(neighborhood_count, 1.0)) - (neighborhood_mean * neighborhood_mean)), vec3<f32>(0.0));
            let neighborhood_deviation: vec3<f32> = sqrt(neighborhood_variance);
            let clamp_radius: vec3<f32> = (neighborhood_deviation * constants.rejection.z);
            if ((constants.rejection.w > 0.0)) {
                let history_luma: f32 = dot(history.rgb, vec3<f32>(0.2126, 0.7152, 0.0722));
                let mean_luma: f32 = dot(neighborhood_mean, vec3<f32>(0.2126, 0.7152, 0.0722));
                let deviation_luma: f32 = dot(neighborhood_deviation, vec3<f32>(0.2126, 0.7152, 0.0722));
                let reactive_limit: f32 = max((deviation_luma * constants.rejection.w), ((abs(mean_luma) * 0.1) + 0.01));
                accepted = (abs((history_luma - mean_luma)) <= reactive_limit);
            }
            history = vec4<f32>(clamp(history.rgb, (neighborhood_mean - clamp_radius), (neighborhood_mean + clamp_radius)), history.a);
            if (accepted) {
                history_length = min((textureLoad(previous_history_length, previous_pixel).r + 1.0), constants.extent_history.z);
            }
            let specular_mean: vec3<f32> = (specular_sum / max(neighborhood_count, 1.0));
            let specular_variance: vec3<f32> = max(((specular_square_sum / max(neighborhood_count, 1.0)) - (specular_mean * specular_mean)), vec3<f32>(0.0));
            let specular_deviation: vec3<f32> = sqrt(specular_variance);
            let specular_clamp_radius: vec3<f32> = (specular_deviation * constants.rejection.z);
            if ((constants.rejection.w > 0.0)) {
                let specular_history_luma: f32 = dot(specular_history.rgb, vec3<f32>(0.2126, 0.7152, 0.0722));
                let specular_mean_luma: f32 = dot(specular_mean, vec3<f32>(0.2126, 0.7152, 0.0722));
                let specular_deviation_luma: f32 = dot(specular_deviation, vec3<f32>(0.2126, 0.7152, 0.0722));
                let specular_reactive_limit: f32 = max((specular_deviation_luma * constants.rejection.w), ((abs(specular_mean_luma) * 0.1) + 0.01));
                specular_accepted = (abs((specular_history_luma - specular_mean_luma)) <= specular_reactive_limit);
            }
            specular_history = vec4<f32>(clamp(specular_history.rgb, (specular_mean - specular_clamp_radius), (specular_mean + specular_clamp_radius)), specular_history.a);
            if (specular_accepted) {
                specular_history_length = min((textureLoad(previous_specular_length, previous_pixel).r + 1.0), constants.extent_history.z);
            }
        }
    }
    if (((motion_sample.w > 0.5) && (length(motion_vector) > 1.0))) {
        history_length = min(history_length, 4.0);
    }
    if (accepted) {
        let alpha: f32 = (1.0 / max(history_length, 1.0));
        current = vec4<f32>(mix(history.rgb, current.rgb, alpha), current.a);
    }
    textureStore(output_radiance, pixel, current);
    textureStore(output_history_length, pixel, vec4<f32>(history_length));
    if (((motion_sample.w > 0.5) && (length(motion_vector) > 1.0))) {
        specular_history_length = min(specular_history_length, 4.0);
    }
    if (specular_accepted) {
        let specular_alpha: f32 = (1.0 / max(specular_history_length, 1.0));
        specular_current = vec4<f32>(mix(specular_history.rgb, specular_current.rgb, specular_alpha), specular_current.a);
    }
    textureStore(output_specular, pixel, specular_current);
    textureStore(output_specular_length, pixel, vec4<f32>(specular_history_length));
}
