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

struct RasterVolumeLight {
    position_type: vec4<f32>,
    direction_range: vec4<f32>,
    color_intensity: vec4<f32>,
    spot: vec4<f32>,
}

struct RasterVolumeShadow {
    view_projection: mat4x4<f32>,
    atlas: vec4<f32>,
    parameters: vec4<f32>,
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
@group(0) @binding(11) var<storage, read> lights: array<RasterVolumeLight>;
@group(0) @binding(12) var occupancy_0: texture_3d<f32>;
@group(0) @binding(13) var occupancy_1: texture_3d<f32>;
@group(0) @binding(14) var occupancy_2: texture_3d<f32>;
@group(0) @binding(15) var occupancy_3: texture_3d<f32>;
@group(0) @binding(16) var shadow_map: texture_depth_2d;
@group(0) @binding(17) var shadow_sampler: sampler_comparison;
@group(0) @binding(18) var<storage, read> shadows: array<RasterVolumeShadow>;

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
        var inside_any: bool = false;
        var occupied_any: bool = false;
        var empty_exit: f32 = 1e+30;
        if ((camera.volume_count.z > u32(0))) {
            for (var occupancy_index: i32 = 0; occupancy_index < 4; occupancy_index += 1) {
                if ((u32(occupancy_index) >= volume_count)) {
                    break;
                }
                let occupancy_header: RasterVolumeHeader = headers[u32(occupancy_index)];
                let occupancy_local: vec3<f32> = (occupancy_header.world_to_local * vec4<f32>(world_position, 1.0)).xyz;
                let occupancy_direction: vec3<f32> = (occupancy_header.world_to_local * vec4<f32>(ray_direction, 0.0)).xyz;
                let occupancy_safe_direction: vec3<f32> = (occupancy_direction + (sign(occupancy_direction) * 1e-08));
                let occupancy_first: vec3<f32> = ((-occupancy_local) / occupancy_safe_direction);
                let occupancy_second: vec3<f32> = ((vec3<f32>(1.0) - occupancy_local) / occupancy_safe_direction);
                let occupancy_lower: vec3<f32> = min(occupancy_first, occupancy_second);
                let occupancy_upper: vec3<f32> = max(occupancy_first, occupancy_second);
                let occupancy_entry: f32 = max(occupancy_lower.x, max(occupancy_lower.y, occupancy_lower.z));
                let occupancy_box_exit: f32 = min(occupancy_upper.x, min(occupancy_upper.y, occupancy_upper.z));
                if (((((((occupancy_local.x >= 0.0) && (occupancy_local.x <= 1.0)) && (occupancy_local.y >= 0.0)) && (occupancy_local.y <= 1.0)) && (occupancy_local.z >= 0.0)) && (occupancy_local.z <= 1.0))) {
                    inside_any = true;
                    let brick_grid: vec3<u32> = occupancy_header.acceleration_parameters.yzw;
                    if ((brick_grid.x == u32(0))) {
                        occupied_any = true;
                        continue;
                    }
                    let brick_coordinate: vec3<u32> = min(vec3<u32>((occupancy_local * vec3<f32>(brick_grid))), (brick_grid - vec3<u32>(1)));
                    let occupancy_uv: vec3<f32> = ((vec3<f32>(brick_coordinate) + vec3<f32>(0.5)) / vec3<f32>(brick_grid));
                    var occupied: f32 = 0.0;
                    if ((occupancy_index == 0)) {
                        occupied = textureSampleLevel(occupancy_0, depth_sampler, occupancy_uv, 0.0).x;
                    } else {
                        if ((occupancy_index == 1)) {
                            occupied = textureSampleLevel(occupancy_1, depth_sampler, occupancy_uv, 0.0).x;
                        } else {
                            if ((occupancy_index == 2)) {
                                occupied = textureSampleLevel(occupancy_2, depth_sampler, occupancy_uv, 0.0).x;
                            } else {
                                occupied = textureSampleLevel(occupancy_3, depth_sampler, occupancy_uv, 0.0).x;
                            }
                        }
                    }
                    if ((occupied > 0.5)) {
                        occupied_any = true;
                    } else {
                        let lower_brick: vec3<f32> = (vec3<f32>(brick_coordinate) / vec3<f32>(brick_grid));
                        let upper_brick: vec3<f32> = ((vec3<f32>(brick_coordinate) + vec3<f32>(1.0)) / vec3<f32>(brick_grid));
                        var axis_exit: vec3<f32> = vec3<f32>(1e+30);
                        if ((abs(occupancy_direction.x) > 1e-10)) {
                            let boundary: f32 = select(lower_brick.x, upper_brick.x, (occupancy_direction.x > 0.0));
                            axis_exit.x = max(((boundary - occupancy_local.x) / occupancy_direction.x), 0.0);
                        }
                        if ((abs(occupancy_direction.y) > 1e-10)) {
                            let boundary: f32 = select(lower_brick.y, upper_brick.y, (occupancy_direction.y > 0.0));
                            axis_exit.y = max(((boundary - occupancy_local.y) / occupancy_direction.y), 0.0);
                        }
                        if ((abs(occupancy_direction.z) > 1e-10)) {
                            let boundary: f32 = select(lower_brick.z, upper_brick.z, (occupancy_direction.z > 0.0));
                            axis_exit.z = max(((boundary - occupancy_local.z) / occupancy_direction.z), 0.0);
                        }
                        empty_exit = min(empty_exit, (distance + min(axis_exit.x, min(axis_exit.y, axis_exit.z))));
                    }
                } else {
                    if ((occupancy_box_exit > max(occupancy_entry, 0.0))) {
                        if ((occupancy_entry > 0.0)) {
                            empty_exit = min(empty_exit, (distance + occupancy_entry));
                        }
                    }
                }
            }
            if ((inside_any && (!occupied_any))) {
                distance = max((distance + step_size), (empty_exit + (step_size * 0.001)));
                continue;
            }
            if (((!inside_any) && (empty_exit < 1e+29))) {
                distance = max((distance + step_size), (empty_exit + (step_size * 0.001)));
                continue;
            }
        }
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
            let reference_alpha: f32 = clamp((sample_value.a * header.value_parameters.z), 0.0, 0.999999);
            let reference_step: f32 = max(header.render_parameters.x, 1e-05);
            let extinction: f32 = ((-log((1.0 - reference_alpha))) / reference_step);
            combined_extinction = (combined_extinction + extinction);
            combined_emission = (combined_emission + (sample_value.rgb * header.value_parameters.w));
            if (((header.scattering_parameters.w > 0.0) && (extinction > 0.0))) {
                var incoming_radiance: vec3<f32> = vec3<f32>(0.0);
                var isotropic_radiance: vec3<f32> = vec3<f32>(0.0);
                let outgoing: vec3<f32> = (-ray_direction);
                let light_count: u32 = min(camera.volume_count.y, u32(8));
                for (var light_index: i32 = 0; light_index < 8; light_index += 1) {
                    if ((u32(light_index) >= light_count)) {
                        break;
                    }
                    let light: RasterVolumeLight = lights[u32(light_index)];
                    let light_type: i32 = i32((light.position_type.w + 0.5));
                    var incoming: vec3<f32> = vec3<f32>(0.0);
                    var attenuation: f32 = 1.0;
                    var distance_to_light: f32 = 1000000.0;
                    var enabled: bool = (light_type != 3);
                    if ((light_type == 1)) {
                        incoming = (-normalize(light.direction_range.xyz));
                    } else {
                        let offset: vec3<f32> = (light.position_type.xyz - world_position);
                        let distance_squared: f32 = max(dot(offset, offset), 1e-06);
                        distance_to_light = sqrt(distance_squared);
                        incoming = (offset / distance_to_light);
                        attenuation = (1.0 / distance_squared);
                        if (((light.direction_range.w > 0.0) && (distance_to_light > light.direction_range.w))) {
                            enabled = false;
                        }
                        if ((light_type == 2)) {
                            let cone: f32 = dot(normalize(light.direction_range.xyz), (-incoming));
                            let spot_outer: f32 = cos(light.spot.y);
                            let spot_inner: f32 = cos(light.spot.x);
                            var spot: f32 = clamp(((cone - spot_outer) / max((spot_inner - spot_outer), 1e-06)), 0.0, 1.0);
                            spot = ((spot * spot) * (3.0 - (2.0 * spot)));
                            attenuation = (attenuation * spot);
                            if ((spot <= 0.0)) {
                                enabled = false;
                            }
                        }
                    }
                    if (enabled) {
                        var incident: vec3<f32> = ((light.color_intensity.xyz * light.color_intensity.w) * attenuation);
                        let light_limit: f32 = select(distance_to_light, 1000000.0, (light_type == 1));
                        var light_optical_depth: f32 = 0.0;
                        for (var shadow_volume_index: i32 = 0; shadow_volume_index < 4; shadow_volume_index += 1) {
                            if ((u32(shadow_volume_index) >= volume_count)) {
                                break;
                            }
                            let shadow_header: RasterVolumeHeader = headers[u32(shadow_volume_index)];
                            let shadow_origin: vec3<f32> = (shadow_header.world_to_local * vec4<f32>(world_position, 1.0)).xyz;
                            let shadow_direction: vec3<f32> = (shadow_header.world_to_local * vec4<f32>(incoming, 0.0)).xyz;
                            let safe_shadow_direction: vec3<f32> = (shadow_direction + (sign(shadow_direction) * 1e-08));
                            let shadow_first: vec3<f32> = ((-shadow_origin) / safe_shadow_direction);
                            let shadow_second: vec3<f32> = ((vec3<f32>(1.0) - shadow_origin) / safe_shadow_direction);
                            let shadow_lower: vec3<f32> = min(shadow_first, shadow_second);
                            let shadow_upper: vec3<f32> = max(shadow_first, shadow_second);
                            let shadow_entry: f32 = max(0.002, max(shadow_lower.x, max(shadow_lower.y, shadow_lower.z)));
                            let shadow_exit: f32 = min(light_limit, min(shadow_upper.x, min(shadow_upper.y, shadow_upper.z)));
                            if ((shadow_exit > shadow_entry)) {
                                let shadow_midpoint: vec3<f32> = (world_position + (incoming * (0.5 * (shadow_entry + shadow_exit))));
                                let shadow_local: vec3<f32> = (shadow_header.world_to_local * vec4<f32>(shadow_midpoint, 1.0)).xyz;
                                var shadow_occupied: f32 = 1.0;
                                if ((camera.volume_count.z > u32(0))) {
                                    let shadow_brick_grid: vec3<u32> = shadow_header.acceleration_parameters.yzw;
                                    if ((shadow_brick_grid.x > u32(0))) {
                                        let shadow_brick_coordinate: vec3<u32> = min(vec3<u32>((shadow_local * vec3<f32>(shadow_brick_grid))), (shadow_brick_grid - vec3<u32>(1)));
                                        let shadow_occupancy_uv: vec3<f32> = ((vec3<f32>(shadow_brick_coordinate) + vec3<f32>(0.5)) / vec3<f32>(shadow_brick_grid));
                                        if ((shadow_volume_index == 0)) {
                                            shadow_occupied = textureSampleLevel(occupancy_0, depth_sampler, shadow_occupancy_uv, 0.0).x;
                                        } else {
                                            if ((shadow_volume_index == 1)) {
                                                shadow_occupied = textureSampleLevel(occupancy_1, depth_sampler, shadow_occupancy_uv, 0.0).x;
                                            } else {
                                                if ((shadow_volume_index == 2)) {
                                                    shadow_occupied = textureSampleLevel(occupancy_2, depth_sampler, shadow_occupancy_uv, 0.0).x;
                                                } else {
                                                    shadow_occupied = textureSampleLevel(occupancy_3, depth_sampler, shadow_occupancy_uv, 0.0).x;
                                                }
                                            }
                                        }
                                    }
                                }
                                var shadow_scalar: f32 = 0.0;
                                if ((shadow_volume_index == 0)) {
                                    shadow_scalar = textureSampleLevel(volume_0, linear_sampler, shadow_local, 0.0).x;
                                } else {
                                    if ((shadow_volume_index == 1)) {
                                        shadow_scalar = textureSampleLevel(volume_1, linear_sampler, shadow_local, 0.0).x;
                                    } else {
                                        if ((shadow_volume_index == 2)) {
                                            shadow_scalar = textureSampleLevel(volume_2, linear_sampler, shadow_local, 0.0).x;
                                        } else {
                                            shadow_scalar = textureSampleLevel(volume_3, linear_sampler, shadow_local, 0.0).x;
                                        }
                                    }
                                }
                                let shadow_transfer_count: u32 = max(u32(shadow_header.value_parameters.y), u32(1));
                                let shadow_transfer_coordinate: f32 = (clamp(shadow_scalar, 0.0, 1.0) * f32((shadow_transfer_count - u32(1))));
                                let shadow_transfer_index: u32 = min(u32((shadow_transfer_coordinate + 0.5)), (shadow_transfer_count - u32(1)));
                                let shadow_sample: vec4<f32> = transfers[(u32(shadow_header.value_parameters.x) + shadow_transfer_index)];
                                let shadow_alpha: f32 = clamp((shadow_sample.a * shadow_header.value_parameters.z), 0.0, 0.999999);
                                let shadow_reference_step: f32 = max(shadow_header.render_parameters.x, 1e-05);
                                let shadow_extinction: f32 = ((-log((1.0 - shadow_alpha))) / shadow_reference_step);
                                light_optical_depth = (light_optical_depth + ((shadow_extinction * (shadow_exit - shadow_entry)) * select(0.0, 1.0, (shadow_occupied > 0.5))));
                            }
                        }
                        incident = (incident * exp((-light_optical_depth)));
                        var opaque_visibility: f32 = 1.0;
                        let shadow_count: u32 = min(camera.volume_count.w, u32(24));
                        for (var shadow_index: i32 = 0; shadow_index < 24; shadow_index += 1) {
                            if ((u32(shadow_index) >= shadow_count)) {
                                break;
                            }
                            let shadow_record: RasterVolumeShadow = shadows[u32(shadow_index)];
                            if ((abs((shadow_record.parameters.x - f32(light_index))) > 0.25)) {
                                continue;
                            }
                            var shadow_face_matches: bool = true;
                            if ((shadow_record.parameters.w > 1.5)) {
                                let point_shadow_delta: vec3<f32> = (world_position - light.position_type.xyz);
                                let point_shadow_axis: vec3<f32> = abs(point_shadow_delta);
                                var point_shadow_face: f32 = 0.0;
                                if (((point_shadow_axis.y >= point_shadow_axis.x) && (point_shadow_axis.y >= point_shadow_axis.z))) {
                                    point_shadow_face = select(3.0, 2.0, (point_shadow_delta.y >= 0.0));
                                } else {
                                    if ((point_shadow_axis.z >= point_shadow_axis.x)) {
                                        point_shadow_face = select(5.0, 4.0, (point_shadow_delta.z >= 0.0));
                                    } else {
                                        point_shadow_face = select(1.0, 0.0, (point_shadow_delta.x >= 0.0));
                                    }
                                }
                                shadow_face_matches = (abs((shadow_record.parameters.y - point_shadow_face)) < 0.25);
                            }
                            if ((!shadow_face_matches)) {
                                continue;
                            }
                            let shadow_clip: vec4<f32> = (shadow_record.view_projection * vec4<f32>((world_position + (incoming * shadow_record.parameters.z)), 1.0));
                            let shadow_w: f32 = max(abs(shadow_clip.w), 1e-06);
                            let shadow_ndc: vec3<f32> = (shadow_clip.xyz / shadow_w);
                            if ((((((shadow_clip.w > 0.0) && (abs(shadow_ndc.x) <= 1.0001)) && (abs(shadow_ndc.y) <= 1.0001)) && (shadow_ndc.z >= 0.0)) && (shadow_ndc.z <= 1.0))) {
                                let shadow_uv: vec2<f32> = vec2<f32>((shadow_record.atlas.x + (shadow_record.atlas.y * shadow_ndc.x)), (shadow_record.atlas.z - (shadow_record.atlas.w * shadow_ndc.y)));
                                opaque_visibility = textureSampleCompare(shadow_map, shadow_sampler, shadow_uv, (shadow_ndc.z - 2e-05));
                                break;
                            }
                        }
                        incident = (incident * opaque_visibility);
                        let cosine: f32 = dot((-incoming), outgoing);
                        var phase: f32 = 0.0795774715459;
                        if ((header.phase_parameters.y > 0.5)) {
                            let anisotropy: f32 = clamp(header.phase_parameters.x, (-0.95), 0.95);
                            let denominator: f32 = max(((1.0 + (anisotropy * anisotropy)) - ((2.0 * anisotropy) * cosine)), 0.0001);
                            phase = ((1.0 - (anisotropy * anisotropy)) / ((12.5663706144 * denominator) * sqrt(denominator)));
                        }
                        incoming_radiance = (incoming_radiance + (incident * phase));
                        isotropic_radiance = (isotropic_radiance + (incident * 0.0795774715459));
                    }
                }
                var scattered: vec3<f32> = incoming_radiance;
                let scattering_orders: u32 = min(u32((header.multiple_scattering_parameters.w + 0.5)), u32(8));
                let ratio: vec3<f32> = clamp((header.multiple_scattering_parameters.xyz * (1.0 - exp(((-extinction) * (exit_distance - entry))))), vec3<f32>(0.0), vec3<f32>(0.999));
                var order_weight: vec3<f32> = ratio;
                for (var order: i32 = 2; order < 9; order += 1) {
                    if ((u32(order) > scattering_orders)) {
                        break;
                    }
                    scattered = (scattered + (isotropic_radiance * order_weight));
                    order_weight = (order_weight * ratio);
                }
                combined_emission = (combined_emission + ((scattered * header.scattering_parameters.xyz) * header.scattering_parameters.w));
            }
        }
        let opacity: f32 = (1.0 - exp(((-combined_extinction) * step_size)));
        radiance = (radiance + ((transmittance * combined_emission) * opacity));
        transmittance = (transmittance * (1.0 - opacity));
        distance = (distance + step_size);
    }
    return vec4<f32>((radiance + (background.rgb * transmittance)), background.a);
}
