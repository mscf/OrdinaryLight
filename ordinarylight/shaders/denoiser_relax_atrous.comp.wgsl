struct AtrousConstants {
    extent_step: vec4<f32>,
    weights: vec4<f32>,
}

@group(0) @binding(0) var input_radiance: texture_storage_2d<rgba16float, read>;
@group(0) @binding(1) var normal_roughness: texture_storage_2d<rgba16float, read>;
@group(0) @binding(2) var view_z: texture_storage_2d<r32float, read>;
@group(0) @binding(3) var material_id: texture_storage_2d<r32uint, read>;
@group(0) @binding(4) var output_radiance: texture_storage_2d<rgba16float, write>;
@group(0) @binding(5) var<uniform> constants: AtrousConstants;

@compute @workgroup_size(8, 8, 1)
fn main(
    @builtin(global_invocation_id) global_invocation_id: vec3<u32>,
    @builtin(local_invocation_id) local_invocation_id: vec3<u32>,
    @builtin(local_invocation_index) local_invocation_index: u32,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>,
) {
    let pixel: vec2<i32> = vec2<i32>(global_invocation_id.xy);
    let extent: vec2<i32> = vec2<i32>(constants.extent_step.xy);
    if (((pixel.x >= extent.x) || (pixel.y >= extent.y))) {
        return;
    }
    let step_width: i32 = i32(constants.extent_step.z);
    let center: vec4<f32> = textureLoad(input_radiance, pixel);
    let center_normal: vec3<f32> = textureLoad(normal_roughness, pixel).xyz;
    let center_depth: f32 = textureLoad(view_z, pixel).r;
    let center_material: u32 = textureLoad(material_id, pixel).r;
    let center_luma: f32 = dot(center.rgb, vec3<f32>(0.2126, 0.7152, 0.0722));
    var total: vec3<f32> = center.rgb;
    var weight_sum: f32 = 1.0;
    var neighborhood_luma_sum: f32 = 0.0;
    var neighborhood_luma_square_sum: f32 = 0.0;
    var neighborhood_count: f32 = 0.0;
    for (var y: i32 = (-1); y < 2; y += 1) {
        for (var x: i32 = (-1); x < 2; x += 1) {
            if (((x == 0) && (y == 0))) {
                continue;
            }
            let sample_pixel: vec2<i32> = (pixel + (vec2<i32>(x, y) * step_width));
            if (((sample_pixel.x < 0) || (sample_pixel.y < 0))) {
                continue;
            }
            if (((sample_pixel.x >= extent.x) || (sample_pixel.y >= extent.y))) {
                continue;
            }
            let sample_depth: f32 = textureLoad(view_z, sample_pixel).r;
            let sample_material: u32 = textureLoad(material_id, sample_pixel).r;
            if ((sample_material != center_material)) {
                continue;
            }
            if (((sample_depth == 0.0) != (center_depth == 0.0))) {
                continue;
            }
            let sample_normal: vec3<f32> = textureLoad(normal_roughness, sample_pixel).xyz;
            let sample: vec4<f32> = textureLoad(input_radiance, sample_pixel);
            let normal_weight: f32 = pow(max(dot(center_normal, sample_normal), 0.0), constants.weights.x);
            let depth_scale: f32 = max((abs(center_depth) * constants.weights.y), 0.001);
            let depth_weight: f32 = exp(((-abs((sample_depth - center_depth))) / depth_scale));
            let sample_luma: f32 = dot(sample.rgb, vec3<f32>(0.2126, 0.7152, 0.0722));
            neighborhood_luma_sum = (neighborhood_luma_sum + sample_luma);
            neighborhood_luma_square_sum = (neighborhood_luma_square_sum + (sample_luma * sample_luma));
            neighborhood_count = (neighborhood_count + 1.0);
            let color_scale: f32 = max((abs(center_luma) / constants.weights.z), 0.02);
            let color_weight: f32 = exp(((-abs((sample_luma - center_luma))) / color_scale));
            var kernel: f32 = 0.25;
            if (((x == 0) || (y == 0))) {
                kernel = 0.5;
            }
            let weight: f32 = (((kernel * normal_weight) * depth_weight) * color_weight);
            total = (total + (sample.rgb * weight));
            weight_sum = (weight_sum + weight);
        }
    }
    if (((constants.weights.w > 0.5) && (neighborhood_count > 0.0))) {
        let neighborhood_luma_mean: f32 = (neighborhood_luma_sum / neighborhood_count);
        let neighborhood_luma_variance: f32 = max(((neighborhood_luma_square_sum / neighborhood_count) - (neighborhood_luma_mean * neighborhood_luma_mean)), 0.0);
        let firefly_limit: f32 = ((neighborhood_luma_mean + (4.0 * sqrt(neighborhood_luma_variance))) + 0.02);
        if ((center_luma > firefly_limit)) {
            let clamped_center: vec3<f32> = (center.rgb * (firefly_limit / max(center_luma, 1e-06)));
            total = ((total - center.rgb) + clamped_center);
        }
    }
    textureStore(output_radiance, pixel, vec4<f32>((total / max(weight_sum, 1e-06)), center.a));
}
