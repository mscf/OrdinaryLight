struct AtrousConstants {
    extent_step: vec4<f32>,
    weights: vec4<f32>,
}

@group(0) @binding(0) var input_radiance: texture_storage_2d<rgba16float, read>;
@group(0) @binding(1) var normal_roughness: texture_storage_2d<rgba16float, read>;
@group(0) @binding(2) var view_z: texture_storage_2d<r32float, read>;
@group(0) @binding(3) var material_id: texture_storage_2d<r32uint, read>;
@group(0) @binding(4) var output_radiance: texture_storage_2d<rgba16float, write>;
@group(0) @binding(7) var<uniform> constants: AtrousConstants;
@group(0) @binding(5) var specular_input: texture_storage_2d<rgba16float, read>;
@group(0) @binding(6) var specular_output: texture_storage_2d<rgba16float, write>;

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
    let center_spec: vec4<f32> = textureLoad(specular_input, pixel);
    let center_normal: vec3<f32> = textureLoad(normal_roughness, pixel).xyz;
    let center_depth: f32 = textureLoad(view_z, pixel).r;
    let center_material: u32 = textureLoad(material_id, pixel).r;
    let center_luma: f32 = dot(center.rgb, vec3<f32>(0.2126, 0.7152, 0.0722));
    let center_luma_spec: f32 = dot(center_spec.rgb, vec3<f32>(0.2126, 0.7152, 0.0722));
    var total: vec3<f32> = center.rgb;
    var total_spec: vec3<f32> = center_spec.rgb;
    var weight_sum: f32 = 1.0;
    var weight_sum_spec: f32 = 1.0;
    var neighborhood_luma_sum: f32 = 0.0;
    var neighborhood_luma_sum_spec: f32 = 0.0;
    var neighborhood_luma_square_sum: f32 = 0.0;
    var neighborhood_luma_square_sum_spec: f32 = 0.0;
    var neighborhood_count: f32 = 0.0;
    var neighborhood_count_spec: f32 = 0.0;
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
            let sample_spec: vec4<f32> = textureLoad(specular_input, sample_pixel);
            let normal_weight: f32 = pow(max(dot(center_normal, sample_normal), 0.0), constants.weights.x);
            let depth_scale: f32 = max((abs(center_depth) * constants.weights.y), 0.001);
            let depth_weight: f32 = exp(((-abs((sample_depth - center_depth))) / depth_scale));
            let sample_luma: f32 = dot(sample.rgb, vec3<f32>(0.2126, 0.7152, 0.0722));
            let sample_luma_spec: f32 = dot(sample_spec.rgb, vec3<f32>(0.2126, 0.7152, 0.0722));
            neighborhood_luma_sum = (neighborhood_luma_sum + sample_luma);
            neighborhood_luma_sum_spec = (neighborhood_luma_sum_spec + sample_luma_spec);
            neighborhood_luma_square_sum = (neighborhood_luma_square_sum + (sample_luma * sample_luma));
            neighborhood_luma_square_sum_spec = (neighborhood_luma_square_sum_spec + (sample_luma_spec * sample_luma_spec));
            neighborhood_count = (neighborhood_count + 1.0);
            neighborhood_count_spec = (neighborhood_count_spec + 1.0);
            let color_scale: f32 = max((abs(center_luma) / constants.weights.z), 0.02);
            let color_scale_spec: f32 = max((abs(center_luma_spec) / constants.weights.z), 0.02);
            let color_weight: f32 = exp(((-abs((sample_luma - center_luma))) / color_scale));
            let color_weight_spec: f32 = exp(((-abs((sample_luma_spec - center_luma_spec))) / color_scale_spec));
            var kernel: f32 = 0.25;
            if (((x == 0) || (y == 0))) {
                kernel = 0.5;
            }
            let weight: f32 = (((kernel * normal_weight) * depth_weight) * color_weight);
            let weight_spec: f32 = (((kernel * normal_weight) * depth_weight) * color_weight_spec);
            total = (total + (sample.rgb * weight));
            total_spec = (total_spec + (sample_spec.rgb * weight_spec));
            weight_sum = (weight_sum + weight);
            weight_sum_spec = (weight_sum_spec + weight_spec);
        }
    }
    if (((constants.weights.w > 0.5) && (neighborhood_count_spec > 0.0))) {
        let neighborhood_luma_mean_spec: f32 = (neighborhood_luma_sum_spec / neighborhood_count_spec);
        let neighborhood_luma_variance_spec: f32 = max(((neighborhood_luma_square_sum_spec / neighborhood_count_spec) - (neighborhood_luma_mean_spec * neighborhood_luma_mean_spec)), 0.0);
        let firefly_limit_spec: f32 = ((neighborhood_luma_mean_spec + (4.0 * sqrt(neighborhood_luma_variance_spec))) + 0.02);
        if ((center_luma_spec > firefly_limit_spec)) {
            let clamped_center_spec: vec3<f32> = (center_spec.rgb * (firefly_limit_spec / max(center_luma_spec, 1e-06)));
            total_spec = ((total_spec - center_spec.rgb) + clamped_center_spec);
        }
    }
    textureStore(output_radiance, pixel, vec4<f32>((total / max(weight_sum, 1e-06)), center.a));
    textureStore(specular_output, pixel, vec4<f32>((total_spec / max(weight_sum_spec, 1e-06)), center_spec.a));
}
