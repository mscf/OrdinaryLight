struct WavePathState {
    throughput: vec4<f32>,
    radiance: vec4<f32>,
    metadata: vec4<u32>,
}

struct SecondaryPathState {
    position_valid: vec4<f32>,
    normal_pdf: vec4<f32>,
    primary_throughput: vec4<f32>,
    primary_radiance: vec4<f32>,
    diffuse_radiance_hit_distance: vec4<f32>,
    specular_radiance_hit_distance: vec4<f32>,
    primary_position: vec4<f32>,
}

struct PrepareCamera {
    origin: vec4<f32>,
    forward: vec4<f32>,
    right: vec4<f32>,
    up: vec4<f32>,
}

struct PrepareConstants {
    extent_paths: vec4<u32>,
}

@group(0) @binding(0) var<storage, read> paths: array<WavePathState>;
@group(0) @binding(1) var<storage, read> secondary_paths: array<SecondaryPathState>;
@group(0) @binding(2) var packed_normal: texture_storage_2d<r32uint, read>;
@group(0) @binding(3) var packed_material: texture_storage_2d<r32uint, read>;
@group(0) @binding(4) var diffuse_output: texture_storage_2d<rgba16float, write>;
@group(0) @binding(5) var specular_output: texture_storage_2d<rgba16float, write>;
@group(0) @binding(6) var normal_roughness_output: texture_storage_2d<rgba16float, write>;
@group(0) @binding(7) var view_z_output: texture_storage_2d<r32float, write>;
@group(0) @binding(8) var motion_output: texture_storage_2d<rgba16float, write>;
@group(0) @binding(9) var<storage, read> current_camera: PrepareCamera;
@group(0) @binding(10) var<storage, read> previous_camera: PrepareCamera;
@group(0) @binding(11) var<uniform> constants: PrepareConstants;

fn prepare_decode_normal(encoded: vec2<f32>) -> vec3<f32> {
    var normal: vec3<f32> = vec3<f32>(encoded, ((1.0 - abs(encoded.x)) - abs(encoded.y)));
    if ((normal.z < 0.0)) {
        let folded: vec2<f32> = ((1.0 - abs(normal.yx)) * sign(normal.xy));
        normal.x = folded.x;
        normal.y = folded.y;
    }
    return normalize(normal);
}

fn prepare_unpack_normal(packed: u32) -> vec3<f32> {
    let unit: vec2<f32> = (vec2<f32>(f32((packed & u32(32767))), f32(((packed >> u32(15)) & u32(32767)))) / 32767.0);
    return prepare_decode_normal(((unit * 2.0) - 1.0));
}

fn prepare_previous_pixel(world_position: vec3<f32>, extent: vec2<i32>) -> vec3<f32> {
    let offset: vec3<f32> = (world_position - previous_camera.origin.xyz);
    let depth: f32 = dot(offset, previous_camera.forward.xyz);
    let vertical_scale: f32 = length(previous_camera.up.xyz);
    let aspect: f32 = (f32(extent.x) / f32(extent.y));
    if (((depth <= 0.0001) || (vertical_scale <= 0.0001))) {
        return vec3<f32>((-1.0), (-1.0), depth);
    }
    let ndc: vec2<f32> = vec2<f32>((dot(offset, normalize(previous_camera.right.xyz)) / ((depth * aspect) * vertical_scale)), ((-dot(offset, normalize(previous_camera.up.xyz))) / (depth * vertical_scale)));
    let pixel: vec2<f32> = ((((ndc * 0.5) + 0.5) * vec2<f32>(extent)) - 0.5);
    return vec3<f32>(pixel, depth);
}

@compute @workgroup_size(64, 1, 1)
fn main(
    @builtin(global_invocation_id) global_invocation_id: vec3<u32>,
    @builtin(local_invocation_id) local_invocation_id: vec3<u32>,
    @builtin(local_invocation_index) local_invocation_index: u32,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>,
) {
    let path_index: u32 = global_invocation_id.x;
    let path_count: u32 = constants.extent_paths.z;
    if ((path_index >= path_count)) {
        return;
    }
    let extent: vec2<i32> = vec2<i32>(constants.extent_paths.xy);
    let pixel_index: u32 = paths[path_index].metadata.x;
    if ((pixel_index >= u32((extent.x * extent.y)))) {
        return;
    }
    let pixel: vec2<i32> = vec2<i32>(i32((pixel_index % u32(extent.x))), i32((pixel_index / u32(extent.x))));
    let secondary: SecondaryPathState = secondary_paths[path_index];
    let valid: bool = (secondary.primary_position.w > 0.5);
    if ((!valid)) {
        textureStore(diffuse_output, pixel, vec4<f32>(0.0));
        textureStore(specular_output, pixel, vec4<f32>(0.0));
        textureStore(normal_roughness_output, pixel, vec4<f32>(0.0));
        textureStore(view_z_output, pixel, vec4<f32>(0.0));
        textureStore(motion_output, pixel, vec4<f32>(0.0));
        return;
    }
    let world_position: vec3<f32> = secondary.primary_position.xyz;
    let normal: vec3<f32> = prepare_unpack_normal(textureLoad(packed_normal, pixel).x);
    let roughness: f32 = clamp((secondary.primary_position.w - 1.0), 0.0, 1.0);
    let view_z: f32 = dot((world_position - current_camera.origin.xyz), current_camera.forward.xyz);
    let old: vec3<f32> = prepare_previous_pixel(world_position, extent);
    var motion: vec2<f32> = (old.xy - vec2<f32>(pixel));
    if ((((old.z <= 0.0001) || any((old.xy < vec2<f32>((-0.5))))) || any((old.xy >= (vec2<f32>(extent) - 0.5))))) {
        motion = vec2<f32>(0.0);
    }
    textureStore(diffuse_output, pixel, secondary.diffuse_radiance_hit_distance);
    textureStore(specular_output, pixel, secondary.specular_radiance_hit_distance);
    textureStore(normal_roughness_output, pixel, vec4<f32>(normal, roughness));
    textureStore(view_z_output, pixel, vec4<f32>(view_z));
    textureStore(motion_output, pixel, vec4<f32>(motion, 0.0, 0.0));
}
