struct ComposeConstants {
    extent: vec4<f32>,
}

@group(0) @binding(0) var diffuse: texture_storage_2d<rgba16float, read>;
@group(0) @binding(1) var specular: texture_storage_2d<rgba16float, read>;
@group(0) @binding(2) var view_z: texture_storage_2d<r32float, read>;
@group(0) @binding(3) var output_hdr: texture_storage_2d<rgba16float, write>;
@group(0) @binding(4) var<uniform> constants: ComposeConstants;

@compute @workgroup_size(8, 8, 1)
fn main(
    @builtin(global_invocation_id) global_invocation_id: vec3<u32>,
    @builtin(local_invocation_id) local_invocation_id: vec3<u32>,
    @builtin(local_invocation_index) local_invocation_index: u32,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>,
) {
    let pixel: vec2<i32> = vec2<i32>(global_invocation_id.xy);
    let extent: vec2<i32> = vec2<i32>(constants.extent.xy);
    if (((pixel.x >= extent.x) || (pixel.y >= extent.y))) {
        return;
    }
    if ((textureLoad(view_z, pixel).r == 0.0)) {
        return;
    }
    let diffuse_value: vec4<f32> = textureLoad(diffuse, pixel);
    let specular_value: vec4<f32> = textureLoad(specular, pixel);
    textureStore(output_hdr, pixel, vec4<f32>((diffuse_value.rgb + specular_value.rgb), 1.0));
}
