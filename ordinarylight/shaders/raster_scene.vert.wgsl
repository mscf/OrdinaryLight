struct RasterCamera {
    view_projection: mat4x4<f32>,
    position_exposure: vec4<f32>,
}

@group(0) @binding(3) var<uniform> camera: RasterCamera;

struct SceneVertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) base_color: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) world_position: vec3<f32>,
    @location(3) material: vec4<f32>,
    @location(4) emission: vec3<f32>,
    @location(5) camera_position: vec3<f32>,
    @location(6) light_position_type: vec4<f32>,
    @location(7) light_color_ambient: vec4<f32>,
    @location(8) base_color_uv: vec2<f32>,
    @location(9) shadow_coordinate: vec4<f32>,
    @location(10) shadow_visibility: f32,
}

@vertex
fn main(
    @location(0) position: vec4<f32>,
    @location(1) base_color: vec3<f32>,
    @location(2) world_normal: vec3<f32>,
    @location(3) world_position: vec3<f32>,
    @location(4) material: vec4<f32>,
    @location(5) emission: vec3<f32>,
    @location(6) camera_position: vec3<f32>,
    @location(7) light_position_type: vec4<f32>,
    @location(8) light_color_ambient: vec4<f32>,
    @location(9) base_color_uv: vec2<f32>,
    @location(10) shadow_coordinate: vec4<f32>,
    @location(11) shadow_visibility: f32
) -> SceneVertexOutput {
    return SceneVertexOutput((camera.view_projection * position), base_color, world_normal, world_position, material, emission, camera.position_exposure.xyz, light_position_type, light_color_ambient, base_color_uv, shadow_coordinate, shadow_visibility);
}
