struct GeometryProductCamera {
    current_view_projection: mat4x4<f32>,
    previous_view_projection: mat4x4<f32>,
    viewport: vec4<f32>,
    camera_position: vec4<f32>,
}

@group(0) @binding(0) var<uniform> camera: GeometryProductCamera;

struct GeometryProductVertexOutput {
    @invariant @builtin(position) position: vec4<f32>,
    @location(0) world_normal: vec3<f32>,
    @location(1) object_id: f32,
    @location(2) current_clip: vec4<f32>,
    @location(3) previous_clip: vec4<f32>,
    @location(4) world_position: vec3<f32>,
}

@vertex
fn main(
    @location(0) world_position: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) object_id: f32,
    @location(3) previous_world_position: vec3<f32>
) -> GeometryProductVertexOutput {
    let current_clip: vec4<f32> = (camera.current_view_projection * vec4<f32>(world_position, 1.0));
    let previous_clip: vec4<f32> = (camera.previous_view_projection * vec4<f32>(previous_world_position, 1.0));
    return GeometryProductVertexOutput(current_clip, world_normal, object_id, current_clip, previous_clip, world_position);
}
