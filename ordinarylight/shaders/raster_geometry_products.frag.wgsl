struct GeometryProductCamera {
    current_view_projection: mat4x4<f32>,
    previous_view_projection: mat4x4<f32>,
    viewport: vec4<f32>,
    camera_position: vec4<f32>,
}

@group(0) @binding(0) var<uniform> camera: GeometryProductCamera;

struct GeometryProductOutput {
    @location(0) normal_depth: vec4<f32>,
    @location(1) motion_object: vec4<f32>,
}

@fragment
fn main(
    @location(0) world_normal: vec3<f32>,
    @location(1) object_id: f32,
    @location(2) current_clip: vec4<f32>,
    @location(3) previous_clip: vec4<f32>,
    @location(4) world_position: vec3<f32>
) -> GeometryProductOutput {
    let current_ndc: vec2<f32> = (current_clip.xy / max(abs(current_clip.w), 1e-06));
    let previous_ndc: vec2<f32> = (previous_clip.xy / max(abs(previous_clip.w), 1e-06));
    let motion: vec2<f32> = ((current_ndc - previous_ndc) * vec2<f32>((camera.viewport.x * 0.5), ((-camera.viewport.y) * 0.5)));
    return GeometryProductOutput(vec4<f32>(normalize(world_normal), length((world_position - camera.camera_position.xyz))), vec4<f32>(motion, round(object_id), 0.0));
}
