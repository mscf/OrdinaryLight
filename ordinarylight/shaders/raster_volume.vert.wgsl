struct VolumeVertexOutput {
    @invariant @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn main(
    @location(0) clip_position: vec2<f32>,
    @location(1) texture_coordinate: vec2<f32>
) -> VolumeVertexOutput {
    return VolumeVertexOutput(vec4<f32>(clip_position, 0.0, 1.0), texture_coordinate);
}
