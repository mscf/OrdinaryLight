struct ShadowVertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) clip_depth: vec2<f32>,
}

@vertex
fn main(
    @location(0) position: vec4<f32>
) -> ShadowVertexOutput {
    return ShadowVertexOutput(position, vec2<f32>(position.z, position.w));
}
