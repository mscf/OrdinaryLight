@fragment
fn main(
    @location(0) clip_depth: vec2<f32>
) -> @location(0) vec4<f32> {
    let depth: f32 = (clip_depth.x / max(abs(clip_depth.y), 1e-06));
    let normalized_depth: f32 = ((depth * 0.5) + 0.5);
    let encoded_depth: f32 = pow(max((1.0 - normalized_depth), 0.0), 0.25);
    return vec4<f32>(1.0, 1.0, 1.0, encoded_depth);
}
