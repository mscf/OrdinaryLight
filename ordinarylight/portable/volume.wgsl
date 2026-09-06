struct Settings { viewport: vec4<f32>, camera: vec4<f32>, slice: vec4<f32>, dimensions: vec4<f32> }
@group(0) @binding(0) var<storage, read> density: array<f32>;
@group(0) @binding(1) var<uniform> settings: Settings;
@group(0) @binding(2) var<storage, read> transfer: array<vec4<f32>>;
struct Vertex { @builtin(position) position: vec4<f32>, @location(0) uv: vec2<f32> }
@vertex fn vertex_main(@builtin(vertex_index) i: u32) -> Vertex {
    var p = array<vec2<f32>,3>(vec2<f32>(-1,-1), vec2<f32>(3,-1), vec2<f32>(-1,3));
    return Vertex(vec4<f32>(p[i],0,1), p[i]);
}
fn color_at(p: vec3<f32>) -> vec4<f32> {
    let d=vec3<u32>(settings.dimensions.xyz);
    let cell=min(vec3<u32>(clamp(p*.5+.5,vec3<f32>(0),vec3<f32>(1))*vec3<f32>(d)), d-1u);
    let value=density[(cell.z*d.y+cell.y)*d.x+cell.x];
    let t=clamp(value/settings.viewport.w,0.,1.);
    let i=u32(t*f32(arrayLength(&transfer)-1u));
    return transfer[i];
}
@fragment fn fragment_main(v: Vertex) -> @location(0) vec4<f32> {
    let yaw=settings.camera.x; let pitch=settings.camera.y;
    let eye=settings.camera.z*vec3<f32>(cos(pitch)*sin(yaw),sin(pitch),cos(pitch)*cos(yaw));
    let forward=normalize(-eye); let right=normalize(cross(forward,vec3<f32>(0,1,0)));
    let up=cross(right,forward);
    let ray=normalize(forward+right*v.uv.x*settings.viewport.x/settings.viewport.y*.5+up*v.uv.y*.5);
    let inv=1./select(vec3<f32>(1e-7),ray,abs(ray)>vec3<f32>(1e-7));
    let a=(-vec3<f32>(1)-eye)*inv; let b=(vec3<f32>(1)-eye)*inv;
    let near=max(max(min(a.x,b.x),min(a.y,b.y)),min(a.z,b.z));
    let far=min(min(max(a.x,b.x),max(a.y,b.y)),max(a.z,b.z));
    var result=vec4<f32>(0);
    if far>=max(near,0.) {
        if settings.slice.x>0.5 {
            let t=(settings.slice.y*2.-1.-eye.z)*inv.z;
            if t>=max(near,0.) && t<=far { result=color_at(eye+t*ray); result.a=1.; }
        } else {
            var t=max(near,0.); var steps=0u;
            loop {
                if t>far || result.a>.995 || steps>=1024u { break; }
                let color=color_at(eye+t*ray);
                let alpha=1.-exp(-color.a*settings.viewport.z*settings.camera.w*40.);
                result=vec4<f32>(result.rgb+(1.-result.a)*alpha*color.rgb,result.a+(1.-result.a)*alpha);
                t+=settings.camera.w; steps++;
            }
        }
    }
    return vec4<f32>(result.rgb+(1.-result.a)*vec3<f32>(.025,.03,.04),1.);
}
