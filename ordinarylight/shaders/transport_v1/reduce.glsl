#version 460
layout(local_size_x=64) in;
struct SampleAccumulation { vec4 radiance; uvec4 counts; uvec4 events; };
layout(set=0,binding=0,std430) readonly buffer Scratch { SampleAccumulation contributions[]; };
layout(set=0,binding=1,std430) buffer Output { SampleAccumulation accumulated[]; };
layout(set=0,binding=2,std430) readonly buffer Groups { uvec4 groups[]; };
layout(set=0,binding=3,std430) readonly buffer Indices { uint indices[]; };
layout(push_constant) uniform Constants { uint group_count; } pc;
void main() {
    uint i=gl_GlobalInvocationID.x; if(i>=pc.group_count) return;
    uvec4 group=groups[i];
    SampleAccumulation result=accumulated[group.x];
    for(uint j=0u;j<group.z;++j) {
        SampleAccumulation value=contributions[indices[group.y+j]];
        result.radiance+=value.radiance;
        result.counts.xy+=value.counts.xy;
        result.counts.z|=value.counts.z;
        result.counts.w+=value.counts.w;
        result.events+=value.events;
    }
    if(any(isnan(result.radiance))||any(isinf(result.radiance))) result.counts.z|=16u;
    accumulated[group.x]=result;
}
