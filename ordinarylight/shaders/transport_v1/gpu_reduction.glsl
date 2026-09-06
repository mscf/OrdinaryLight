layout(local_size_x=64) in;
layout(set=0,binding=0,std430) readonly buffer Counts { uvec4 counts; };
layout(set=0,binding=1,std430) readonly buffer Groups { uvec4 groups[]; };
layout(set=0,binding=2,std430) readonly buffer Indices { uint indices[]; };
layout(set=0,binding=3,std430) readonly buffer Weights { vec2 weights[]; };
layout(set=0,binding=4,std430) readonly buffer Samples { OrdinaryLightSurfaceSample samples[]; };
layout(set=0,binding=5,std430) buffer State { uvec4 state; uvec4 commands[4]; };
struct SampleAccumulation { vec4 radiance; uvec4 counts; uvec4 events; };
layout(set=0,binding=6,std430) buffer Output { SampleAccumulation accumulated[]; };
layout(push_constant) uniform Constants { uint capacity; uint groups_capacity; uint output_capacity; uint mode; } pc;

void main() {
    uint i=gl_GlobalInvocationID.x;
    if(pc.mode==0u) {
        if(i!=0u) return;
        bool valid=counts.x<=pc.capacity && counts.y<=pc.groups_capacity &&
            counts.y<=counts.x && (counts.x==0u)==(counts.y==0u);
        state=uvec4(valid?counts.xy:uvec2(0),valid?0u:32u,0);
        commands[0]=uvec4((state.x+63u)/64u,1,1,0);
        commands[1]=commands[2]=commands[3]=uvec4(0,1,1,0);
        return;
    }
    if(pc.mode==2u) {
        if(i!=0u) return;
        commands[1]=uvec4(state.z==0u?(state.x+63u)/64u:0u,1,1,0);
        commands[2]=uvec4(state.z==0u?(state.y+63u)/64u:0u,1,1,0);
        commands[3]=uvec4(state.z!=0u?(pc.output_capacity+63u)/64u:0u,1,1,0);
        return;
    }
    if(pc.mode==3u) {
        if(i<pc.output_capacity && state.z!=0u) accumulated[i].counts.z|=32u;
        return;
    }
    if(i>=state.x) return;
    if(i<state.y) {
        uvec4 group=groups[i];
        bool valid=group.x<pc.output_capacity && group.y<=state.x &&
            group.z>0u && group.z<=state.x-group.y;
        if(i==0u) valid=valid && group.y==0u;
        else {
            uvec4 previous=groups[i-1u];
            valid=valid && previous.x<group.x && previous.y<=state.x &&
                previous.z<=state.x-previous.y && group.y==previous.y+previous.z;
        }
        if(i+1u==state.y) valid=valid && group.y+group.z==state.x;
        if(!valid) atomicOr(state.z,32u);
    }
    // Binary search a bounded group range. Invalid maps never dereference an
    // unvalidated input slot, even while other invocations diagnose the map.
    uint lower=0u, upper=state.y;
    while(lower<upper) {
        uint middle=lower+(upper-lower)/2u;
        if(groups[middle].y<=i) lower=middle+1u; else upper=middle;
    }
    if(lower==0u) { atomicOr(state.z,32u); return; }
    uvec4 group=groups[lower-1u];
    uint slot=indices[i];
    if(slot>=state.x || group.y>i || group.z>state.x ||
       i-group.y>=group.z) { atomicOr(state.z,32u); return; }
    if((i>group.y && indices[i-1u]>=slot) || samples[slot].identity.x!=group.x ||
       any(isnan(weights[slot])) || any(isinf(weights[slot])) ||
       any(lessThan(weights[slot],vec2(0)))) atomicOr(state.z,32u);
}
