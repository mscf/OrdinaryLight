layout(local_size_x=64) in;
struct Record { vec4 lower; vec4 upper; vec4 parameters; uvec4 metadata; };
layout(set=0,binding=0,std430) readonly buffer Input { Record input_records[]; };
layout(set=0,binding=1,std430) writeonly buffer Output { Record output_records[]; };
layout(set=0,binding=2,std430) writeonly buffer Bounds { float bounds[]; };
layout(set=0,binding=3,std430) writeonly buffer Diagnostics { uint diagnostics[]; };
void main() {
    uint i=gl_GlobalInvocationID.x; if(i>=CAPACITY) return;
    Record record=input_records[i];
    uint status=0u;
    bool enabled=record.metadata.x!=0xffffffffu;
    if(enabled) {
        if(record.metadata.x>=PROGRAMS || record.metadata.y>=MATERIALS) status|=1u;
        if(any(isnan(record.lower)) || any(isinf(record.lower)) || any(isnan(record.upper)) || any(isinf(record.upper)) ||
           any(isnan(record.parameters)) || any(isinf(record.parameters)) || any(lessThanEqual(record.upper.xyz,record.lower.xyz))) status|=2u;
        uint boundary=boundaryIndex(record.metadata.z);
        if((record.metadata.z!=0xffffffffu && boundary==0xffffffffu) ||
           dielectricMaterial(record.metadata.y)!=(boundary!=0xffffffffu)) status|=4u;
        record.metadata.y+=TRIANGLES;
        record.metadata.z=boundary;
    }
    if(!enabled || status!=0u) {
        record.lower=vec4(0); record.upper=vec4(1e-4); record.parameters=vec4(0);
        record.metadata=uvec4(0xffffffffu,0,0xffffffffu,0);
    }
    output_records[i]=record;
    for(uint axis=0u;axis<3u;++axis) {
        bounds[i*6u+axis]=record.lower[axis]; bounds[i*6u+axis+3u]=record.upper[axis];
    }
    diagnostics[i]=status;
}
