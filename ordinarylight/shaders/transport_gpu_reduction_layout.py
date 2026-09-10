"""Mechanical resource ABI and variant guards. Algorithms are typed helpers."""
LAYOUT = r'''layout(local_size_x=64) in;
layout(set=0,binding=0,std430) readonly buffer Counts { uvec4 counts; };
layout(set=0,binding=1,std430) readonly buffer Groups { uvec4 groups[]; };
layout(set=0,binding=2,std430) readonly buffer Indices { uint indices[]; };
layout(set=0,binding=3,std430) readonly buffer Weights { vec2 weights[]; };
layout(set=0,binding=4,std430) readonly buffer Samples { OrdinaryLightSurfaceSample samples[]; };
layout(set=0,binding=5,std430) buffer State { uvec4 state; uvec4 commands[4]; };
struct SampleAccumulation { vec4 radiance; uvec4 counts; uvec4 events; };
layout(set=0,binding=6,std430) buffer Output { SampleAccumulation accumulated[]; };
layout(push_constant) uniform Constants { uint capacity; uint groups_capacity; uint output_capacity; uint mode; } pc;

@main@
'''
