"""Mechanical resource ABI and variant guards. Algorithms are typed helpers."""
LAYOUT = r'''#version 460
layout(local_size_x=64) in;
struct SampleAccumulation { vec4 radiance; uvec4 counts; uvec4 events; };
layout(set=0,binding=0,std430) readonly buffer Scratch { SampleAccumulation contributions[]; };
layout(set=0,binding=1,std430) buffer Output { SampleAccumulation accumulated[]; };
layout(set=0,binding=2,std430) readonly buffer Groups { uvec4 groups[]; };
layout(set=0,binding=3,std430) readonly buffer Indices { uint indices[]; };
layout(set=0,binding=4,std430) readonly buffer Weights { vec2 weights[]; };
layout(push_constant) uniform Constants { uint group_count; } pc;
#ifdef OL_GPU_REDUCTION
layout(set=0,binding=5,std430) readonly buffer GpuControl { uvec4 gpu_control; };
#endif
@main@
'''
