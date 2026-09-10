"""Mechanical resource ABI and variant guards. Algorithms are typed helpers."""
LAYOUT = r'''// One invocation per input slot. A separate deterministic pass reduces slots
// into output identities, so duplicate destinations never race.
struct SampleAccumulation { vec4 radiance; uvec4 counts; uvec4 events; };
layout(set=0,binding=8,std430) readonly buffer Samples { OrdinaryLightSurfaceSample transport_samples[]; };
layout(set=0,binding=9,std430) buffer Accumulation { SampleAccumulation accumulated[]; };
layout(set=0,binding=10,std430) readonly buffer InitialStack { uvec4 initial_stack[]; };
layout(push_constant) uniform Constants {
    uint count; uint samples_per_element; uint max_bounces; uint sample_offset;
    uint seed; uint initial_depth; float tolerance; float ray_epsilon;
    uint max_steps; float max_distance; uint initial_stack_count; uint environment_nee;
    vec4 environment;
} pc;

#ifdef OL_GPU_REDUCTION
layout(set=0,binding=11,std430) readonly buffer GpuControl { uvec4 gpu_control; };
#endif
@main@
'''
