"""Mechanical resource ABI and variant guards. Algorithms are typed helpers."""
LAYOUT = r'''layout(local_size_x=64) in;
struct Record { vec4 lower; vec4 upper; vec4 parameters; uvec4 metadata; };
layout(set=0,binding=0,std430) readonly buffer Input { Record input_records[]; };
layout(set=0,binding=1,std430) writeonly buffer Output { Record output_records[]; };
layout(set=0,binding=2,std430) writeonly buffer Bounds { float bounds[]; };
layout(set=0,binding=3,std430) writeonly buffer Diagnostics { uint diagnostics[]; };
@main@
'''
