"""Attach the diagnostic kernel to the existing transport resource layout."""
import ordinaryshade as osh
from . import diagnostic_programs as p


def diagnostic_source(*, colors=False, hdr=False):
    layout = f'#define OL_QUERY_COLORS {int(colors)}\n#define OL_QUERY_HDR {int(hdr)}\n'
    layout += '''struct RayInput { vec4 origin; vec4 direction; };
layout(set=0,binding=8,std430) readonly buffer Rays { RayInput rays[]; };
layout(push_constant) uniform Constants { uint count; float t_min; float t_max; float tolerance; uint max_steps; } pc;
'''
    if colors:
        layout += '''struct OrdinaryLightColorHit { vec4 color; uvec4 identity; };
layout(set=0,binding=10,std430) readonly buffer QueryColors { vec4 ordinarylightQueryColors[]; };
layout(set=0,binding=9,std430) writeonly buffer Hits { OrdinaryLightColorHit color_hits[]; };
'''
    else:
        layout += 'layout(set=0,binding=9,std430) writeonly buffer Hits { OrdinaryLightHit hits[]; };\n'
    if hdr:
        layout += 'layout(set=0,binding=11,rgba32f) writeonly uniform image2D ordinarylightQueryHdr;\n'
    return layout + osh.compile_function(p.main, externals=(osh.external(p.ordinarylightIntersect.function),),
        external_values=dict(gl_GlobalInvocationID=osh.uvec3, pc=p.QueryConstants,
            rays=osh.runtime_array(p.RayInput), hits=osh.runtime_array(p.OrdinaryLightHit),
            color_hits=osh.runtime_array(p.OrdinaryLightColorHit), ordinarylightQueryColors=osh.runtime_array(osh.vec4),
            ordinarylightQueryHdr=osh.storage_image('rgba32f', access='write'))).source
