"""Shader assembly shared by ray diagnostics and the public integrator."""

from importlib.resources import files

SCENE_BINDINGS = (
    "tlas",
    "vertices",
    "attributes",
    "triangles",
    "custom",
    "materials",
    "media",
    "boundaries",
)


def scene_source(scene):
    source = """#version 460
#extension GL_EXT_ray_query : require
layout(local_size_x=64) in;
struct CustomRecord { vec4 lower; vec4 upper; vec4 parameters; uvec4 metadata; };
struct TransportMaterialRecord { vec4 albedo_kind; vec4 emission; };
layout(set=0,binding=0) uniform accelerationStructureEXT transport_tlas;
layout(set=0,binding=1,std430) readonly buffer Vertices { vec4 transport_vertices[]; };
layout(set=0,binding=2,std430) readonly buffer Attributes { vec4 transport_attributes[]; };
layout(set=0,binding=3,std430) readonly buffer Triangles { uvec4 triangle_records[]; };
layout(set=0,binding=4,std430) readonly buffer Custom { CustomRecord custom_geometry[]; };
layout(set=0,binding=5,std430) readonly buffer Materials { TransportMaterialRecord transport_materials[]; };
layout(set=0,binding=6,std430) readonly buffer Media { vec4 optical_media[]; };
layout(set=0,binding=7,std430) readonly buffer Boundaries { uvec4 medium_boundaries[]; };
"""
    source += scene.custom_declarations
    source += f"\n#define OL_MATERIAL_COUNT {len(scene.materials)}u\n#define OL_BOUNDARY_COUNT {len(scene.boundaries)}u\n"
    source += "\n".join(program.source for program in scene.programs.values())
    source += """
uint ordinarylightCustomIntersect(uint program,vec3 origin,vec3 direction,float t_min,float t_max,
    vec4 parameters,float tolerance,uint max_steps,out float distance,out vec3 normal) {
    switch(program) {
"""
    for index, program in enumerate(scene.programs.values()):
        source += f"case {index}u: return {program.name}(origin,direction,t_min,t_max,parameters,tolerance,max_steps,distance,normal);\n"
    source += "default: return 2u;\n}\n}\n"
    source += (
        files("ordinarylight.shaders")
        .joinpath("transport_v1/intersections.glsl")
        .read_text()
    )
    return source
