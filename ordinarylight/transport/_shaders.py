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
struct TransportMaterialRecord { vec4 albedo_kind; vec4 emission; vec4 optics; };
layout(set=0,binding=0) uniform accelerationStructureEXT transport_tlas;
layout(set=0,binding=1,std430) readonly buffer Vertices { vec4 transport_vertices[]; };
layout(set=0,binding=2,std430) readonly buffer Attributes { vec4 transport_attributes[]; };
layout(set=0,binding=3,std430) readonly buffer Triangles { uvec4 triangle_records[]; };
layout(set=0,binding=4,std430) readonly buffer Custom { CustomRecord custom_geometry[]; };
layout(set=0,binding=5,std430) readonly buffer Materials { TransportMaterialRecord transport_materials[]; };
layout(set=0,binding=6,std430) readonly buffer Media { vec4 optical_media[]; };
layout(set=0,binding=7,std430) readonly buffer Boundaries { uvec4 medium_boundaries[]; };
"""
    source += "layout(set=0,binding=12,std430) readonly buffer AnalyticLights { vec4 analytic_lights[]; };\n"
    source += f"#define OL_ANALYTIC_LIGHT_COUNT {len(scene.lights)}u\n"
    source += scene.custom_declarations
    source += f"\n#define OL_MATERIAL_COUNT {len(scene.materials)}u\n#define OL_BOUNDARY_COUNT {len(scene.boundaries)}u\n"
    source += f"#define OL_CUSTOM_MATERIAL_OFFSET {len(scene.materials) - len(scene._custom_materials)}u\n"
    source += """
#define OL_HIT_MATERIAL 1u
#define OL_HIT_BOUNDARY 2u
#define OL_HIT_IDENTITY 4u
#define OL_HIT_UV 8u
#define OL_HIT_SHADING_NORMAL 16u
#define OL_NO_BOUNDARY 0xffffffffu
struct OrdinaryLightCustomHit {
    float distance;
    vec3 geometric_normal;
    uint flags;
    uint material;
    uint boundary;
    uint identity;
    vec2 uv;
    vec3 shading_normal;
};
"""
    source += "\n".join(program.source for program in scene.programs.values())
    source += """
uint ordinarylightCustomIntersect(uint program,vec3 origin,vec3 direction,float t_min,float t_max,
    vec4 parameters,float tolerance,uint max_steps,inout OrdinaryLightCustomHit result) {
    switch(program) {
"""
    for index, program in enumerate(scene.programs.values()):
        outputs = "result" if program.hit_version == 2 else "result.distance,result.geometric_normal"
        source += f"case {index}u: return {program.name}(origin,direction,t_min,t_max,parameters,tolerance,max_steps,{outputs});\n"
    source += "default: return 2u;\n}\n}\n"
    source += (
        files("ordinarylight.shaders")
        .joinpath("transport_v1/intersections.glsl")
        .read_text()
    )
    return source


def material_source(scene):
    """Compile the same parameter programs used by camera material evaluation."""
    from . import shader_source
    from ..materials import builtin_material

    source = shader_source("types") + shader_source("material_contracts")
    programs = [material.program or builtin_material for material in scene.materials]
    source += "\n".join(
        program.glsl(f"olMaterial_{i}") for i, program in enumerate(programs)
    )
    source += """
MaterialEvaluation ordinarylightEvaluateMaterial(uint index, OrdinaryLightHit hit,
    vec3 direction, float bounce, float current_ior, float exterior_ior, vec2 randoms) {
    TransportMaterialRecord fixed_material=transport_materials[index];
    MaterialData material;
    material.base_roughness=vec4(fixed_material.albedo_kind.rgb,fixed_material.optics.x);
    material.emission_metallic=vec4(fixed_material.emission.rgb,fixed_material.optics.y);
    material.attenuation_transmission=vec4(1,1,1,float(fixed_material.albedo_kind.w==1.0));
    material.ior_distance=vec4(hit.boundary.x!=0xffffffffu?optical_media[medium_boundaries[hit.boundary.x].y].a:fixed_material.optics.z,1e30,0,0);
    bool entering=dot(direction,hit.geometric_normal.xyz)<0.0;
    vec3 normal=entering?hit.shading_normal.xyz:-hit.shading_normal.xyz;
    vec2 uv=vec2(hit.geometric_normal.w,hit.shading_normal.w);
    switch(index) {
"""
    for i in range(len(programs)):
        source += f"case {i}u: return olMaterial_{i}(material,normal,uv,direction,entering,randoms.x,randoms.y,bounce,current_ior,exterior_ior);\n"
    source += "default: return olMaterial_0(material,normal,uv,direction,entering,randoms.x,randoms.y,bounce,current_ior,exterior_ior);\n}\n}\n"
    return source


def surface_sampling_source(scene):
    """Shared forward/reverse area densities for emissive next-event estimation."""
    samplers = {}
    for program in scene.programs.values():
        if program.sampling is not None:
            sampler = program.sampling
            previous = samplers.setdefault(sampler.name, sampler)
            if previous != sampler:
                raise ValueError("Conflicting custom surface sampling program names")
    source = (
        f"\n#define OL_TRIANGLE_COUNT {scene.triangle_count}u\n"
        f"#define OL_SURFACE_SLOT_COUNT {scene.triangle_count + scene.custom_capacity}u\n"
        + "\n".join(sampler.source for sampler in samplers.values())
    )
    source += """
uint ordinarylightCustomSurfaceSample(uint program,vec4 parameters,vec3 randoms,
    out vec3 position,out vec3 normal,out float area_pdf) {
    switch(program) {
"""
    for index, program in enumerate(scene.programs.values()):
        if program.sampling is not None:
            source += f"case {index}u: return {program.sampling.name}(parameters,randoms,position,normal,area_pdf);\n"
    source += "default: return 0u;\n}\n}\n"
    source += """
float ordinarylightCustomSurfacePdf(uint program,vec4 parameters,vec3 position,vec3 normal) {
    switch(program) {
"""
    for index, program in enumerate(scene.programs.values()):
        if program.sampling is not None:
            source += f"case {index}u: return {program.sampling.name}_pdf(parameters,position,normal);\n"
    source += "default: return 0.0;\n}\n}\n"
    source += files("ordinarylight.shaders").joinpath("transport_v1/emissive.glsl").read_text()
    return source
