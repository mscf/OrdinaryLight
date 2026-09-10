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
    from ..shaders.scene_dispatch import intersection_dispatch
    source += intersection_dispatch(scene.programs.values())
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
    from ..shaders.scene_dispatch import transport_material_dispatch
    source += transport_material_dispatch(len(programs))
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
    from ..shaders.scene_dispatch import sampling_dispatch
    source += sampling_dispatch(scene.programs.values())
    source += files("ordinarylight.shaders").joinpath("transport_v1/emissive.glsl").read_text()
    return source
