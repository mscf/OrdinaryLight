"""Compilation of generated material programs into Vulkan SPIR-V shaders."""

from functools import lru_cache
from importlib.resources import files
from pathlib import Path
import shutil
import subprocess
import tempfile

from ..materials import MaterialProgram, material_dispatch_glsl


_BEGIN = "// WAVE_RENDER_MATERIAL_BEGIN"
_END = "// WAVE_RENDER_MATERIAL_END"


def find_glsl_compiler():
    """Return a supported GLSL compiler executable, or ``None``."""
    compiler = shutil.which("glslangValidator") or shutil.which("glslc")
    if compiler:
        return compiler
    development_compiler = (
        Path(__file__).resolve().parents[2]
        / ".tools/glslang/usr/bin/glslangValidator"
    )
    return str(development_compiler) if development_compiler.is_file() else None


def material_shader_source(
    shader_name, program, *, attribute_layout=None, material_modifier=None,
):
    """Inject a material program into one of the packaged shader templates."""
    programs = (program,) if isinstance(program, MaterialProgram) else tuple(program)
    if not programs or any(not isinstance(item, MaterialProgram) for item in programs):
        raise TypeError("programs must be MaterialProgram objects created by @material")
    source = files("ordinarylight").joinpath(f"shaders/{shader_name}").read_text()
    begin = source.find(_BEGIN)
    end = source.find(_END)
    if begin < 0 or end < begin:
        raise RuntimeError(f"Shader template {shader_name!r} has no material insertion point")
    required = {}
    for item in programs:
        for name, components in item.required_attributes:
            previous = required.get(name)
            if previous is not None and previous != components:
                raise ValueError(
                    f"attribute {name!r} has conflicting material declarations"
                )
            required[name] = components
    slots = None
    if required:
        if attribute_layout is None:
            raise ValueError(
                "shader-visible material attributes require an "
                "attribute_layout"
            )
        layout_channels = dict(attribute_layout.channels)
        for name, components in required.items():
            if layout_channels.get(name) != components:
                raise ValueError(
                    f"attribute layout does not provide {name!r} with "
                    f"{components} components"
                )
        slots = {name: attribute_layout.slot(name) for name in required}
    generated = (
        f"{_BEGIN}\n"
        f"{material_dispatch_glsl(programs, attribute_slots=slots, material_modifier=material_modifier)}\n"
        f"{_END}"
    )
    if required:
        channel_count = len(attribute_layout.channels)
        support = f"""layout(set = 0, binding = 15, std430) readonly buffer WaveCustomAttributeBuffer {{
    vec4 wave_custom_attributes[];
}};
uint wave_attribute_primitive;
vec3 wave_attribute_weights;
vec4 waveVertexAttribute4(uint slot)
{{
    uint base = wave_attribute_primitive * {3 * channel_count}u + slot;
    return wave_custom_attributes[base] * wave_attribute_weights.x
        + wave_custom_attributes[base + {channel_count}u] * wave_attribute_weights.y
        + wave_custom_attributes[base + {2 * channel_count}u] * wave_attribute_weights.z;
}}
float waveVertexAttribute1(uint slot) {{ return waveVertexAttribute4(slot).x; }}
vec2 waveVertexAttribute2(uint slot) {{ return waveVertexAttribute4(slot).xy; }}
vec3 waveVertexAttribute3(uint slot) {{ return waveVertexAttribute4(slot).xyz; }}
"""
        generated = support + generated
    result = source[:begin] + generated + source[end + len(_END):]
    if required:
        call = "MaterialEvaluation evaluated = evaluateMaterial("
        state = (
            "wave_attribute_primitive = primitive;\n"
            "        wave_attribute_weights = weights;\n"
            "        " + call
        )
        if call not in result:
            raise RuntimeError(
                f"Shader template {shader_name!r} has no material call site"
            )
        result = result.replace(call, state)
    return result


def _expanded_shader_source(shader_name, seen=()):
    """Read a packaged shader and inline its local GLSL includes."""
    if shader_name in seen:
        raise RuntimeError(f"cyclic shader include involving {shader_name!r}")
    source = files("ordinarylight").joinpath(f"shaders/{shader_name}").read_text()
    result = []
    for line in source.splitlines():
        stripped = line.strip()
        if stripped.startswith('#include "') and stripped.endswith('"'):
            include = stripped[len('#include "'):-1]
            result.append(_expanded_shader_source(include, (*seen, shader_name)))
        else:
            result.append(line)
    return "\n".join(result) + "\n"


def wavefront_material_shader_source(
    shader_name, programs, *, attribute_layout, attribute_binding,
    overlapping_volumes=False, scattering_volumes=False,
    multiple_scattering_volumes=False, volume_empty_space_skipping=False,
    native_textures=False, profiling=False, material_modifier=None,
):
    """Generate a wavefront specialization for material or surface programs."""
    from ..materials import MaterialEvaluation, SurfaceResponse

    programs = (programs,) if isinstance(programs, MaterialProgram) else tuple(programs)
    if not programs:
        raise ValueError("at least one material program is required")
    for program in programs:
        if not isinstance(program.evaluation, (MaterialEvaluation, SurfaceResponse)):
            raise ValueError(
                "staged custom materials require MaterialEvaluation or "
                "SurfaceResponse"
            )
        if isinstance(program.evaluation, MaterialEvaluation):
            expressions = tuple(vars(program.evaluation).values())
            if any(
                token in expression.code
                for expression in expressions
                for token in (
                    "random_u", "random_v", "current_ior", "exterior_ior"
                )
            ):
                raise ValueError(
                    "staged MaterialEvaluation programs must be deterministic "
                    "and independent of medium state"
                )
    slots = {
        name: attribute_layout.slot(name)
        for program in programs
        for name, _components in program.required_attributes
    }
    source = _expanded_shader_source(shader_name)
    if shader_name == "wavefront_primary.comp":
        source = source.replace(
            "#version 460\n",
            "#version 460\n#define WAVE_CUSTOM_MATERIAL_PROGRAM 1\n",
            1,
        )
    if native_textures:
        source = source.replace(
            "#version 460\n",
            "#version 460\n#define WAVE_NATIVE_TEXTURES 1\n",
            1,
        )
    if profiling:
        source = source.replace(
            "#version 460\n",
            "#version 460\n#define WAVE_WORK_COUNTERS 1\n",
            1,
        )
    if overlapping_volumes:
        source = source.replace(
            "#version 460\n",
            "#version 460\n#define WAVE_OVERLAPPING_VOLUMES 1\n",
            1,
        )
    if scattering_volumes:
        source = source.replace(
            "#version 460\n",
            "#version 460\n#define WAVE_VOLUME_SCATTERING 1\n",
            1,
        )
    if multiple_scattering_volumes:
        source = source.replace(
            "#version 460\n",
            "#version 460\n#define WAVE_VOLUME_MULTIPLE_SCATTERING 1\n",
            1,
        )
    if volume_empty_space_skipping:
        source = source.replace(
            "#version 460\n",
            "#version 460\n#define WAVE_VOLUME_EMPTY_SPACE_SKIPPING 1\n",
            1,
        )
    if shader_name == "wavefront_shade_candidate.glsl":
        begin = source.find(_BEGIN)
        end = source.find(_END)
        if begin < 0 or end < begin:
            raise RuntimeError(
                "Ordinary Shade production source has no material insertion "
                "point"
            )
        channel_count = len(attribute_layout.channels)
        attribute_support = f"""float waveFresnelSchlick(float cosine, float ior_from, float ior_to)
{{
    float ratio = (ior_from - ior_to) / max(ior_from + ior_to, 0.000001);
    float r0 = ratio * ratio;
    float one_minus_cosine = 1.0 - clamp(cosine, 0.0, 1.0);
    return r0 + (1.0 - r0) * one_minus_cosine * one_minus_cosine
        * one_minus_cosine * one_minus_cosine * one_minus_cosine;
}}
layout(set = 0, binding = {int(attribute_binding)}, std430) readonly buffer WaveCustomAttributeBuffer {{
    vec4 wave_custom_attributes[];
}};
uint wave_attribute_primitive;
vec3 wave_attribute_weights;
vec4 waveVertexAttribute4(uint slot)
{{
    uint base = wave_attribute_primitive * {3 * channel_count}u + slot;
    return wave_custom_attributes[base] * wave_attribute_weights.x
        + wave_custom_attributes[base + {channel_count}u] * wave_attribute_weights.y
        + wave_custom_attributes[base + {2 * channel_count}u] * wave_attribute_weights.z;
}}
float waveVertexAttribute1(uint slot) {{ return waveVertexAttribute4(slot).x; }}
vec2 waveVertexAttribute2(uint slot) {{ return waveVertexAttribute4(slot).xy; }}
vec3 waveVertexAttribute3(uint slot) {{ return waveVertexAttribute4(slot).xyz; }}
"""
        generated = (
            f"{_BEGIN}\n{attribute_support}"
            f"{material_dispatch_glsl(programs, attribute_slots=slots, material_modifier=material_modifier)}\n"
            f"{_END}"
        )
        source = source[:begin] + generated + source[end + len(_END):]
        call = "    MaterialEvaluation evaluated = evaluateMaterial("
        if call not in source:
            raise RuntimeError(
                "Ordinary Shade production material call site changed"
            )
        source = source.replace(
            call,
            "    wave_attribute_primitive = loaded.hit.primitive_index;\n"
            "    wave_attribute_weights = surface.weights;\n" + call,
            1,
        )
        return source
    anchor = "struct PointLightData"
    position = source.find(anchor)
    if position < 0:
        raise RuntimeError(f"shader {shader_name!r} has no material insertion anchor")
    channel_count = len(attribute_layout.channels)
    support = f"""struct MaterialEvaluation {{
    vec3 base_color; vec3 emission; float metallic; float roughness;
    float transmission; float ior; vec3 attenuation_color;
    float attenuation_distance; float custom_scattering; vec3 weight;
    vec3 next_direction; float event; float pdf;
}};
float waveFresnelSchlick(float cosine, float ior_from, float ior_to)
{{
    float ratio = (ior_from - ior_to) / max(ior_from + ior_to, 0.000001);
    float r0 = ratio * ratio;
    float one_minus_cosine = 1.0 - clamp(cosine, 0.0, 1.0);
    return r0 + (1.0 - r0) * one_minus_cosine * one_minus_cosine
        * one_minus_cosine * one_minus_cosine * one_minus_cosine;
}}
vec3 waveCosineHemisphere(vec3 normal, float random_u, float random_v)
{{
    float radius = sqrt(random_u);
    float phi = 6.28318530718 * random_v;
    vec3 tangent = normalize(abs(normal.z) < 0.999
        ? cross(normal, vec3(0.0, 0.0, 1.0))
        : cross(normal, vec3(0.0, 1.0, 0.0)));
    vec3 bitangent = cross(normal, tangent);
    return normalize(tangent * radius * cos(phi)
        + bitangent * radius * sin(phi)
        + normal * sqrt(max(0.0, 1.0 - random_u)));
}}
layout(set = 0, binding = {int(attribute_binding)}, std430) readonly buffer WaveCustomAttributeBuffer {{
    vec4 wave_custom_attributes[];
}};
uint wave_attribute_primitive;
vec3 wave_attribute_weights;
vec4 waveVertexAttribute4(uint slot)
{{
    uint base = wave_attribute_primitive * {3 * channel_count}u + slot;
    return wave_custom_attributes[base] * wave_attribute_weights.x
        + wave_custom_attributes[base + {channel_count}u] * wave_attribute_weights.y
        + wave_custom_attributes[base + {2 * channel_count}u] * wave_attribute_weights.z;
}}
float waveVertexAttribute1(uint slot) {{ return waveVertexAttribute4(slot).x; }}
vec2 waveVertexAttribute2(uint slot) {{ return waveVertexAttribute4(slot).xy; }}
vec3 waveVertexAttribute3(uint slot) {{ return waveVertexAttribute4(slot).xyz; }}
{material_dispatch_glsl(programs, attribute_slots=slots, material_modifier=material_modifier)}
MaterialEvaluation waveApplyMaterialProgram(
    inout MaterialData material, vec3 normal, vec2 uv, vec3 direction,
    bool entering, uint primitive, vec3 weights, float bounce_index)
{{
    wave_attribute_primitive = primitive;
    wave_attribute_weights = weights;
    MaterialEvaluation evaluated = evaluateMaterial(
        material, normal, uv, direction, entering, 0.5, 0.5,
        bounce_index, 1.0, material.ior_distance.x);
    material.base_roughness = vec4(evaluated.base_color, evaluated.roughness);
    material.emission_metallic = vec4(evaluated.emission, evaluated.metallic);
    material.attenuation_transmission = vec4(
        evaluated.attenuation_color, evaluated.transmission);
    material.ior_distance.xy = vec2(evaluated.ior, evaluated.attenuation_distance);
    return evaluated;
}}
"""
    source = source[:position] + support + source[position:]
    uv = """vec2 wave_material_uv =
            attributes[primitive * 3u + 0u].texcoord.xy * weights.x
            + attributes[primitive * 3u + 1u].texcoord.xy * weights.y
            + attributes[primitive * 3u + 2u].texcoord.xy * weights.z;"""
    if shader_name == "wavefront_primary.comp":
        first_anchor = "        // WAVE_MATERIAL_APPLICATION_SECONDARY\n"
        first_insert = (
            "        " + uv + "\n"
            "        MaterialEvaluation wave_surface_response = "
            "ordinarylight_apply_material_program("
            "material, normal, wave_material_uv, "
            "direction, entering, primitive, weights, float(bounce));\n"
            "        if (wave_surface_response.custom_scattering > 0.5) {\n"
            "            float wave_current_ior = "
            "stacks[path_index].ior[medium_depth - 1u];\n"
            "            float wave_exterior_ior = entering\n"
            "                ? max(material.ior_distance.x, 1.0001)\n"
            "                : (medium_depth > 1u\n"
            "                    ? stacks[path_index].ior[medium_depth - 2u] "
            ": 1.0);\n"
            "            float wave_random_u = randomFloat(rng);\n"
            "            float wave_random_v = randomFloat(rng);\n"
            "            wave_surface_response = evaluateMaterial(\n"
            "                material, normal, wave_material_uv, direction, "
            "entering, wave_random_u, wave_random_v, float(bounce),\n"
            "                wave_current_ior, wave_exterior_ior);\n"
            "        }\n"
        )
        second_anchor = (
            "#endif\n\n#if WAVE_ORDINARYSHADE_PRIMARY_SURFACE"
        )
        second_insert = (
            "#endif\n    " + uv.replace("            ", "    ") + "\n"
            "    MaterialEvaluation wave_surface_response = "
            "ordinarylight_apply_material_program("
            "material, normal, wave_material_uv, "
            "incoming, entering, primitive, weights, 0.0);\n"
            "    if (wave_surface_response.custom_scattering > 0.5) {\n"
            "        float wave_random_u = randomFloat(rng);\n"
            "        float wave_random_v = randomFloat(rng);\n"
            "        wave_surface_response = evaluateMaterial(\n"
            "            material, normal, wave_material_uv, incoming, "
            "entering, wave_random_u, wave_random_v, 0.0, 1.0,\n"
            "            max(material.ior_distance.x, 1.0001));\n"
            "    }\n\n"
            "#if WAVE_ORDINARYSHADE_PRIMARY_SURFACE"
        )
        if first_anchor not in source or second_anchor not in source:
            raise RuntimeError("primary shader material application anchors changed")
        source = source.replace(first_anchor, first_insert, 1)
        source = source.replace(second_anchor, second_insert, 1)
        scatter_anchor = "        if (transmission > 0.001) {"
        custom_loop = """        if (wave_surface_response.custom_scattering > 0.5) {
            int wave_event = int(wave_surface_response.event + 0.5);
            if (wave_event == 0) {
                path.metadata.w &= ~PATH_ACTIVE_BIT;
                break;
            }
            next_direction = normalize(wave_surface_response.next_direction);
            bsdf_pdf = max(wave_surface_response.pdf, 0.000001);
            path.throughput.rgb *= wave_surface_response.weight / bsdf_pdf;
            transmission = wave_event == 3 ? 1.0 : 0.0;
            if (wave_event == 3) {
                float target_ior = max(material.ior_distance.x, 1.0001);
                if (entering && medium_depth < WAVE_MAX_MEDIUM_STACK_DEPTH) {
                    stacks[path_index].ior[medium_depth] = target_ior;
                    medium_depth++;
                } else if (!entering && medium_depth > 1u) {
                    medium_depth--;
                }
            }
        } else if (transmission > 0.001) {"""
        if scatter_anchor not in source:
            raise RuntimeError("primary continuation scattering anchor changed")
        source = source.replace(scatter_anchor, custom_loop, 1)

        primary_scatter_anchor = "    if (transmission > 0.001) {"
        custom_primary = """    if (wave_surface_response.custom_scattering > 0.5) {
        int wave_event = int(wave_surface_response.event + 0.5);
        if (wave_event == 0) {
            path.metadata.w &= ~PATH_ACTIVE_BIT;
            setPathRng(path, rng);
            paths[path_index] = path;
            return;
        }
        next_direction = normalize(wave_surface_response.next_direction);
        bsdf_pdf = max(wave_surface_response.pdf, 0.000001);
        path.throughput.rgb *= wave_surface_response.weight / bsdf_pdf;
        transmission = wave_event == 3 ? 1.0 : 0.0;
        if (wave_event == 3 && entering) {
            stacks[path_index].ior[1] = max(material.ior_distance.x, 1.0001);
            medium_depth = 2u;
        }
    } else if (transmission > 0.001) {"""
        if primary_scatter_anchor not in source:
            raise RuntimeError("primary initial scattering anchor changed")
        source = source.replace(primary_scatter_anchor, custom_primary, 1)
    elif shader_name == "wavefront_shade.comp":
        shade_anchor = (
            "    if ((path.metadata.w & PATH_INDIRECT_CAPTURE_BIT) != 0u"
        )
        shade_insert = (
            "    " + uv.replace("            ", "    ") + "\n"
            "    MaterialEvaluation wave_surface_response = "
            "waveApplyMaterialProgram(material, normal, wave_material_uv, "
            "incoming, entering, primitive, weights, float(pathBounce(path)));\n"
            "    uint wave_surface_rng = pathRng(path);\n"
            "    if (wave_surface_response.custom_scattering > 0.5) {\n"
            "        uint wave_medium_depth = max(path.metadata.w >> 8u, 1u);\n"
            "        float wave_current_ior = "
            "stacks[path_index].ior[wave_medium_depth - 1u];\n"
            "        float wave_exterior_ior = entering\n"
            "            ? max(material.ior_distance.x, 1.0001)\n"
            "            : (wave_medium_depth > 1u\n"
            "                ? stacks[path_index].ior[wave_medium_depth - 2u] "
            ": 1.0);\n"
            "        float wave_random_u = randomFloat(wave_surface_rng);\n"
            "        float wave_random_v = randomFloat(wave_surface_rng);\n"
            "        wave_surface_response = evaluateMaterial(\n"
            "            material, normal, wave_material_uv, incoming, "
            "entering, wave_random_u, wave_random_v, "
            "float(pathBounce(path)), wave_current_ior, wave_exterior_ior);\n"
            "    }\n"
            + shade_anchor
        )
        if shade_anchor not in source:
            raise RuntimeError("shade shader material application anchor changed")
        source = source.replace(shade_anchor, shade_insert, 1)
        rng_anchor = "    uint rng = pathRng(path);"
        if rng_anchor not in source:
            raise RuntimeError("shade shader RNG anchor changed")
        source = source.replace(
            rng_anchor,
            "    uint rng = wave_surface_response.custom_scattering > 0.5\n"
            "        ? wave_surface_rng : pathRng(path);",
            1,
        )
        scatter_anchor = "    if (transmission > 0.001) {"
        custom_scatter = """    if (wave_surface_response.custom_scattering > 0.5) {
        int wave_event = int(wave_surface_response.event + 0.5);
        if (wave_event == 0) {
            path.metadata.w &= ~PATH_ACTIVE_BIT;
            setPathRng(path, rng);
            paths[path_index] = path;
            return;
        }
        next_direction = normalize(wave_surface_response.next_direction);
        bsdf_pdf = max(wave_surface_response.pdf, 0.000001);
        path.throughput.rgb *= wave_surface_response.weight / bsdf_pdf;
        transmission = wave_event == 3 ? 1.0 : 0.0;
        if (wave_event == 3) {
            float target_ior = max(material.ior_distance.x, 1.0001);
            if (entering && medium_depth < WAVE_MAX_MEDIUM_STACK_DEPTH) {
                stacks[path_index].ior[medium_depth] = target_ior;
                medium_depth++;
            } else if (!entering && medium_depth > 1u) {
                medium_depth--;
            }
        }
    } else if (transmission > 0.001) {"""
        if scatter_anchor not in source:
            raise RuntimeError("shade shader scattering anchor changed")
        source = source.replace(scatter_anchor, custom_scatter, 1)
    else:
        raise ValueError("staged material shader must be primary or shade")
    return source


def compile_wavefront_material_shader(
    shader_name, programs, *, attribute_layout, attribute_binding,
    overlapping_volumes=False, scattering_volumes=False,
    multiple_scattering_volumes=False, volume_empty_space_skipping=False,
    native_textures=False, profiling=False, material_modifier=None,
    compiler=None,
):
    compiler = compiler or find_glsl_compiler()
    if compiler is None:
        raise RuntimeError("custom materials require glslangValidator or glslc")
    return _compile_source(
        wavefront_material_shader_source(
            shader_name, programs, attribute_layout=attribute_layout,
            attribute_binding=attribute_binding,
            overlapping_volumes=overlapping_volumes,
            scattering_volumes=scattering_volumes,
            multiple_scattering_volumes=multiple_scattering_volumes,
            volume_empty_space_skipping=volume_empty_space_skipping,
            native_textures=native_textures,
            profiling=profiling,
            material_modifier=material_modifier,
        ),
        compiler,
    )


@lru_cache(maxsize=32)
def _compile_source(source, compiler):
    with tempfile.TemporaryDirectory(prefix="ordinarylight-shader-") as directory:
        directory = Path(directory)
        input_path = directory / "generated.comp"
        output_path = directory / "generated.spv"
        input_path.write_text(source)
        if Path(compiler).name == "glslc":
            command = [compiler, "--target-env=vulkan1.2", str(input_path), "-o", str(output_path)]
        else:
            command = [
                compiler, "-V", "--target-env", "vulkan1.2", "-S", "comp",
                str(input_path), "-o", str(output_path),
            ]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode:
            diagnostics = (result.stdout + result.stderr).strip()
            raise RuntimeError(f"Material shader compilation failed:\n{diagnostics}")
        return output_path.read_bytes()


def compile_material_shader(
    shader_name, program, compiler=None, *, attribute_layout=None,
    material_modifier=None,
):
    """Generate and compile a complete shader for ``program``."""
    compiler = compiler or find_glsl_compiler()
    if compiler is None:
        raise RuntimeError(
            "Custom materials require glslangValidator or glslc on PATH; "
            "install glslang-tools or the Vulkan SDK"
        )
    return _compile_source(
        material_shader_source(
            shader_name, program, attribute_layout=attribute_layout,
            material_modifier=material_modifier,
        ),
        compiler,
    )
