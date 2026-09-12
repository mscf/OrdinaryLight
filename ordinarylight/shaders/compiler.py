"""Compilation of generated material programs into Vulkan SPIR-V shaders."""

from functools import lru_cache
from importlib.resources import files
from pathlib import Path
import hashlib
import shutil
import subprocess
import tempfile

from ..materials import MaterialProgram, material_dispatch_glsl
from .material_support import support_source
from ordinarylight.runtime.lifecycle import timed_call


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
    shader_name, program, *, attribute_layout=None, material_modifier=None, material_resources=None,
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
        f"{material_dispatch_glsl(programs, attribute_slots=slots, material_modifier=material_modifier, material_resources=material_resources)}\n"
        f"{_END}"
    )
    if required:
        support = support_source(len(attribute_layout.channels), 15, staged=False)
        generated = support + generated
        source = source.replace("#version 460\n", "#version 460\n#define WAVE_CUSTOM_ATTRIBUTES 1\n", 1)
        begin = source.find(_BEGIN)
        end = source.find(_END)
    result = source[:begin] + generated + source[end + len(_END):]
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
    native_textures=False, profiling=False, denoiser_signal_capture=False,
    inline_continuation=False,
    material_modifier=None, material_resources=None,
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
    if inline_continuation:
        if shader_name != "wavefront_primary.comp":
            raise ValueError("inline continuation requires the primary material shader")
        source = source.replace("#define WAVE_HYBRID 0", "#define WAVE_HYBRID 1", 1)
    if denoiser_signal_capture and shader_name == "wavefront_primary.comp":
        source = source.replace(
            "#version 460\n", "#version 460\n#define WAVE_DENOISER_SIGNAL_CAPTURE 1\n", 1,
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
    if denoiser_signal_capture and shader_name in ("wavefront_shade_candidate.glsl", "wavefront_shade.comp"):
        source = source.replace("#version 460\n", "#version 460\n#define WAVE_DENOISER_SIGNAL_CAPTURE 1\n", 1)
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
        source = source.replace("#version 460\n", "#version 460\n#define WAVE_CUSTOM_ATTRIBUTES 1\n", 1)
        begin, end = source.find(_BEGIN), source.find(_END)
        if begin < 0 or end < begin:
            raise RuntimeError("OrdinaryShade production source has no material insertion point")
        support = support_source(len(attribute_layout.channels), attribute_binding, candidate=True)
        generated = (f"{_BEGIN}\n{support}"
                     f"{material_dispatch_glsl(programs, attribute_slots=slots, material_modifier=material_modifier, material_resources=material_resources)}\n{_END}")
        return source[:begin] + generated + source[end + len(_END):]
    if shader_name in ("wavefront_primary.comp", "wavefront_shade.comp"):
        source = source.replace("#version 460\n", "#version 460\n#define WAVE_CUSTOM_MATERIAL_PROGRAM 1\n", 1)
        position = source.find("struct PointLightData")
        if position < 0:
            raise RuntimeError("Primary shader has no material module insertion point")
        support = support_source(len(attribute_layout.channels), attribute_binding)
        support += material_dispatch_glsl(programs, attribute_slots=slots,
                                         material_modifier=material_modifier, material_resources=material_resources)
        return source[:position] + support + source[position:]
    raise ValueError("staged material shader must be primary or shade")


def compile_wavefront_material_shader(
    shader_name, programs, *, attribute_layout, attribute_binding,
    overlapping_volumes=False, scattering_volumes=False,
    multiple_scattering_volumes=False, volume_empty_space_skipping=False,
    native_textures=False, profiling=False, denoiser_signal_capture=False,
    inline_continuation=False,
    material_modifier=None, material_resources=None,
    compiler=None, shared_primary_reservoirs=0, surface_only=False, opaque_primary=False,
    production_restir=False, camera_restir_policy=False, primary_hits=False,
    geometry_program=None,
):
    compiler = compiler or find_glsl_compiler()
    if compiler is None:
        raise RuntimeError("custom materials require glslangValidator or glslc")
    source = wavefront_material_shader_source(
            shader_name, programs, attribute_layout=attribute_layout,
            attribute_binding=attribute_binding,
            overlapping_volumes=overlapping_volumes,
            scattering_volumes=scattering_volumes,
            multiple_scattering_volumes=multiple_scattering_volumes,
            volume_empty_space_skipping=volume_empty_space_skipping,
            native_textures=native_textures,
            profiling=profiling,
            denoiser_signal_capture=denoiser_signal_capture,
            inline_continuation=inline_continuation,
            material_modifier=material_modifier, material_resources=material_resources,
        )
    if geometry_program is not None:
        from ..geometry.native import NativeGeometryProgram
        if not isinstance(geometry_program, NativeGeometryProgram):
            raise TypeError("geometry_program must be NativeGeometryProgram")
        source = source.replace("#version 460\n", "#version 460\n#define WAVE_CUSTOM_GEOMETRY 1\n", 1)
        if geometry_program.boundary is not None:
            source = source.replace("#version 460\n", "#version 460\n#define WAVE_NATIVE_OPTICAL_BOUNDARIES 1\n", 1)
        if geometry_program.emitters is not None:
            source = source.replace("#version 460\n", "#version 460\n#define WAVE_NATIVE_EMITTERS 1\n", 1)
        source += "\n" + geometry_program.source
    if primary_hits:
        if shader_name != "wavefront_primary.comp":
            raise ValueError("primary hit outputs require a primary shader")
        source = source.replace("#version 460\n", "#version 460\n#define WAVE_PRIMARY_HITS 1\n", 1)
    if surface_only:
        if shader_name not in ("wavefront_shade_candidate.glsl", "wavefront_primary.comp") or any((
            overlapping_volumes, scattering_volumes,
            multiple_scattering_volumes, volume_empty_space_skipping,
        )):
            raise ValueError("surface-only specialization requires volume-free Ordinary Shade shading")
        source = source.replace(
            "#version 460\n",
            "#version 460\n#define WAVE_SURFACE_ONLY 1\n",
            1,
        )
    if camera_restir_policy:
        if shader_name != "wavefront_primary.comp":
            raise ValueError("camera ReSTIR policy requires a primary shader")
        source = source.replace(
            "#version 460\n", "#version 460\n#define WAVE_CAMERA_RESTIR_POLICY 1\n", 1,
        )
    if production_restir:
        if shader_name != "wavefront_primary.comp":
            raise ValueError("production ReSTIR specialization requires a primary shader")
        source = source.replace(
            "#version 460\n", "#version 460\n#define WAVE_PRODUCTION_RESTIR 1\n", 1,
        )
    if opaque_primary:
        if shader_name != "wavefront_primary.comp" or not surface_only:
            raise ValueError("opaque primary specialization requires a surface-only primary shader")
        source = source.replace(
            "#version 460\n", "#version 460\n#define WAVE_OPAQUE_SCENE 1\n", 1,
        )
    if shared_primary_reservoirs:
        if shader_name != "wavefront_primary.comp" or not 1 <= shared_primary_reservoirs <= 8:
            raise ValueError("shared primary reservoirs require primary shader and count 1..8")
        source = source.replace(
            "#version 460\n",
            f"#version 460\n#define WAVE_SHARED_PRIMARY_RESERVOIRS {shared_primary_reservoirs}\n",
            1,
        )
    return _compile_source(source, compiler)


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
        result = timed_call(
            "shader_compile", subprocess.run, command,
            capture_output=True, text=True,
            lifecycle_details={
                "compiler": Path(compiler).name,
                "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
            },
        )
        if result.returncode:
            diagnostics = (result.stdout + result.stderr).strip()
            raise RuntimeError(f"Material shader compilation failed:\n{diagnostics}")
        return output_path.read_bytes()


def compile_material_shader(
    shader_name, program, compiler=None, *, attribute_layout=None,
    material_modifier=None, material_resources=None,
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
            material_modifier=material_modifier, material_resources=material_resources,
        ),
        compiler,
    )
