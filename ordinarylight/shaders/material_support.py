"""Link typed material support into a Vulkan stage's existing resource ABI."""
import ast
import inspect
import textwrap
import ordinaryshade as osh
from . import material_support_programs as p
from .sampling_programs import cosineHemisphere


def support_source(channel_count, binding, *, staged=True, candidate=False):
    helpers = [p.waveVertexAttribute4, p.waveVertexAttribute1,
               p.waveVertexAttribute2, p.waveVertexAttribute3, p.waveSetMaterialAttributes]
    if staged:
        helpers.insert(0, p.waveFresnelSchlick)
    if staged and not candidate:
        helpers.append(p.waveApplyMaterialProgram)
    values = dict(wave_attribute_primitive=osh.u32, wave_attribute_weights=osh.vec3,
                  wave_custom_attributes=osh.runtime_array(osh.vec4), WAVE_VERTEX_CHANNEL_COUNT=osh.u32)
    parts = [f'#define WAVE_VERTEX_CHANNEL_COUNT {int(channel_count)}u\n',
             f'layout(set = 0, binding = {int(binding)}, std430) readonly buffer WaveCustomAttributeBuffer {{ vec4 wave_custom_attributes[]; }};\n',
             'uint wave_attribute_primitive;\nvec3 wave_attribute_weights;\n']
    if staged and not candidate:
        parts.insert(0, '#ifndef ORDINARYLIGHT_GENERATED_TRANSPORT_MATERIAL_CONTRACTS\n#define ORDINARYLIGHT_GENERATED_TRANSPORT_MATERIAL_CONTRACTS 1\nstruct MaterialEvaluation {\n' + ''.join(
            f'    {field.type.name} {field.name};\n' for field in p.MaterialEvaluation.fields) + '};\n#endif\n')
    for helper in helpers:
        calls = {n.func.id for n in ast.walk(ast.parse(textwrap.dedent(inspect.getsource(helper.function))))
                 if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
        externals = tuple(osh.external(h.function) for h in helpers if h.function.__name__ in calls)
        if 'evaluateMaterial' in calls:
            externals += (p.evaluateMaterial,)
        parts.append(osh.compile_function(helper, external_values=values, externals=externals).source)
    if staged and not candidate:
        parts.append(osh.compile_function(cosineHemisphere).source.replace('cosineHemisphere', 'waveCosineHemisphere'))
    return '\n'.join(parts)
