"""Lower material expression graphs and dispatch through OrdinaryShade."""
import ordinaryshade as osh
from ..shaders.dynamic import compile_typed, external_signature
from ..shaders.transport_programs import MaterialData, MaterialEvaluation
from ..shaders import material_dispatch_programs as dispatch
from .gpu import SurfaceParameters, SurfaceContext, default_material_modifier

PARAMETERS = ('material: MaterialData, normal: osh.vec3, uv: osh.vec2, direction: osh.vec3, '
              'entering: osh.boolean, random_u: osh.f32, random_v: osh.f32, '
              'bounce_index: osh.f32, current_ior: osh.f32, exterior_ior: osh.f32')
ARGUMENTS = 'material, normal, uv, direction, entering, random_u, random_v, bounce_index, current_ior, exterior_ior'
NAMESPACE = dict(MaterialData=MaterialData, MaterialEvaluation=MaterialEvaluation)


def material_signature(name):
    return external_signature(name, PARAMETERS, 'MaterialEvaluation', NAMESPACE)


def compile_material(program, name, *, attribute_slots=None):
    from ._core import MaterialEvaluation as Evaluation, LayeredMaterialEvaluation, MaterialContext
    if not name.isidentifier() or not name.isascii():
        raise ValueError('function_name must be a valid shader identifier')
    evaluation = program.evaluation.resolved if isinstance(program.evaluation, LayeredMaterialEvaluation) else program.evaluation
    base = MaterialContext.shader_inputs()
    expressions = {}
    for field in MaterialEvaluation.fields:
        key = field.name
        if hasattr(evaluation, key):
            expression = getattr(evaluation, key)
            expected = 'float' if field.type == osh.f32 else field.type.name
            if expression.type != expected:
                raise TypeError(f'{key} must be a {expected} expression')
            expressions[key] = expression.python
        elif hasattr(base, key):
            expressions[key] = getattr(base, key).python
    expressions['custom_scattering'] = '0.0' if isinstance(evaluation, Evaluation) else '1.0'
    expressions.setdefault('weight', 'osh.vec3(0.0)')
    expressions.setdefault('next_direction', 'direction')
    expressions.setdefault('event', '0.0')
    expressions.setdefault('pdf', '1.0')
    body = f'def {name}({PARAMETERS}) -> MaterialEvaluation:\n'
    zero = ', '.join('0.0' if field.type == osh.f32 else 'osh.vec3(0.0)' for field in MaterialEvaluation.fields)
    body += f'    result = MaterialEvaluation({zero})\n'
    body += ''.join(f'    result.{field.name} = {expressions[field.name]}\n' for field in MaterialEvaluation.fields)
    body += '    return result\n'
    externals = [external_signature('waveFresnelSchlick', 'cosine: osh.f32, ior_from: osh.f32, ior_to: osh.f32', 'osh.f32'),
                 external_signature('waveCosineHemisphere', 'normal: osh.vec3, random_u: osh.f32, random_v: osh.f32', 'osh.vec3')]
    values = {}
    for attribute, components in program.required_attributes:
        macro = f'WAVE_ATTRIBUTE_{attribute}'
        if attribute_slots is not None:
            if attribute not in attribute_slots:
                raise ValueError(f'no shader slot was supplied for attribute {attribute!r}')
            slot = int(attribute_slots[attribute])
            if slot < 0:
                raise ValueError('attribute slots cannot be negative')
            # Identifier substitution happens in the typed Python AST, before lowering.
            import ast
            class BindAttribute(ast.NodeTransformer):
                def visit_Name(self, node):
                    return ast.parse(f'osh.u32({slot})', mode='eval').body if node.id == macro else node
            body = ast.unparse(BindAttribute().visit(ast.parse(body))) + '\n'
        else:
            values[macro] = osh.u32
    for components in range(1, 5):
        externals.append(external_signature(f'waveVertexAttribute{components}', 'channel: osh.u32',
                                           'osh.f32' if components == 1 else f'osh.vec{components}'))
    for resource in program.resources:
        parameters = {'uniform': '', 'buffer': 'index: osh.f32', 'texture': 'uv: osh.vec2'}[resource.kind]
        externals.append(external_signature(f'ol_graph_{resource.name}', parameters, 'osh.vec4'))
    return compile_typed(body, name, namespace=NAMESPACE, externals=externals, values=values)


def compile_dispatch(count):
    source = f'def selectMaterial({PARAMETERS}) -> MaterialEvaluation:\n'
    source += '    program_id = osh.i32(osh.floor(material.ior_distance.z))\n'
    source += f'    evaluated = evaluateMaterial_0({ARGUMENTS})\n'
    for index in range(1, count):
        source += f'    if program_id == {index}:\n        evaluated = evaluateMaterial_{index}({ARGUMENTS})\n'
    source += '    return evaluated\n'
    selected = compile_typed(source, 'selectMaterial', namespace=NAMESPACE,
                             externals=[material_signature(f'evaluateMaterial_{index}') for index in range(count)])
    return selected + osh.compile_function(dispatch.evaluateMaterial,
        externals=(material_signature('selectMaterial'), osh.external(default_material_modifier.function)),
        external_values=dict(SurfaceParameters=SurfaceParameters, SurfaceContext=SurfaceContext)).source


def resource_accessor(prefix, kind):
    """Compile graph resource reads; descriptors are supplied by the host ABI."""
    from ordinaryshade.types import opaque_type
    if kind == 'uniform':
        @osh.structure
        class GraphUniform:
            value: osh.vec4
        body = f'def {prefix}() -> osh.vec4:\n    return {prefix}_data.value\n'
        values = {prefix + '_data': GraphUniform}
    elif kind == 'buffer':
        @osh.structure
        class GraphBuffer:
            values: osh.runtime_array(osh.vec4)
        body = f'''def {prefix}(index: osh.f32) -> osh.vec4:
    if osh.is_nan(index) or osh.is_inf(index) or index < 0.0 or index >= osh.f32(osh.array_length({prefix}_data.values)):
        return osh.vec4(0)
    return {prefix}_data.values[osh.u32(index)]
'''
        values = {prefix + '_data': GraphBuffer}
    else:
        body = f'def {prefix}(uv: osh.vec2) -> osh.vec4:\n    return {prefix}_image.sample_level_with({prefix}_sampler, uv, 0.0)\n'
        values = {prefix + '_image': opaque_type('sampled_texture_2d'), prefix + '_sampler': opaque_type('sampler')}
    return compile_typed(body, prefix, values=values)
