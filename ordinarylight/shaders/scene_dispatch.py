"""Scene-dependent transport dispatch authored as typed OrdinaryShade Python."""
import ordinaryshade as osh
from .dynamic import compile_typed, external_signature
from .transport_programs import OrdinaryLightCustomHit, OrdinaryLightHit, MaterialData, MaterialEvaluation, TransportMaterialRecord
from ..materials.shade import material_signature

NAMESPACE = dict(OrdinaryLightCustomHit=OrdinaryLightCustomHit, OrdinaryLightHit=OrdinaryLightHit,
                 MaterialData=MaterialData, MaterialEvaluation=MaterialEvaluation)


def intersection_dispatch(programs):
    parameters = ('origin: osh.vec3, direction: osh.vec3, t_min: osh.f32, t_max: osh.f32, '
                  'parameters: osh.vec4, tolerance: osh.f32, max_steps: osh.u32, ')
    body = 'def ordinarylightCustomIntersect(program: osh.u32, ' + parameters + 'result: osh.inout(OrdinaryLightCustomHit)) -> osh.u32:\n'
    externals = []
    for index, program in enumerate(programs):
        output_signature = ('result: osh.inout(OrdinaryLightCustomHit)' if program.hit_version == 2 else
                            'distance: osh.out(osh.f32), normal: osh.out(osh.vec3)')
        externals.append(external_signature(program.name, parameters + output_signature, 'osh.u32', NAMESPACE))
        output_args = 'result' if program.hit_version == 2 else 'result.distance, result.geometric_normal'
        body += f'    if program == osh.u32({index}):\n        return {program.name}(origin, direction, t_min, t_max, parameters, tolerance, max_steps, {output_args})\n'
    body += '    return osh.u32(2)\n'
    return compile_typed(body, 'ordinarylightCustomIntersect', namespace=NAMESPACE, externals=externals)


def sampling_dispatch(programs):
    programs = tuple(programs)
    parameters = 'parameters: osh.vec4, randoms: osh.vec3, position: osh.out(osh.vec3), normal: osh.out(osh.vec3), area_pdf: osh.out(osh.f32)'
    pdf_parameters = 'parameters: osh.vec4, position: osh.vec3, normal: osh.vec3'
    body = 'def ordinarylightCustomSurfaceSample(program: osh.u32, ' + parameters + ') -> osh.u32:\n'
    pdf = 'def ordinarylightCustomSurfacePdf(program: osh.u32, ' + pdf_parameters + ') -> osh.f32:\n'
    externals, pdf_externals = [], []
    for index, program in enumerate(programs):
        if program.sampling is not None:
            name = program.sampling.name
            externals.append(external_signature(name, parameters, 'osh.u32'))
            pdf_externals.append(external_signature(name + '_pdf', pdf_parameters, 'osh.f32'))
            body += f'    if program == osh.u32({index}):\n        return {name}(parameters, randoms, position, normal, area_pdf)\n'
            pdf += f'    if program == osh.u32({index}):\n        return {name}_pdf(parameters, position, normal)\n'
    body += '    return osh.u32(0)\n'
    pdf += '    return 0.0\n'
    return compile_typed(body, 'ordinarylightCustomSurfaceSample', externals=externals) + compile_typed(pdf, 'ordinarylightCustomSurfacePdf', externals=pdf_externals)


def transport_material_dispatch(count):
    body = '''def ordinarylightEvaluateMaterial(index: osh.u32, hit: OrdinaryLightHit,
    direction: osh.vec3, bounce: osh.f32, current_ior: osh.f32, exterior_ior: osh.f32,
    randoms: osh.vec2) -> MaterialEvaluation:
    fixed_material = transport_materials[index]
    ior = optical_media[medium_boundaries[hit.boundary.x].y].a if hit.boundary.x != osh.u32(4294967295) else fixed_material.optics.z
    material = MaterialData(
        osh.vec4(fixed_material.albedo_kind.rgb, fixed_material.optics.x),
        osh.vec4(fixed_material.emission.rgb, fixed_material.optics.y),
        osh.vec4(1, 1, 1, osh.f32(fixed_material.albedo_kind.w == 1.0)),
        osh.vec4(ior, 1e30, 0, 0),
        osh.vec4(0), osh.vec4(0), osh.vec4(0), osh.vec4(0),
        osh.vec4(0), osh.vec4(0), osh.vec4(0), osh.vec4(0))
    entering = osh.dot(direction, hit.geometric_normal.xyz) < 0.0
    normal = hit.shading_normal.xyz if entering else -hit.shading_normal.xyz
    uv = osh.vec2(hit.geometric_normal.w, hit.shading_normal.w)
'''
    args = 'material, normal, uv, direction, entering, randoms.x, randoms.y, bounce, current_ior, exterior_ior'
    for index in range(1, count):
        body += f'    if index == osh.u32({index}):\n        return olMaterial_{index}({args})\n'
    body += f'    return olMaterial_0({args})\n'
    return compile_typed(body, 'ordinarylightEvaluateMaterial', namespace=NAMESPACE,
        externals=[material_signature(f'olMaterial_{index}') for index in range(count)],
        values=dict(transport_materials=osh.runtime_array(TransportMaterialRecord), optical_media=osh.runtime_array(osh.vec4), medium_boundaries=osh.runtime_array(osh.uvec4)))
