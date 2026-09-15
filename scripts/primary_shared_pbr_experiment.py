"""Diagnostic OrdinaryShade shared PDF/evaluation math for primary continuation."""
from pathlib import Path
from primary_hit_layout_experiment import load_module


def variant(directory):
    from ordinarylight.shaders import fused_primary_programs, lighting_programs
    source = Path(fused_primary_programs.__file__).read_text()
    lighting = Path(lighting_programs.__file__).read_text()
    start = lighting.index('@osh.function\ndef samplePbr(')
    end = lighting.index('\n@osh.function', start + 1)
    sample = lighting[start:end]
    begin = sample.index('    pdf = osh.maximum(ordinarylight_pbr_pdf(')
    finish = sample.index('    transportLastSpecularFraction', begin)
    shared = '''    normal_view = osh.maximum(osh.dot(normal, view), 0.0)
    normal_light = osh.maximum(osh.dot(normal, outgoing), 0.0)
    half_vector = osh.normalize(view + outgoing)
    normal_half = osh.maximum(osh.dot(normal, half_vector), 0.0)
    view_half = osh.maximum(osh.dot(view, half_vector), 0.0)
    f0 = osh.mix(osh.vec3(0.04), material.base_roughness.rgb, material.emission_metallic.a)
    fresnel = f0 + (osh.vec3(1.0) - f0) * osh.power(1.0 - osh.clamp(view_half, 0.0, 1.0), 5.0)
    alpha = osh.maximum(material.base_roughness.a * material.base_roughness.a, 0.0009)
    alpha_squared = alpha * alpha
    denominator = normal_half * normal_half * (alpha_squared - 1.0) + 1.0
    distribution = alpha_squared / osh.maximum(3.14159265359 * denominator * denominator, 0.000001)
    pdf = 0.0
    if normal_light > 0.0:
        specular_pdf = distribution * normal_half / (4.0 * osh.maximum(osh.dot(view, half_vector), 0.000001))
        diffuse_pdf = normal_light / 3.14159265359
        pdf = osh.mix(diffuse_pdf, specular_pdf, probability)
    pdf = osh.maximum(pdf, 0.000001)
    diffuse = (osh.vec3(1.0) - fresnel) * (1.0 - material.emission_metallic.a) * material.base_roughness.rgb / 3.14159265359
    evaluated = osh.vec3(0.0)
    if normal_view > 0.0 and normal_light > 0.0:
        view_geometry = 2.0 * normal_view / osh.maximum(normal_view + osh.sqrt(alpha_squared + (1.0 - alpha_squared) * normal_view * normal_view), 0.000001)
        light_geometry = 2.0 * normal_light / osh.maximum(normal_light + osh.sqrt(alpha_squared + (1.0 - alpha_squared) * normal_light * normal_light), 0.000001)
        geometry = view_geometry * light_geometry
        specular_value = fresnel * distribution * geometry / osh.maximum(4.0 * normal_view * normal_light, 0.000001)
        evaluated = diffuse + specular_value
'''
    sample = sample[:begin] + shared + sample[finish:]
    return load_module('ordinarylight.shaders.primary_shared_pbr_diagnostic',
                       source + '\n' + sample, directory)
