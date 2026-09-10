"""Typed transport helpers. Generated artifacts must be rebuilt from this source."""
import ordinaryshade as osh
from .transport_programs import MaterialData, MaterialEvaluation
from ..materials.gpu import SurfaceParameters, SurfaceContext


@osh.function
def evaluateMaterial(material: osh.inout(MaterialData), normal: osh.inout(osh.vec3), uv: osh.vec2, direction: osh.vec3, entering: osh.boolean, random_u: osh.f32, random_v: osh.f32, bounce_index: osh.f32, current_ior: osh.f32, exterior_ior: osh.f32) -> MaterialEvaluation:
    program_id = osh.i32(osh.floor(material.ior_distance.z))
    evaluated = selectMaterial(material, normal, uv, direction, entering, random_u, random_v, bounce_index, current_ior, exterior_ior)
    surface = SurfaceParameters(evaluated.base_color, evaluated.emission, normal, evaluated.metallic, evaluated.roughness, evaluated.transmission, 1.0, material.advanced0.x, material.advanced0.y, material.sheen_color.rgb, material.advanced0.z, material.advanced0.w, material.advanced1.z, material.advanced1.x, material.subsurface_color.rgb, material.advanced1.y)
    surface = ordinarylight_material_modifier(surface, SurfaceContext(uv, normal, (-direction), osh.f32(program_id)))
    normal = osh.normalize(surface.normal)
    material.advanced0 = osh.vec4(surface.clearcoat, surface.clearcoat_roughness, surface.sheen_roughness, surface.anisotropy)
    material.advanced1 = osh.vec4(surface.subsurface, surface.subsurface_radius, surface.thin_walled, 0.0)
    material.sheen_color = osh.vec4(surface.sheen_color, 0.0)
    material.subsurface_color = osh.vec4(surface.subsurface_color, 0.0)
    subsurface = osh.clamp(surface.subsurface, 0.0, 1.0)
    evaluated.base_color = osh.clamp((osh.mix(surface.base_color, surface.subsurface_color, (subsurface * 0.5)) + ((surface.sheen_color * ((1.0 - osh.clamp(surface.sheen_roughness, 0.0, 1.0)))) * 0.2)), osh.vec3(0.0), osh.vec3(1.0))
    evaluated.emission = osh.maximum(surface.emission, osh.vec3(0.0))
    evaluated.metallic = osh.clamp(surface.metallic, 0.0, 1.0)
    coat_mix = (osh.clamp(surface.clearcoat, 0.0, 1.0) * 0.25)
    anisotropic_roughness = (surface.roughness * ((1.0 - (0.25 * osh.absolute(osh.clamp(surface.anisotropy, -1.0, 1.0))))))
    evaluated.roughness = osh.clamp((osh.mix(anisotropic_roughness, surface.clearcoat_roughness, coat_mix) + ((subsurface * osh.clamp(surface.subsurface_radius, 0.0, 1.0)) * 0.1)), 0.001, 1.0)
    evaluated.transmission = osh.clamp(surface.transmission, 0.0, 1.0)
    if ((surface.thin_walled > 0.5)):
        evaluated.attenuation_distance = 1e30
    evaluated.emission = evaluated.emission * osh.clamp(surface.occlusion, 0.0, 1.0)
    return evaluated
