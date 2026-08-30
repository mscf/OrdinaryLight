"""Ordinary Shade ABI shared by raster and global-illumination materials."""

from __future__ import annotations

import ordinaryshade as osh


@osh.structure
class SurfaceContext:
    uv: osh.vec2
    normal: osh.vec3
    view_direction: osh.vec3
    program_id: osh.f32


@osh.structure
class SurfaceParameters:
    base_color: osh.vec3
    emission: osh.vec3
    normal: osh.vec3
    metallic: osh.f32
    roughness: osh.f32
    transmission: osh.f32
    occlusion: osh.f32
    clearcoat: osh.f32
    clearcoat_roughness: osh.f32
    sheen_color: osh.vec3
    sheen_roughness: osh.f32
    anisotropy: osh.f32
    thin_walled: osh.f32
    subsurface: osh.f32
    subsurface_color: osh.vec3
    subsurface_radius: osh.f32


@osh.function(name="blend_surface_parameters")
def blend_surface_parameters(
    base: SurfaceParameters, layer: SurfaceParameters, weight: osh.f32,
) -> SurfaceParameters:
    amount = osh.maximum(0.0, osh.minimum(1.0, weight))
    return SurfaceParameters(
        osh.mix(base.base_color, layer.base_color, amount),
        base.emission + layer.emission * amount,
        osh.normalize(osh.mix(base.normal, layer.normal, amount)),
        osh.mix(base.metallic, layer.metallic, amount),
        osh.mix(base.roughness, layer.roughness, amount),
        osh.mix(base.transmission, layer.transmission, amount),
        osh.mix(base.occlusion, layer.occlusion, amount),
        osh.mix(base.clearcoat, layer.clearcoat, amount),
        osh.mix(base.clearcoat_roughness, layer.clearcoat_roughness, amount),
        osh.mix(base.sheen_color, layer.sheen_color, amount),
        osh.mix(base.sheen_roughness, layer.sheen_roughness, amount),
        osh.mix(base.anisotropy, layer.anisotropy, amount),
        osh.mix(base.thin_walled, layer.thin_walled, amount),
        osh.mix(base.subsurface, layer.subsurface, amount),
        osh.mix(base.subsurface_color, layer.subsurface_color, amount),
        osh.mix(base.subsurface_radius, layer.subsurface_radius, amount),
    )


@osh.function(name="ordinarylight_material_modifier")
def default_material_modifier(
    surface: SurfaceParameters, context: SurfaceContext,
) -> SurfaceParameters:
    return surface


def material_modifier(function):
    """Declare a portable pre-lighting/pre-BSDF surface modifier."""
    return osh.function(name="ordinarylight_material_modifier")(function)


def modifier_signature(modifier):
    if modifier is None:
        return None
    if not (
        hasattr(modifier, "function")
        and getattr(modifier, "__name__", None)
        == "ordinarylight_material_modifier"
    ):
        raise TypeError("material_modifier must be created by @material_modifier")
    import inspect
    return (
        modifier.function.__module__, modifier.function.__qualname__,
        inspect.getsource(modifier.function),
    )


def _blend_surface_parameters_abi(
    base: SurfaceParameters, layer: SurfaceParameters, weight: osh.f32,
) -> SurfaceParameters:
    raise NotImplementedError


_blend_surface_parameters_abi.__name__ = "blend_surface_parameters"
_blend_surface_parameters_external = osh.external(_blend_surface_parameters_abi)


def material_modifier_glsl(modifier=None):
    """Compile the portable surface ABI and modifier to reusable GLSL."""
    modifier = modifier or default_material_modifier
    modifier_signature(modifier)
    structures = """struct SurfaceContext {
    vec2 uv;
    vec3 normal;
    vec3 view_direction;
    float program_id;
};
struct SurfaceParameters {
    vec3 base_color;
    vec3 emission;
    vec3 normal;
    float metallic;
    float roughness;
    float transmission;
    float occlusion;
    float clearcoat;
    float clearcoat_roughness;
    vec3 sheen_color;
    float sheen_roughness;
    float anisotropy;
    float thin_walled;
    float subsurface;
    vec3 subsurface_color;
    float subsurface_radius;
};
"""
    blend = osh.compile_function(
        blend_surface_parameters, target="glsl"
    ).source
    compiled_modifier = osh.compile_function(
        modifier, target="glsl",
        externals=(_blend_surface_parameters_external,),
    ).source
    declaration = (
        "SurfaceParameters blend_surface_parameters(SurfaceParameters base, "
        "SurfaceParameters layer, float weight);\n"
    )
    if compiled_modifier.startswith(declaration):
        compiled_modifier = compiled_modifier[len(declaration):]
    return structures + blend + compiled_modifier


__all__ = [
    "SurfaceContext", "SurfaceParameters", "blend_surface_parameters",
    "default_material_modifier", "material_modifier", "modifier_signature",
    "material_modifier_glsl",
]
