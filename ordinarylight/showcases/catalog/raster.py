"""Portable raster feature demonstrations."""

from ordinarylight.integrations.workbench import OrbitCamera, Showcase
from ordinarylight.showcases.raster_features import (
    build_advanced_material_scene, build_directional_shadow_scene,
    build_material_program_parity_scene, build_spot_shadow_scene,
)

from ordinarylight.showcases.raster_material_hooks import (
    advanced_surface_showcase_modifier,
)
from ordinarylight.showcases.advanced_materials import (
    build_anisotropy_scene, build_clearcoat_scene, build_sheen_scene,
    build_subsurface_scene, build_thin_transmission_scene,
)


_CAMERA = OrbitCamera(target=(0.0, 0.9, 0.0), radius=8.5, height=4.2)

SHOWCASES = (
    *tuple(Showcase(
        f"advanced-{slug}", f"Materials: {title}", builder,
        description=description,
        camera=OrbitCamera(target=(0.0, 1.0, 0.0), radius=9.5, height=3.3),
        renderer={"shadows": True, "shadow_map_size": 1024},
        tags=("raster-feature", "materials", "advanced", slug),
    ) for slug, title, builder, description in (
        ("clearcoat", "clearcoat", build_clearcoat_scene,
         "Energy-conserving dielectric coat with independent roughness."),
        ("sheen", "sheen", build_sheen_scene,
         "Grazing-angle cloth sheen color and roughness."),
        ("anisotropy", "anisotropy", build_anisotropy_scene,
         "Directional microfacet roughness on metallic surfaces."),
        ("thin-transmission", "thin transmission", build_thin_transmission_scene,
         "Solid and thin-walled transmission compared side by side."),
        ("subsurface", "subsurface", build_subsurface_scene,
         "Portable subsurface diffusion approximation and radius."),
    )),
    Showcase(
        "raster-directional-shadows", "Raster: directional shadows",
        build_directional_shadow_scene,
        description="Native directional shadow-map generation and sampling.",
        camera=_CAMERA,
        renderer={"shadows": True, "shadow_map_size": 512},
        tags=("raster-feature", "lighting", "shadows"),
    ),
    Showcase(
        "raster-spot-shadows", "Raster: spot shadows",
        build_spot_shadow_scene,
        description="Native perspective shadow map for a cone light.",
        camera=_CAMERA,
        renderer={"shadows": True, "shadow_map_size": 512},
        tags=("raster-feature", "lighting", "shadows"),
    ),
    Showcase(
        "raster-advanced-materials", "Raster: advanced material textures",
        build_advanced_material_scene,
        description=(
            "GPU base-color, metallic/roughness, emissive, tangent-space "
            "normal, occlusion, and transmission texture channels."
        ),
        camera=OrbitCamera(target=(0.0, 1.1, 0.0), radius=9.0, height=3.6),
        renderer={"shadows": False, "shadow_map_size": 1024},
        tags=("raster-feature", "materials", "textures", "normal-mapping"),
    ),
    Showcase(
        "raster-material-programs", "Raster/GI material program parity",
        build_material_program_parity_scene,
        description=(
            "Deterministic raster equivalents for Python-authored diffuse, "
            "mirror, Fresnel-glass, and unlit GI material programs."
        ),
        camera=OrbitCamera(target=(0.0, 1.0, 0.0), radius=10.0, height=3.4),
        renderer={"shadows": False, "shadow_map_size": 1024},
        tags=("raster-feature", "materials", "shaders", "parity"),
    ),
    Showcase(
        "portable-surface-modifier", "Raster/GI: portable surface modifier",
        build_advanced_material_scene,
        description=(
            "A portable Ordinary Shade modifier demonstrates clearcoat, "
            "sheen, anisotropy, and thin-walled transmission."
        ),
        camera=OrbitCamera(target=(0.0, 1.1, 0.0), radius=9.0, height=3.6),
        renderer={
            "shadows": False,
            "shadow_map_size": 1024,
            "material_modifier": advanced_surface_showcase_modifier,
        },
        tags=("raster-feature", "materials", "shaders", "layering"),
    ),
)
