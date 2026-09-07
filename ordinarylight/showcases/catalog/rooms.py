"""Core room and lighting showcases for the Ordinary Light workbench."""

from ordinarylight.integrations.workbench import OrbitCamera, Showcase

from ordinarylight.showcases.area_lights import build_area_light_showcase
from ordinarylight.showcases.rooms import (
    animate_object_motion_room,
    build_dense_geometry,
    build_diffuse_room,
    build_glossy_glass,
    build_planar_mirror_guides,
    build_nested_glass,
    build_occlusion_room,
    build_object_motion_room,
    build_small_emitter,
    build_textured_room,
)


from ordinarylight.showcases.glass_detail import (
    animate_glass_detail, build_glass_detail, build_moving_glass_detail,
    build_moving_glass_target,
)

GLASS_DETAIL_CAMERA = OrbitCamera(target=(0, 1, 0), radius=-5, height=1, arc_radians=0.3)

ROOM_CAMERA = OrbitCamera(
    target=(0.0, 1.25, 0.0), radius=-8.5, height=3.2, arc_radians=0.48,
)

SHOWCASES = (
    Showcase("glass-detail-camera", "Glass detail: camera motion", build_glass_detail,
             description="Orbit the glass and striped target. Disable Animate to inspect settling.",
             camera=GLASS_DETAIL_CAMERA, renderer={"denoiser_enabled": True},
             tags=("raster-feature", "denoising", "glass", "motion")),
    Showcase("glass-detail-motion", "Glass detail: moving sphere", build_moving_glass_detail,
             description="The sphere moves for one second, then holds for two. Camera and target stay fixed.",
             camera=GLASS_DETAIL_CAMERA, renderer={"denoiser_enabled": True},
             tags=("raster-feature", "denoising", "glass", "motion"),
             animate=animate_glass_detail),
    Showcase("glass-detail-target", "Glass detail: moving target", build_moving_glass_target,
             description="The striped target moves for one second, then holds for two. Glass and camera stay fixed.",
             camera=GLASS_DETAIL_CAMERA, renderer={"denoiser_enabled": True},
             tags=("raster-feature", "denoising", "glass", "motion"),
             animate=animate_glass_detail),
    Showcase("planar-mirror-guides", "Planar mirror guides", build_planar_mirror_guides,
             camera=OrbitCamera(target=(0, 1.8, 0), radius=-7, height=2, arc_radians=0.3),
             renderer={"denoiser_enabled": True},
             tags=("raster-feature", "denoising", "mirror")),

    Showcase("area-lights", "Area lights", build_area_light_showcase,
             camera=ROOM_CAMERA, tags=("lighting", "baseline")),
    Showcase("diffuse-room", "Diffuse room", build_diffuse_room,
             camera=ROOM_CAMERA, tags=("lighting", "diffuse")),
    Showcase("glossy-glass", "Glossy + glass", build_glossy_glass,
             camera=ROOM_CAMERA, tags=("materials", "glass", "stress")),
    Showcase("textured-room", "Textured room", build_textured_room,
             camera=ROOM_CAMERA, tags=("textures",)),
    Showcase("small-emitter", "Small emitter", build_small_emitter,
             camera=ROOM_CAMERA, tags=("lighting", "stress")),
    Showcase("occlusion-room", "Occlusion room", build_occlusion_room,
             camera=ROOM_CAMERA, tags=("lighting", "visibility")),
    Showcase("nested-glass", "Nested glass", build_nested_glass,
             camera=ROOM_CAMERA, tags=("materials", "glass")),
    Showcase("dense-geometry", "Dense geometry", build_dense_geometry,
             camera=ROOM_CAMERA, tags=("geometry", "stress")),
    Showcase(
        "object-motion-relax", "ReLAX object motion",
        build_object_motion_room,
        description=(
            "A moving foreground sphere and stationary background sphere "
            "exercise rigid-object motion reprojection and disocclusion."
        ),
        camera=ROOM_CAMERA,
        renderer={"denoiser_enabled": True},
        tags=("raster-feature", "denoising", "animation", "motion"),
        animate=animate_object_motion_room,
    ),
)
