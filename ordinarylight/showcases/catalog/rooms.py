"""Core room and lighting showcases for the Ordinary Light workbench."""

from ordinarylight.integrations.workbench import OrbitCamera, Showcase

from ordinarylight.showcases.area_lights import build_area_light_showcase
from ordinarylight.showcases.rooms import (
    build_dense_geometry,
    build_diffuse_room,
    build_glossy_glass,
    build_nested_glass,
    build_occlusion_room,
    build_small_emitter,
    build_textured_room,
)


ROOM_CAMERA = OrbitCamera(
    target=(0.0, 1.25, 0.0), radius=-8.5, height=3.2, arc_radians=0.48,
)

SHOWCASES = (
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
)
