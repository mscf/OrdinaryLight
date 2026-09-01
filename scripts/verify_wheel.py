"""Perform lightweight structural validation of an Ordinary Light wheel."""

from __future__ import annotations

import sys
from pathlib import Path
from zipfile import ZipFile


REQUIRED_SUFFIXES = (
    "ordinarylight/__init__.py",
    "ordinarylight/renderer/__init__.py",
    "ordinarylight/renderer/renderer.py",
    "ordinarylight/cameras/perspective_camera.py",
    "ordinarylight/cameras/orthographic_camera.py",
    "ordinarylight/scene/scene.py",
    "ordinarylight/renderers/base.py",
    "ordinarylight/renderers/gi/vulkan.py",
    "ordinarylight/renderers/raster/vulkan.py",
    "ordinarylight/renderers/raster/webgpu.py",
    "ordinarylight/targets/vulkan/__init__.py",
    "ordinarylight/targets/vulkan/api.py",
    "ordinarylight/targets/vulkan/core.py",
    "ordinarylight/targets/webgpu.py",
    "ordinarylight/loaders/gltf.py",
    "ordinarylight/outputs/image.py",
    "ordinarylight/outputs/nvenc.py",
    "ordinarylight/effects/__init__.py",
    "ordinarylight/selection/__init__.py",
    "ordinarylight/integrations/qt_workbench.py",
    "ordinarylight/showcases/catalog/volumes.py",
    "ordinarylight/shaders/wavefront_primary.comp",
    "ordinarylight/shaders/wavefront_primary.comp.spv",
    "ordinarylight/shaders/rgba_to_nv12.comp.spv",
    "ordinarylight/shaders/hdr_to_p010.comp.spv",
    "ordinarylight/shaders/raster_scene.vert.spv",
    "ordinarylight/shaders/raster_scene.frag.spv",
    "ordinarylight/shaders/raster_scene.vert.wgsl",
    "ordinarylight/shaders/raster_scene.frag.wgsl",
    "ordinarylight/shaders/raster_geometry_products.vert.spv",
    "ordinarylight/shaders/raster_geometry_products.frag.spv",
    "ordinarylight/shaders/raster_geometry_products.vert.wgsl",
    "ordinarylight/shaders/raster_geometry_products.frag.wgsl",
    "ordinarylight/shaders/raster_scene.json",
    "ordinarylight/shaders/manifest.json",
)

FORBIDDEN_PREFIXES = (
    "ordinarylight/backends/",
)
FORBIDDEN_FILES = (
    "ordinarylight/backend_selection.py",
    "ordinarylight/renderer_selection.py",
    "ordinarylight/cameras.py",
    "ordinarylight/capabilities.py",
    "ordinarylight/effects.py",
    "ordinarylight/gpu.py",
    "ordinarylight/lights.py",
    "ordinarylight/materials.py",
    "ordinarylight/pipeline.py",
    "ordinarylight/primitives.py",
    "ordinarylight/raster.py",
    "ordinarylight/reference.py",
    "ordinarylight/renderer.py",
    "ordinarylight/scene.py",
    "ordinarylight/selection.py",
    "ordinarylight/shader_compiler.py",
    "ordinarylight/state.py",
    "ordinarylight/surface.py",
    "ordinarylight/validation.py",
    "ordinarylight/volume.py",
    "ordinarylight/vulkan.py",
    "ordinarylight/vulkan_rt.py",
    "ordinarylight/wavefront.py",
)


def verify(path):
    wheel = Path(path)
    if wheel.suffix != ".whl" or not wheel.is_file():
        raise ValueError(f"not a wheel: {wheel}")
    with ZipFile(wheel) as archive:
        names = set(archive.namelist())
    missing = [name for name in REQUIRED_SUFFIXES if name not in names]
    if missing:
        raise RuntimeError(
            f"{wheel.name} is missing required package data: {', '.join(missing)}"
        )
    obsolete = sorted(
        name for name in names
        if name in FORBIDDEN_FILES
        or any(name.startswith(prefix) for prefix in FORBIDDEN_PREFIXES)
    )
    if obsolete:
        raise RuntimeError(
            f"{wheel.name} contains obsolete renderer namespaces: "
            f"{', '.join(obsolete)}"
        )
    print(f"Verified {wheel}: {len(names)} packaged files")


def main(argv=None):
    paths = sys.argv[1:] if argv is None else list(argv)
    if not paths:
        raise SystemExit("usage: verify_wheel.py WHEEL [WHEEL ...]")
    for path in paths:
        verify(path)


if __name__ == "__main__":
    main()
