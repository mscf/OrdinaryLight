"""Perform lightweight structural validation of an Ordinary Light wheel."""

from __future__ import annotations

import sys
from pathlib import Path
from zipfile import ZipFile


REQUIRED_SUFFIXES = (
    "ordinarylight/__init__.py",
    "ordinarylight/renderer.py",
    "ordinarylight/backends/base.py",
    "ordinarylight/backends/vulkan.py",
    "ordinarylight/loaders/gltf.py",
    "ordinarylight/outputs/image.py",
    "ordinarylight/integrations/qt_workbench.py",
    "ordinarylight/showcases/catalog/volumes.py",
    "ordinarylight/shaders/wavefront_primary.comp",
    "ordinarylight/shaders/wavefront_primary.comp.spv",
    "ordinarylight/shaders/manifest.json",
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
    print(f"Verified {wheel}: {len(names)} packaged files")


def main(argv=None):
    paths = sys.argv[1:] if argv is None else list(argv)
    if not paths:
        raise SystemExit("usage: verify_wheel.py WHEEL [WHEEL ...]")
    for path in paths:
        verify(path)


if __name__ == "__main__":
    main()
