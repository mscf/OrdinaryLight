"""Compatibility imports for the former :mod:`ordinarylight.gltf` module.

New code should import from :mod:`ordinarylight.loaders` or call
``ordinarylight.loaders.gltf.load()``.
"""

from ..loaders.gltf import load, load_gltf

__all__ = ["load", "load_gltf"]
