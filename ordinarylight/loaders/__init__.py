"""Scene and asset loaders.

Loader implementations live in this namespace so adding a file format does
not expand the renderer's core modules.  Format modules expose a concise
``load()`` entry point; descriptive aliases remain available for code that
prefers an explicit function name.
"""

from . import gltf
from .gltf import load_gltf


_LOADERS = {
    ".gltf": gltf.load,
    ".glb": gltf.load,
}


def load(path, *, format=None):
    """Load a scene by explicit format or filename extension.

    Format-specific options belong on that format module's ``load`` function;
    this dispatcher intentionally provides only common discovery semantics.
    """
    from pathlib import Path

    key = str(format).lower().lstrip(".") if format is not None else None
    suffix = f".{key}" if key else Path(path).suffix.lower()
    try:
        loader = _LOADERS[suffix]
    except KeyError as exc:
        supported = ", ".join(supported_formats())
        raise ValueError(
            f"unsupported scene format {suffix or '<none>'!r}; "
            f"supported formats: {supported}"
        ) from exc
    return loader(path)


def supported_formats():
    """Return filename suffixes understood by :func:`load`."""
    return tuple(sorted(_LOADERS))


__all__ = ["gltf", "load", "load_gltf", "supported_formats"]
