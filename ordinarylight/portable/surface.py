"""Packaged portable scalar-mesh renderer assets, independent of plotting UI."""
from pathlib import Path


def surface_assets():
    """Return generated OrdinaryShade WGSL and the browser mesh runtime source."""
    root = Path(__file__).parent
    return {"shader": (root / "surface.wgsl").read_text(),
            "runtime": (root / "web/surface-runtime.js").read_text(),
            "drag": (root / "web/drag-controller.js").read_text()}


def line_assets():
    """Generic WebGPU screen-space strokes, reusing the generated scalar shader."""
    assets=surface_assets()
    assets["lines"]=(Path(__file__).parent / "web/line-runtime.js").read_text()
    return assets
