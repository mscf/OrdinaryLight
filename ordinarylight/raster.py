"""Backend-neutral raster programs and draw data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class RasterProgram:
    """A linked vertex/fragment pair produced by Ordinary Shade."""

    vertex: Any
    fragment: Any
    reflection: Any

    @classmethod
    def compile(cls, vertex, fragment, *, target: str, validate: bool = True):
        try:
            import ordinaryshade as osh
        except ImportError as error:
            raise RuntimeError(
                "compiling Python raster shaders requires ordinaryshade"
            ) from error
        options = {"target": target, "validate": validate}
        if target == "spirv":
            from .shader_compiler import find_glsl_compiler
            compiler = find_glsl_compiler()
            if compiler is not None:
                options["spirv_compiler"] = compiler
        vertex_result = osh.compile(vertex, **options)
        fragment_result = osh.compile(fragment, **options)
        reflection = osh.link_graphics(vertex_result, fragment_result)
        return cls(vertex_result, fragment_result, reflection)

    @property
    def cache_key(self) -> str:
        return f"{self.vertex.cache_key}:{self.fragment.cache_key}"


@dataclass(frozen=True, slots=True)
class RasterMesh:
    """Minimal interleaved vertex data consumed by raster backends."""

    vertices: np.ndarray
    indices: np.ndarray | None = None

    def __post_init__(self):
        vertices = np.ascontiguousarray(self.vertices, dtype=np.float32)
        if vertices.ndim != 2 or vertices.shape[1] != 2:
            raise ValueError("minimal raster vertices must have shape (N, 2)")
        indices = self.indices
        if indices is not None:
            indices = np.ascontiguousarray(indices, dtype=np.uint32).reshape(-1)
        object.__setattr__(self, "vertices", vertices)
        object.__setattr__(self, "indices", indices)


def triangle_mesh() -> RasterMesh:
    return RasterMesh(np.array(((-0.7, -0.6), (0.7, -0.6), (0.0, 0.7)), np.float32))


__all__ = ["RasterMesh", "RasterProgram", "triangle_mesh"]
