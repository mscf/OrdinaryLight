"""Backend-neutral raster programs, pipeline state, and draw data."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np


_FORMATS = {
    "float32": (np.float32, 1), "float32x2": (np.float32, 2),
    "float32x3": (np.float32, 3), "float32x4": (np.float32, 4),
}


@dataclass(frozen=True, slots=True)
class RasterVertexAttribute:
    """One shader input in an interleaved vertex stream."""

    location: int
    format: str
    offset: int
    semantic: str = ""

    def __post_init__(self):
        if self.format not in _FORMATS:
            raise ValueError(f"unsupported raster vertex format {self.format!r}")
        if self.location < 0 or self.offset < 0:
            raise ValueError("vertex locations and offsets cannot be negative")


@dataclass(frozen=True, slots=True)
class RasterVertexLayout:
    """Portable interleaved vertex-buffer ABI."""

    stride: int
    attributes: tuple[RasterVertexAttribute, ...]

    def __post_init__(self):
        attributes = tuple(self.attributes)
        if self.stride < 1 or not attributes:
            raise ValueError("vertex layout requires a positive stride and attributes")
        if len({item.location for item in attributes}) != len(attributes):
            raise ValueError("vertex attribute locations must be unique")
        for item in attributes:
            _dtype, components = _FORMATS[item.format]
            if item.offset + components * 4 > self.stride:
                raise ValueError("vertex attribute exceeds the declared stride")
        object.__setattr__(self, "attributes", attributes)


@dataclass(frozen=True, slots=True)
class RasterState:
    """Backend-neutral fixed-function raster state."""

    topology: str = "triangle-list"
    cull_mode: str = "back"
    front_face: str = "ccw"
    depth_test: bool = True
    depth_write: bool = True
    depth_compare: str = "less"
    blend_mode: str = "opaque"

    def __post_init__(self):
        if self.topology not in {"triangle-list", "triangle-strip", "line-list"}:
            raise ValueError("unsupported raster topology")
        if self.cull_mode not in {"none", "front", "back"}:
            raise ValueError("cull_mode must be none, front, or back")
        if self.front_face not in {"cw", "ccw"}:
            raise ValueError("front_face must be cw or ccw")
        if self.depth_compare not in {"never", "less", "less-equal", "always"}:
            raise ValueError("unsupported depth comparison")
        if self.blend_mode not in {"opaque", "alpha", "additive"}:
            raise ValueError("blend_mode must be opaque, alpha, or additive")


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
            raise RuntimeError("compiling Python raster shaders requires ordinaryshade") from error
        options = {"target": target, "validate": validate}
        if target == "spirv":
            from .shader_compiler import find_glsl_compiler
            compiler = find_glsl_compiler()
            if compiler is not None:
                options["spirv_compiler"] = compiler
        vertex_result = osh.compile(vertex, **options)
        fragment_result = osh.compile(fragment, **options)
        return cls(vertex_result, fragment_result, osh.link_graphics(vertex_result, fragment_result))

    @classmethod
    def scene(cls, *, target: str, validate: bool = True):
        """Compile Ordinary Light's built-in unlit scene raster program."""
        try:
            from .raster_shaders import scene_fragment, scene_vertex
        except ImportError as error:
            raise RuntimeError(
                "built-in raster shaders require the ordinaryshade package"
            ) from error
        return cls.compile(scene_vertex, scene_fragment, target=target, validate=validate)

    @property
    def cache_key(self) -> str:
        return f"{self.vertex.cache_key}:{self.fragment.cache_key}"


@dataclass(frozen=True, slots=True)
class RasterMesh:
    """Typed interleaved vertex data consumed by every raster backend."""

    vertices: np.ndarray
    indices: np.ndarray | None = None
    layout: RasterVertexLayout | None = None

    def __post_init__(self):
        vertices = np.ascontiguousarray(self.vertices, dtype=np.float32)
        if vertices.ndim != 2 or vertices.shape[1] < 1:
            raise ValueError("raster vertices must have shape (N, components)")
        layout = self.layout or RasterVertexLayout(
            vertices.shape[1] * 4,
            (RasterVertexAttribute(0, f"float32x{vertices.shape[1]}" if vertices.shape[1] > 1 else "float32", 0, "position"),),
        )
        if vertices.strides[0] != layout.stride:
            raise ValueError("vertex array row size must match the vertex layout stride")
        indices = None if self.indices is None else np.ascontiguousarray(self.indices, dtype=np.uint32).reshape(-1)
        if indices is not None and indices.size and int(indices.max()) >= len(vertices):
            raise ValueError("a raster index refers to a missing vertex")
        object.__setattr__(self, "vertices", vertices)
        object.__setattr__(self, "indices", indices)
        object.__setattr__(self, "layout", layout)


def camera_matrix(camera, width: int, height: int) -> np.ndarray:
    """Return an OpenGL-style world-to-clip matrix for an Ordinary Light camera."""
    from .cameras import OrthographicCamera, PanoramicCamera
    if isinstance(camera, PanoramicCamera):
        raise ValueError("panoramic cameras require a non-linear raster projection")
    eye = np.asarray(camera.position, np.float32)
    forward = np.asarray(camera.target, np.float32) - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.asarray(camera.up, np.float32)); right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    view = np.eye(4, dtype=np.float32)
    view[:3, :3] = np.array((right, up, -forward), np.float32)
    view[:3, 3] = -view[:3, :3] @ eye
    aspect, near, far = width / height, 0.01, 10000.0
    if isinstance(camera, OrthographicCamera):
        top = camera.vertical_size * 0.5; right_extent = top * aspect
        projection = np.array(((1/right_extent,0,0,0),(0,1/top,0,0),(0,0,-2/(far-near),-(far+near)/(far-near)),(0,0,0,1)), np.float32)
    else:
        scale = 1.0 / np.tan(np.radians(camera.vertical_fov_degrees) * 0.5)
        projection = np.array(((scale/aspect,0,0,0),(0,scale,0,0),(0,0,-(far+near)/(far-near),-2*far*near/(far-near)),(0,0,-1,0)), np.float32)
    return projection @ view


def scene_mesh(scene, camera, width: int, height: int) -> RasterMesh:
    """Flatten visible scene meshes into position/color clip-space draw data."""
    matrix = camera_matrix(camera, width, height)
    rows, indices, base = [], [], 0
    for mesh in scene.visible_meshes:
        world = np.column_stack((mesh.world_vertices, np.ones(len(mesh.vertices), np.float32)))
        clip = world @ matrix.T
        color = np.broadcast_to(np.asarray(mesh.material.base_color, np.float32), (len(world), 3))
        rows.append(np.column_stack((clip, color)))
        indices.append(mesh.indices.reshape(-1) + base)
        base += len(world)
    vertices = np.concatenate(rows).astype(np.float32) if rows else np.empty((0, 7), np.float32)
    index_data = np.concatenate(indices).astype(np.uint32) if indices else np.empty(0, np.uint32)
    layout = RasterVertexLayout(28, (
        RasterVertexAttribute(0, "float32x4", 0, "position"),
        RasterVertexAttribute(1, "float32x3", 16, "base_color"),
    ))
    return RasterMesh(vertices, index_data, layout)


def triangle_mesh() -> RasterMesh:
    return RasterMesh(np.array(((-0.7, -0.6), (0.7, -0.6), (0.0, 0.7)), np.float32))


__all__ = ["RasterMesh", "RasterProgram", "RasterState", "RasterVertexAttribute", "RasterVertexLayout", "camera_matrix", "scene_mesh", "triangle_mesh"]
