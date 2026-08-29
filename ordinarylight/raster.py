"""Backend-neutral raster programs, pipeline state, and draw data."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
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
class RasterConfig:
    """Portable raster scene-evaluation and post-processing configuration."""

    state: RasterState = field(default_factory=RasterState)
    ambient_light: float = 1.0
    direct_lighting: bool = True
    shadows: bool = True
    textures: bool = True
    temporal_history: bool = False
    temporal_weight: float = 0.9
    tone_mapping: str = "none"
    volume_slices: int = 0

    def __post_init__(self):
        if self.ambient_light < 0.0:
            raise ValueError("ambient_light cannot be negative")
        if not 0.0 <= self.temporal_weight < 1.0:
            raise ValueError("temporal_weight must be in [0, 1)")
        if self.tone_mapping not in {"none", "reinhard", "aces"}:
            raise ValueError("tone_mapping must be none, reinhard, or aces")
        if not 0 <= int(self.volume_slices) <= 1024:
            raise ValueError("volume_slices must be between 0 and 1024")


@dataclass(frozen=True, slots=True)
class _ArtifactStage:
    stage: str
    entry_point: str
    workgroup_size: tuple[int, int, int]
    resources: tuple[Any, ...] = ()
    inputs: tuple[Any, ...] = ()
    outputs: tuple[Any, ...] = ()


@dataclass(frozen=True, slots=True)
class _ArtifactIO:
    name: str
    type: str
    location: int | None = None
    builtin: str | None = None


@dataclass(frozen=True, slots=True)
class _ArtifactReflection:
    vertex: _ArtifactStage
    fragment: _ArtifactStage
    varyings: tuple[_ArtifactIO, ...]


@dataclass(frozen=True, slots=True)
class _ArtifactShader:
    target: str
    source: str
    binary: bytes | None
    reflection: _ArtifactStage
    cache_key: str


def _artifact_stage(record):
    def io(item):
        return _ArtifactIO(
            item["name"], item["type"], item.get("location"),
            item.get("builtin"),
        )
    return _ArtifactStage(
        record["stage"], record.get("entry_point", "main"),
        tuple(record.get("workgroup_size", (1, 1, 1))),
        tuple(record.get("resources", ())),
        tuple(io(item) for item in record.get("inputs", ())),
        tuple(io(item) for item in record.get("outputs", ())),
    )


def _load_scene_artifact(target):
    root = Path(__file__).with_name("shaders")
    manifest_path = root / "raster_scene.json"
    if not manifest_path.is_file():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_path = Path(__file__).with_name("raster_shaders.py")
    if (
        source_path.is_file()
        and hashlib.sha256(source_path.read_bytes()).hexdigest()
        != manifest.get("source_sha256")
    ):
        # A source checkout with edited Python shaders should compile those
        # sources instead of silently loading stale packaged output.
        return None
    target_record = manifest.get("targets", {}).get(target)
    if target_record is None:
        return None
    stages = {}
    for name in ("vertex", "fragment"):
        record = target_record[name]
        artifact_path = root / record["file"]
        payload = artifact_path.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        if digest != record["sha256"]:
            raise RuntimeError(
                f"built-in raster artifact checksum mismatch: {artifact_path.name}"
            )
        reflection = _artifact_stage(record["reflection"])
        stages[name] = _ArtifactShader(
            target=target,
            source=payload.decode("utf-8") if target == "wgsl" else "",
            binary=payload if target == "spirv" else None,
            reflection=reflection,
            cache_key=record["cache_key"],
        )
    varyings = tuple(
        _ArtifactIO(
            item["name"], item["type"], item.get("location"),
            item.get("builtin"),
        )
        for item in manifest["reflection"]["varyings"]
    )
    reflection = _ArtifactReflection(
        stages["vertex"].reflection, stages["fragment"].reflection, varyings,
    )
    return RasterProgram(stages["vertex"], stages["fragment"], reflection)


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
        artifact = _load_scene_artifact(target)
        if artifact is not None:
            return artifact
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


def _sample_base_color(mesh, enabled):
    color = np.broadcast_to(
        np.asarray(mesh.material.base_color, np.float32),
        (len(mesh.vertices), 3),
    ).copy()
    texture = mesh.material.base_color_texture
    if not enabled or texture is None:
        return color
    transform = mesh.material.base_color_transform
    uv = mesh.texcoords if transform.texcoord_set == 0 else mesh.texcoords1
    cosine, sine = np.cos(transform.rotation), np.sin(transform.rotation)
    uv = uv * np.asarray(transform.scale, np.float32)
    uv = uv @ np.array(((cosine, sine), (-sine, cosine)), np.float32)
    uv += np.asarray(transform.offset, np.float32)
    modes = (texture.wrap_s, texture.wrap_t)
    for axis, mode in enumerate(modes):
        if mode == "repeat": uv[:, axis] %= 1.0
        elif mode == "mirror": uv[:, axis] = 1.0 - np.abs((uv[:, axis] % 2.0) - 1.0)
        else: uv[:, axis] = np.clip(uv[:, axis], 0.0, 1.0)
    height, width = texture.pixels.shape[:2]
    x = np.clip(np.rint(uv[:, 0] * (width - 1)).astype(int), 0, width - 1)
    y = np.clip(np.rint((1.0 - uv[:, 1]) * (height - 1)).astype(int), 0, height - 1)
    sampled = texture.pixels[y, x, :3].astype(np.float32) / 255.0
    return color * sampled


def _shadow_visibility(scene, owner, origins, directions, maximum_distance):
    """Portable hard-shadow query used until backend shadow maps are resident."""
    visible = np.ones(len(origins), np.float32)
    epsilon = 1e-4
    for candidate in scene.visible_meshes:
        if candidate is owner or not len(candidate.indices):
            continue
        triangles = candidate.world_vertices[candidate.indices]
        edge1 = triangles[:, 1] - triangles[:, 0]
        edge2 = triangles[:, 2] - triangles[:, 0]
        for index, (origin, direction) in enumerate(zip(origins, directions)):
            if visible[index] == 0.0:
                continue
            p = np.cross(np.broadcast_to(direction, edge2.shape), edge2)
            determinant = np.sum(edge1 * p, axis=1)
            valid = np.abs(determinant) > 1e-8
            inverse = np.zeros_like(determinant); inverse[valid] = 1.0 / determinant[valid]
            offset = origin + direction * epsilon - triangles[:, 0]
            u = np.sum(offset * p, axis=1) * inverse
            q = np.cross(offset, edge1)
            v = np.sum(np.broadcast_to(direction, q.shape) * q, axis=1) * inverse
            distance = np.sum(edge2 * q, axis=1) * inverse
            limit = maximum_distance[index] if np.ndim(maximum_distance) else maximum_distance
            hit = valid & (u >= 0.0) & (v >= 0.0) & ((u + v) <= 1.0) & (distance > epsilon) & (distance < limit - epsilon)
            if np.any(hit):
                visible[index] = 0.0
    return visible


def _vertex_lighting(scene, mesh, config):
    color = _sample_base_color(mesh, config.textures)
    if not config.direct_lighting:
        return color
    from .lights import DirectionalLight, PointLight, SpotLight
    positions, normals = mesh.world_vertices, mesh.world_normals
    radiance = np.full_like(color, config.ambient_light)
    for light in scene.lights:
        light_color = np.asarray(light.color, np.float32) * float(light.intensity)
        if isinstance(light, DirectionalLight):
            direction = -np.asarray(light.direction, np.float32)
            direction /= np.linalg.norm(direction)
            incoming = np.broadcast_to(direction, normals.shape)
            attenuation = np.ones(len(normals), np.float32)
        else:
            delta = np.asarray(light.position, np.float32) - positions
            distance = np.linalg.norm(delta, axis=1)
            incoming = delta / np.maximum(distance[:, None], 1e-6)
            attenuation = 1.0 / np.maximum(distance * distance, 1e-4)
            if light.range is not None:
                attenuation *= np.clip(1.0 - distance / light.range, 0.0, 1.0) ** 2
            if isinstance(light, SpotLight):
                axis = np.asarray(light.direction, np.float32); axis /= np.linalg.norm(axis)
                cosine = np.sum(-incoming * axis, axis=1)
                outer, inner = np.cos(light.outer_cone_angle), np.cos(light.inner_cone_angle)
                attenuation *= np.clip((cosine - outer) / max(inner - outer, 1e-5), 0.0, 1.0)
        diffuse = np.maximum(np.sum(normals * incoming, axis=1), 0.0) * attenuation
        if config.shadows:
            limit = np.full(len(positions), np.inf, np.float32) if isinstance(light, DirectionalLight) else distance
            diffuse *= _shadow_visibility(scene, mesh, positions, incoming, limit)
        radiance += diffuse[:, None] * light_color
    return color * radiance + np.asarray(mesh.material.emission, np.float32)


def scene_mesh(scene, camera, width: int, height: int, config=None) -> RasterMesh:
    """Flatten visible scene meshes into position/color clip-space draw data."""
    config = config or RasterConfig()
    matrix = camera_matrix(camera, width, height)
    rows, indices, base = [], [], 0
    for mesh in scene.visible_meshes:
        world = np.column_stack((mesh.world_vertices, np.ones(len(mesh.vertices), np.float32)))
        clip = world @ matrix.T
        color = _vertex_lighting(scene, mesh, config)
        object_id = np.full(len(world), float(mesh.id or 0), np.float32)
        rows.append(np.column_stack((
            clip, color, np.ones(len(world), np.float32),
            mesh.world_normals, object_id,
        )))
        indices.append(mesh.indices.reshape(-1) + base)
        base += len(world)
    if config.volume_slices:
        for volume in scene.visible_volumes:
            count = config.volume_slices
            normalized = volume.normalized_data
            tf = volume.material.transfer_function
            grid_y = min(normalized.shape[1], 32)
            grid_x = min(normalized.shape[2], 32)
            gx, gy = np.meshgrid(
                np.linspace(0.0, 1.0, grid_x, dtype=np.float32),
                np.linspace(0.0, 1.0, grid_y, dtype=np.float32),
            )
            base_indices = []
            for row in range(grid_y - 1):
                for column in range(grid_x - 1):
                    a = row * grid_x + column; b = a + 1
                    c = a + grid_x; d = c + 1
                    base_indices.extend((a, b, d, a, d, c))
            base_indices = np.asarray(base_indices, np.uint32)
            inverse_volume = np.linalg.inv(volume.transform.matrix)
            local_camera = inverse_volume @ np.r_[np.asarray(camera.position, np.float32), 1.0]
            # Source-alpha blending requires far-to-near slice order.
            slice_order = range(count) if local_camera[2] >= 0.5 else range(count - 1, -1, -1)
            for slice_index in slice_order:
                z = (slice_index + 0.5) / count
                data_z = min(int(z * normalized.shape[0]), normalized.shape[0] - 1)
                source = normalized[data_z]
                sx = np.clip(np.rint(gx * (source.shape[1] - 1)).astype(int), 0, source.shape[1] - 1)
                sy = np.clip(np.rint(gy * (source.shape[0] - 1)).astype(int), 0, source.shape[0] - 1)
                rgba = tf.sample(source[sy, sx]).reshape(-1, 4)
                rgba[:, 3] = 1.0 - (1.0 - rgba[:, 3]) ** (volume.material.density_scale / count)
                rgba[:, :3] *= volume.material.emission_scale
                local = np.column_stack((gx.reshape(-1), gy.reshape(-1), np.full(grid_x * grid_y, z, np.float32), np.ones(grid_x * grid_y, np.float32)))
                world = local @ volume.transform.matrix.T
                clip = world @ matrix.T
                normal = np.zeros((len(local), 3), np.float32)
                object_id = np.full(len(local), float(volume.id or 0), np.float32)
                rows.append(np.column_stack((clip, rgba, normal, object_id)))
                indices.append(base_indices + base)
                base += len(local)
    vertices = np.concatenate(rows).astype(np.float32) if rows else np.empty((0, 12), np.float32)
    index_data = np.concatenate(indices).astype(np.uint32) if indices else np.empty(0, np.uint32)
    layout = RasterVertexLayout(48, (
        RasterVertexAttribute(0, "float32x4", 0, "position"),
        RasterVertexAttribute(1, "float32x4", 16, "base_color"),
        RasterVertexAttribute(2, "float32x3", 32, "normal"),
        RasterVertexAttribute(3, "float32", 44, "object_id"),
    ))
    return RasterMesh(vertices, index_data, layout)


def rasterize_geometry_products(mesh: RasterMesh, width: int, height: int):
    """Rasterize portable depth/normal/object-ID products on the CPU.

    This correctness path gives every graphics backend identical named-product
    semantics. Native MRT implementations can replace it without API changes.
    """
    depth = np.full((height, width), np.inf, np.float32)
    normal = np.zeros((height, width, 3), np.float32)
    object_id = np.zeros((height, width), np.uint32)
    if mesh.indices is None or not mesh.indices.size or mesh.vertices.shape[1] < 12:
        depth.fill(1.0)
        return {"depth": depth, "normal": normal, "object_id": object_id}
    vertices = mesh.vertices
    clip = vertices[:, :4]
    ndc = clip[:, :3] / np.where(np.abs(clip[:, 3:4]) > 1e-8, clip[:, 3:4], 1e-8)
    screen = np.column_stack(((ndc[:, 0] * 0.5 + 0.5) * width, (1.0 - (ndc[:, 1] * 0.5 + 0.5)) * height))
    def cross2(a, b):
        return a[0] * b[1] - a[1] * b[0]
    for triangle in mesh.indices.reshape(-1, 3):
        points = screen[triangle]
        x0 = max(0, int(np.floor(points[:, 0].min()))); x1 = min(width - 1, int(np.ceil(points[:, 0].max())))
        y0 = max(0, int(np.floor(points[:, 1].min()))); y1 = min(height - 1, int(np.ceil(points[:, 1].max())))
        area = cross2(points[1] - points[0], points[2] - points[0])
        if abs(float(area)) < 1e-8 or x1 < x0 or y1 < y0: continue
        for y in range(y0, y1 + 1):
            for x in range(x0, x1 + 1):
                point = np.array((x + 0.5, y + 0.5), np.float32)
                w0 = cross2(points[1] - point, points[2] - point) / area
                w1 = cross2(points[2] - point, points[0] - point) / area
                w2 = 1.0 - w0 - w1
                if min(w0, w1, w2) < 0.0: continue
                weights = np.array((w0, w1, w2), np.float32)
                value = float(np.dot(ndc[triangle, 2] * 0.5 + 0.5, weights))
                if value >= depth[y, x]: continue
                depth[y, x] = value
                direction = np.sum(vertices[triangle, 8:11] * weights[:, None], axis=0)
                length = np.linalg.norm(direction)
                normal[y, x] = direction / length if length > 1e-8 else 0.0
                object_id[y, x] = np.uint32(max(0, int(round(float(np.dot(vertices[triangle, 11], weights))))))
    depth[~np.isfinite(depth)] = 1.0
    return {"depth": depth, "normal": normal, "object_id": object_id}


class RasterPostProcessor:
    """Backend-independent temporal accumulation and tone mapping."""

    def __init__(self, config: RasterConfig):
        self.config = config
        self._history = None
        self._signature = None
        self.accumulated_frames = 0

    @staticmethod
    def _signature_for(scene, camera, shape):
        return (getattr(scene, "revision", None), repr(camera), tuple(shape))

    def reset(self):
        self._history = None; self._signature = None; self.accumulated_frames = 0

    def process(self, image, scene, camera):
        image = np.asarray(image, np.float32)
        signature = self._signature_for(scene, camera, image.shape)
        if self.config.temporal_history and signature == self._signature and self._history is not None:
            weight = self.config.temporal_weight
            image = self._history * weight + image * (1.0 - weight)
            self.accumulated_frames += 1
        else:
            self.accumulated_frames = 1
        self._signature = signature
        self._history = image.copy()
        rgb = image[..., :3]
        if self.config.tone_mapping == "reinhard":
            rgb = rgb / (1.0 + rgb)
        elif self.config.tone_mapping == "aces":
            rgb = np.clip((rgb * (2.51 * rgb + 0.03)) / (rgb * (2.43 * rgb + 0.59) + 0.14), 0.0, 1.0)
        result = image.copy(); result[..., :3] = rgb
        return result


def create_raster_pipeline(config=None):
    """Describe the portable multipass raster resource graph."""
    from .pipeline import RenderPipeline, RenderStage
    config = config or RasterConfig()
    stages = [
        RenderStage("geometry", reads={"scene", "camera"}, writes={"color", "depth", "normal", "object_id"}),
    ]
    if config.shadows:
        stages.append(RenderStage("shadows", reads={"scene", "depth"}, writes={"shadow"}))
    if config.direct_lighting:
        stages.append(RenderStage("lighting", reads={"color", "normal", *(('shadow',) if config.shadows else ())}, writes={"lit_color"}))
        color = "lit_color"
    else:
        color = "color"
    if config.temporal_history:
        stages.append(RenderStage("temporal", reads={color}, writes={"history_color"})); color = "history_color"
    stages.append(RenderStage("post", reads={color}, writes={"output"}))
    return RenderPipeline(stages, initial_resources={"scene", "camera"})


def triangle_mesh() -> RasterMesh:
    return RasterMesh(np.array(((-0.7, -0.6), (0.7, -0.6), (0.0, 0.7)), np.float32))


__all__ = ["RasterConfig", "RasterMesh", "RasterPostProcessor", "RasterProgram", "RasterState", "RasterVertexAttribute", "RasterVertexLayout", "camera_matrix", "create_raster_pipeline", "rasterize_geometry_products", "scene_mesh", "triangle_mesh"]
