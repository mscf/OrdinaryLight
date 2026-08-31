"""Backend-neutral raster programs, pipeline state, and draw data."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
import inspect
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np


_FORMATS = {
    "float32": (np.float32, 1), "float32x2": (np.float32, 2),
    "float32x3": (np.float32, 3), "float32x4": (np.float32, 4),
}

_SCENE_PROGRAM_CACHE = {}


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
    cull_mode: str = "none"
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
    shading_model: str = "pbr"
    shadow_map_size: int = 256
    shadow_cull_mode: str = "none"
    shadow_normal_bias: float = 1.5
    optical_quality: str = "environment"
    screen_space_ray_steps: int = 24
    screen_space_optical_layers: int = 4
    optical_debug_view: str = "off"
    material_program: object | None = None
    material_modifier: object | None = None
    material_hook: object | None = None

    def __post_init__(self):
        if self.ambient_light < 0.0:
            raise ValueError("ambient_light cannot be negative")
        if not 0.0 <= self.temporal_weight < 1.0:
            raise ValueError("temporal_weight must be in [0, 1)")
        if self.tone_mapping not in {"none", "reinhard", "aces"}:
            raise ValueError("tone_mapping must be none, reinhard, or aces")
        if not 0 <= int(self.volume_slices) <= 1024:
            raise ValueError("volume_slices must be between 0 and 1024")
        if self.shading_model not in {"pbr", "diffuse"}:
            raise ValueError("shading_model must be pbr or diffuse")
        if not 32 <= int(self.shadow_map_size) <= 8192:
            raise ValueError("shadow_map_size must be between 32 and 8192")
        if self.shadow_cull_mode not in {"none", "front", "back"}:
            raise ValueError("shadow_cull_mode must be none, front, or back")
        if not np.isfinite(self.shadow_normal_bias) or self.shadow_normal_bias < 0.0:
            raise ValueError("shadow_normal_bias must be a finite non-negative texel count")
        if self.optical_quality not in {"environment", "screen-space"}:
            raise ValueError("optical_quality must be environment or screen-space")
        if not 4 <= int(self.screen_space_ray_steps) <= 128:
            raise ValueError("screen_space_ray_steps must be between 4 and 128")
        if not 1 <= int(self.screen_space_optical_layers) <= 16:
            raise ValueError(
                "screen_space_optical_layers must be between 1 and 16"
            )
        if self.optical_debug_view not in {
            "off", "hit", "uv", "depth-delta", "confidence", "object-id",
            "depth-trace", "refraction-hit", "refraction-uv",
            "refraction-source",
        }:
            raise ValueError(
                "optical_debug_view must be off, hit, uv, depth-delta, "
                "confidence, object-id, depth-trace, refraction-hit, "
                "refraction-uv, or refraction-source"
            )
        if self.material_program is not None:
            from ..materials import MaterialProgram
            if not isinstance(self.material_program, MaterialProgram):
                raise TypeError("material_program must be created by @material")
        from ..materials.gpu import modifier_signature
        if self.material_modifier is not None and self.material_hook is not None:
            raise ValueError("set material_modifier or material_hook, not both")
        modifier = self.material_modifier or self.material_hook
        modifier_signature(modifier)
        if modifier is not None and self.material_modifier is None:
            object.__setattr__(self, "material_modifier", modifier)


def raster_material_hook(function):
    """Compatibility alias for :func:`ordinarylight.material_modifier`.

    Hooks receive and return the stable ``RasterSurface`` ABI declared in
    :mod:`ordinarylight.shaders.raster_programs`. They run after texture and
    normal-map evaluation but before lighting, on Vulkan and WebGPU alike.
    """
    from ..materials.gpu import material_modifier
    return material_modifier(function)


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
    root = Path(__file__).resolve().parents[1] / "shaders"
    manifest_path = root / "raster_scene.json"
    if not manifest_path.is_file():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_path = root / "raster_programs.py"
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


def _load_shadow_artifact(target):
    root = Path(__file__).resolve().parents[1] / "shaders"
    manifest_path = root / "raster_scene.json"
    if not manifest_path.is_file():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_path = root / "raster_programs.py"
    if (
        source_path.is_file()
        and hashlib.sha256(source_path.read_bytes()).hexdigest()
        != manifest.get("source_sha256")
    ):
        return None
    records = manifest.get("programs", {}).get(target, {}).get("shadow")
    if records is None:
        return None
    stages = {}
    for name in ("vertex", "fragment"):
        record = records[name]
        artifact_path = root / record["file"]
        payload = artifact_path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != record["sha256"]:
            raise RuntimeError(
                f"built-in raster artifact checksum mismatch: {artifact_path.name}"
            )
        stages[name] = _ArtifactShader(
            target=target,
            source=payload.decode("utf-8") if target == "wgsl" else "",
            binary=payload if target == "spirv" else None,
            reflection=_artifact_stage(record["reflection"]),
            cache_key=record["cache_key"],
        )
    varying_records = manifest.get("program_reflection", {}).get(
        target, {},
    ).get("shadow", {}).get("varyings", ())
    varyings = tuple(
        _ArtifactIO(
            item["name"], item["type"], item.get("location"),
            item.get("builtin"),
        ) for item in varying_records
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
    def compile(
        cls, vertex, fragment, *, target: str, validate: bool = True,
        helpers=(),
    ):
        try:
            import ordinaryshade as osh
        except ImportError as error:
            raise RuntimeError("compiling Python raster shaders requires ordinaryshade") from error
        options = {"target": target, "validate": validate}
        if target == "spirv":
            from ..shaders.compiler import find_glsl_compiler
            compiler = find_glsl_compiler()
            if compiler is not None:
                options["spirv_compiler"] = compiler
        vertex_result = osh.compile(vertex, **options)
        fragment_result = osh.compile(fragment, helpers=helpers, **options)
        return cls(vertex_result, fragment_result, osh.link_graphics(vertex_result, fragment_result))

    @classmethod
    def scene(
        cls, *, target: str, validate: bool = True, material_programs=(),
        material_modifier=None, material_hook=None,
    ):
        """Compile/cache the scene program for one material-program set.

        Material programs share one stable GPU record ABI.  Their ordered
        identities select deterministic raster approximations in the fragment
        stage, while the cache key keeps future specialized variants isolated.
        """
        material_programs = tuple(material_programs)
        from ..materials import MaterialProgram
        if any(not isinstance(item, MaterialProgram) for item in material_programs):
            raise TypeError("material_programs must contain @material programs")
        signature = tuple(
            (item.name, item.raster_kind, item.glsl())
            for item in material_programs
        )
        from ..materials.gpu import modifier_signature
        if material_modifier is not None and material_hook is not None:
            raise ValueError("set material_modifier or material_hook, not both")
        material_modifier = material_modifier or material_hook
        hook_signature = modifier_signature(material_modifier)
        key = (target, bool(validate), signature, hook_signature)
        cached = _SCENE_PROGRAM_CACHE.get(key)
        if cached is not None:
            return cached
        artifact = _load_scene_artifact(target) if material_modifier is None else None
        if artifact is not None:
            if signature:
                variant = hashlib.sha256(
                    repr(signature).encode("utf-8")
                ).hexdigest()[:16]
                artifact = cls(
                    replace(
                        artifact.vertex,
                        cache_key=f"{artifact.vertex.cache_key}:{variant}",
                    ),
                    replace(
                        artifact.fragment,
                        cache_key=f"{artifact.fragment.cache_key}:{variant}",
                    ),
                    artifact.reflection,
                )
            _SCENE_PROGRAM_CACHE[key] = artifact
            return artifact
        try:
            from ..shaders.raster_programs import (
                blend_surface_parameters, default_material_modifier,
                scene_fragment, scene_vertex,
            )
        except ImportError as error:
            raise RuntimeError(
                "built-in raster shaders require the ordinaryshade package"
            ) from error
        result = cls.compile(
            scene_vertex, scene_fragment, target=target, validate=validate,
            helpers=(
                blend_surface_parameters,
                material_modifier or default_material_modifier,
            ),
        )
        if signature:
            variant = hashlib.sha256(repr(signature).encode("utf-8")).hexdigest()[:16]
            result = cls(
                replace(result.vertex, cache_key=f"{result.vertex.cache_key}:{variant}"),
                replace(result.fragment, cache_key=f"{result.fragment.cache_key}:{variant}"),
                result.reflection,
            )
        _SCENE_PROGRAM_CACHE[key] = result
        return result

    @classmethod
    def shadow(cls, *, target: str, validate: bool = True):
        """Load the built-in depth-producing shadow raster program."""
        artifact = _load_shadow_artifact(target)
        if artifact is not None:
            return artifact
        try:
            from ..shaders.raster_programs import shadow_fragment, shadow_vertex
        except ImportError as error:
            raise RuntimeError(
                "built-in raster shaders require the ordinaryshade package"
            ) from error
        return cls.compile(
            shadow_vertex, shadow_fragment, target=target, validate=validate,
        )

    @property
    def cache_key(self) -> str:
        return f"{self.vertex.cache_key}:{self.fragment.cache_key}"


@dataclass(frozen=True, slots=True)
class RasterMesh:
    """Typed interleaved vertex data consumed by every raster backend."""

    vertices: np.ndarray
    indices: np.ndarray | None = None
    layout: RasterVertexLayout | None = None
    resources: dict[str, Any] | None = None

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
        object.__setattr__(self, "resources", dict(self.resources or {}))


_MATERIAL_TEXTURE_FIELDS = (
    ("base_color_texture", "base_color_transform", 0, False),
    ("metallic_roughness_texture", "metallic_roughness_transform", 0, True),
    ("emissive_texture", "emissive_transform", 1, False),
    ("normal_texture", "normal_transform", 2, True),
    ("occlusion_texture", "occlusion_transform", 0, True),
    ("transmission_texture", "transmission_transform", 0, True),
    ("thickness_texture", "thickness_transform", 0, True),
    ("clearcoat_texture", "base_color_transform", 0, True),
    ("sheen_texture", "base_color_transform", 0, False),
    ("anisotropy_texture", "base_color_transform", 0, True),
    ("subsurface_texture", "base_color_transform", 0, True),
)


def _box_blur_environment(image, radius):
    """Blur an equirectangular HDR image with wrapped longitude."""
    if radius <= 0:
        return np.asarray(image, np.float32).copy()

    def blur_axis(values, axis, pad_mode):
        padding = [(0, 0)] * values.ndim
        padding[axis] = (radius, radius)
        padded = np.pad(values, padding, mode=pad_mode)
        cumulative = np.cumsum(padded, axis=axis, dtype=np.float64)
        leading_shape = list(cumulative.shape)
        leading_shape[axis] = 1
        cumulative = np.concatenate((
            np.zeros(leading_shape, np.float64), cumulative,
        ), axis=axis)
        high = [slice(None)] * values.ndim
        low = [slice(None)] * values.ndim
        high[axis] = slice(radius * 2 + 1, None)
        low[axis] = slice(None, -(radius * 2 + 1))
        return (
            cumulative[tuple(high)] - cumulative[tuple(low)]
        ) / float(radius * 2 + 1)

    horizontal = blur_axis(np.asarray(image, np.float32), 1, "wrap")
    return blur_axis(horizontal, 0, "edge").astype(np.float32)


def _prefiltered_environment_texture(image):
    """Pack four roughness levels into one raster-only environment strip."""
    from ..scene import Texture

    image = np.asarray(image, np.float32)
    log_range = max(float(np.log2(1.0 + np.max(image))), 1e-6)
    levels = []
    for radius in (0, 2, 8, 24):
        filtered = _box_blur_environment(image, radius)
        encoded = np.empty((*filtered.shape[:2], 4), np.uint8)
        encoded[..., :3] = np.clip(
            np.log2(1.0 + filtered) / log_range * 255.0 + 0.5,
            0, 255,
        ).astype(np.uint8)
        encoded[..., 3] = 255
        levels.append(encoded)
    return Texture(
        np.concatenate(levels, axis=1),
        wrap_s="clamp", wrap_t="clamp",
    )


def _material_atlas(scene, enabled=True, shadow_depth=None):
    """Pack every material image into one portable RGBA8 atlas.

    The first three texels are stable neutral samples: white, black, and a
    tangent-space +Z normal.  This lets the fragment shader use one branch-free
    sampling path for textured and untextured materials on every target.
    """
    textures = []
    lookup = {}
    if enabled:
        for mesh in scene.visible_meshes:
            for texture_name, _transform_name, _neutral, linear in _MATERIAL_TEXTURE_FIELDS:
                texture = getattr(mesh.material, texture_name)
                key = (id(texture), linear)
                if texture is not None and key not in lookup:
                    lookup[key] = len(textures)
                    textures.append((texture, linear))
        environment_textures = {}
        probe_texture_cache = {}
        captured_probes = [probe for probe in scene.reflection_probes if probe.captured]
        if captured_probes:
            from ..probes import select_reflection_probes
            for mesh_index, mesh in enumerate(scene.visible_meshes):
                center = np.asarray(mesh.world_vertices, np.float32).mean(axis=0)
                selected = select_reflection_probes(captured_probes, center, limit=2)
                if not selected:
                    continue
                selected_textures = []
                for probe, weight in selected:
                    texture = probe_texture_cache.get(id(probe))
                    if texture is None:
                        texture = _prefiltered_environment_texture(
                            probe.image[..., :3] * probe.intensity,
                        )
                        probe_texture_cache[id(probe)] = texture
                        textures.append((texture, True))
                    selected_textures.append(texture)
                environment_textures[mesh_index] = (
                    tuple(selected_textures), selected,
                )
        elif scene.environment is not None and scene.environment.image is not None:
            texture = _prefiltered_environment_texture(scene.environment.image)
            environment_textures[None] = ((texture,), ())
            lookup[("environment", None)] = len(textures)
            textures.append((texture, True))
    color_height = max((item.pixels.shape[0] for item, _ in textures), default=1)
    shadow_height = 0 if shadow_depth is None else shadow_depth.shape[0]
    shadow_width = 0 if shadow_depth is None else shadow_depth.shape[1]
    height = color_height + shadow_height
    width = max(3 + sum(item.pixels.shape[1] for item, _ in textures), shadow_width)
    atlas = np.full((height, width, 4), 255, np.uint8)
    atlas[0, 0] = (255, 255, 255, 255)
    atlas[0, 1] = (0, 0, 0, 255)
    # Stored in an sRGB image, so encode the linear (0.5, 0.5, 1.0)
    # tangent-space neutral before hardware sampling decodes it.
    atlas[0, 2] = (188, 188, 255, 255)
    rectangles = {}
    x = 3
    for texture, linear in textures:
        image = texture.pixels.copy()
        if linear:
            rgb = image[..., :3].astype(np.float32) / 255.0
            encoded = np.where(
                rgb <= 0.0031308, rgb * 12.92,
                1.055 * np.power(rgb, 1.0 / 2.4) - 0.055,
            )
            image[..., :3] = np.rint(encoded * 255.0).astype(np.uint8)
        h, w = image.shape[:2]
        atlas[:h, x:x + w] = image
        rectangles[(id(texture), linear)] = (x, 0, w, h, width, height)
        x += w
    for mesh_index, (selected_textures, selected) in environment_textures.items():
        for slot, texture in enumerate(selected_textures):
            rectangles[("environment", mesh_index, slot)] = rectangles[(id(texture), True)]
        rectangles[("environment", mesh_index)] = rectangles[("environment", mesh_index, 0)]
        rectangles[("probe_selection", mesh_index)] = selected
    if ("environment", None) in rectangles:
        rectangles["environment"] = rectangles[("environment", None)]
    shadow_rectangle = None
    if shadow_depth is not None:
        y = color_height
        atlas[y:y + shadow_height, :shadow_width, :3] = 255
        encoded_depth = np.power(
            np.clip(1.0 - shadow_depth, 0.0, 1.0), 0.25,
        )
        atlas[y:y + shadow_height, :shadow_width, 3] = np.rint(
            encoded_depth * 255.0
        ).astype(np.uint8)
        shadow_rectangle = (
            0, y, shadow_width, shadow_height, width, height,
        )
    return atlas, rectangles, shadow_rectangle


# Compatibility name retained for callers that only care about the resulting
# packed image.  The contents are now the complete material atlas.
_base_color_atlas = _material_atlas


def _rasterize_shadow_depth(scene, matrix, size):
    depth = np.ones((size, size), np.float32)
    for mesh in scene.visible_meshes:
        if mesh.material.alpha_mode == "blend":
            continue
        if not len(mesh.indices):
            continue
        world = np.column_stack((
            mesh.world_vertices,
            np.ones(len(mesh.world_vertices), np.float32),
        ))
        clip = world @ matrix.T
        ndc = clip[:, :3] / np.maximum(np.abs(clip[:, 3:4]), 1e-8)
        screen = np.column_stack((
            (ndc[:, 0] * 0.5 + 0.5) * size,
            (1.0 - (ndc[:, 1] * 0.5 + 0.5)) * size,
        ))
        for triangle in mesh.indices:
            points = screen[triangle]
            minimum = np.maximum(np.floor(points.min(axis=0)).astype(int), 0)
            maximum = np.minimum(np.ceil(points.max(axis=0)).astype(int), size - 1)
            if np.any(maximum < minimum):
                continue
            edge1, edge2 = points[1] - points[0], points[2] - points[0]
            area = edge1[0] * edge2[1] - edge1[1] * edge2[0]
            if abs(float(area)) < 1e-8:
                continue
            xs = np.arange(minimum[0], maximum[0] + 1, dtype=np.float32) + 0.5
            ys = np.arange(minimum[1], maximum[1] + 1, dtype=np.float32) + 0.5
            gx, gy = np.meshgrid(xs, ys)
            px, py = gx - points[0, 0], gy - points[0, 1]
            w1 = (px * edge2[1] - py * edge2[0]) / area
            w2 = (edge1[0] * py - edge1[1] * px) / area
            w0 = 1.0 - w1 - w2
            inside = (w0 >= 0.0) & (w1 >= 0.0) & (w2 >= 0.0)
            values = (
                w0 * ndc[triangle[0], 2]
                + w1 * ndc[triangle[1], 2]
                + w2 * ndc[triangle[2], 2]
            )
            target = depth[minimum[1]:maximum[1] + 1, minimum[0]:maximum[0] + 1]
            np.minimum(target, np.where(inside, values, 1.0), out=target)
    return depth


def _shadow_atlas_coordinates(world, matrix, rectangle):
    clip = world @ matrix.T
    x, y, width, height, atlas_width, atlas_height = rectangle
    # Preserve homogeneous light-space coordinates through camera-space
    # rasterization.  Dividing here would make the already-projected values
    # undergo perspective interpolation a second time, producing view-dependent
    # sawtooth self-shadowing along shared edges.
    u_offset = (x + 0.5 + 0.5 * max(width - 1, 0)) / atlas_width
    u_scale = 0.5 * max(width - 1, 0) / atlas_width
    v_offset = (y + 0.5 + 0.5 * max(height - 1, 0)) / atlas_height
    v_scale = 0.5 * max(height - 1, 0) / atlas_height
    return np.column_stack((
        u_offset * clip[:, 3] + u_scale * clip[:, 0],
        v_offset * clip[:, 3] - v_scale * clip[:, 1],
        clip[:, 2],
        clip[:, 3],
    )).astype(np.float32, copy=False)


def _shadow_geometry(scene, matrix):
    rows, indices, base = [], [], 0
    for mesh in scene.visible_meshes:
        # The baseline shadow map has no colored/transmissive shadow model.
        # Treating blend surfaces as opaque casters creates dark triangular
        # wedges at their quad boundaries, so omit them until a dedicated
        # transmissive-shadow pass is selected.
        if mesh.material.alpha_mode == "blend":
            continue
        world = np.column_stack((
            mesh.world_vertices,
            np.ones(len(mesh.world_vertices), np.float32),
        ))
        rows.append(world @ matrix.T)
        indices.append(mesh.indices.reshape(-1) + base)
        base += len(world)
    vertices = (
        np.ascontiguousarray(np.concatenate(rows), dtype=np.float32)
        if rows else np.empty((0, 4), np.float32)
    )
    index_data = (
        np.ascontiguousarray(np.concatenate(indices), dtype=np.uint32)
        if indices else np.empty(0, np.uint32)
    )
    return vertices, index_data


def _atlas_uv(
    mesh, rectangles, atlas_shape, texture=None, transform=None,
    *, enabled=True, neutral=0,
    linear=False,
):
    atlas_height, atlas_width = atlas_shape[:2]
    if not enabled or texture is None:
        return np.broadcast_to(
            ((neutral + 0.5) / atlas_width, 0.5 / atlas_height),
            (len(mesh.vertices), 2),
        ).copy()
    uv = np.array(
        mesh.texcoords if transform.texcoord_set == 0 else mesh.texcoords1,
        np.float32, copy=True,
    )
    cosine, sine = np.cos(transform.rotation), np.sin(transform.rotation)
    uv *= np.asarray(transform.scale, np.float32)
    uv = uv @ np.array(((cosine, sine), (-sine, cosine)), np.float32)
    uv += np.asarray(transform.offset, np.float32)
    for axis, mode in enumerate((texture.wrap_s, texture.wrap_t)):
        if mode == "repeat": uv[:, axis] %= 1.0
        elif mode == "mirror": uv[:, axis] = 1.0 - np.abs((uv[:, axis] % 2.0) - 1.0)
        else: uv[:, axis] = np.clip(uv[:, axis], 0.0, 1.0)
    x, y, w, h, atlas_width, atlas_height = rectangles[(id(texture), linear)]
    return np.column_stack((
        (x + 0.5 + uv[:, 0] * max(w - 1, 0)) / atlas_width,
        (y + 0.5 + (1.0 - uv[:, 1]) * max(h - 1, 0)) / atlas_height,
    ))


def camera_matrix(camera, width: int, height: int) -> np.ndarray:
    """Return an OpenGL-style world-to-clip matrix for an Ordinary Light camera."""
    from ..cameras import OrthographicCamera, PanoramicCamera
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
        if candidate.material.alpha_mode == "blend":
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


def _vertex_lighting(scene, mesh, camera, config):
    color = _sample_base_color(mesh, config.textures)
    if not config.direct_lighting:
        return color
    if config.shading_model == "pbr":
        from .lighting import evaluate_vertex_lighting
        return evaluate_vertex_lighting(
            scene, mesh, camera, config, _shadow_visibility,
        )
    from ..lights import DirectionalLight, PointLight, SpotLight
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


def scene_mesh(
    scene, camera, width: int, height: int, config=None, *,
    native_shadow_maps=False, prepared_resources=None, gpu_camera=False,
) -> RasterMesh:
    """Flatten visible meshes into the shared per-fragment raster ABI."""
    config = config or RasterConfig()
    matrix = camera_matrix(camera, width, height)
    rows, indices, base = [], [], 0
    from .lighting import material_channels
    if prepared_resources is None:
        prepared_resources = prepare_scene_mesh_resources(
            scene, config, native_shadow_maps=native_shadow_maps,
        )
    shadow_request = prepared_resources["shadow_request"]
    atlas = prepared_resources["base_color_atlas"]
    atlas_rectangles = prepared_resources["atlas_rectangles"]
    shadow_rectangle = prepared_resources["shadow_rectangle"]

    light_position_type = np.zeros(4, np.float32)
    light_color = np.zeros(3, np.float32)
    from ..lights import DirectionalLight
    if scene.lights:
        light = scene.lights[0]
        if isinstance(light, DirectionalLight):
            light_position_type = np.asarray((*light.direction, 1.0), np.float32)
        else:
            light_position_type = np.asarray((*light.position, 0.0), np.float32)
        light_color = np.asarray(light.color, np.float32) * float(light.intensity)
    else:
        for emitter in scene.visible_meshes:
            if not np.any(emitter.material.emission) or not len(emitter.indices):
                continue
            triangles = emitter.world_vertices[emitter.indices]
            cross = np.cross(
                triangles[:, 1] - triangles[:, 0],
                triangles[:, 2] - triangles[:, 0],
            )
            areas = np.linalg.norm(cross, axis=1) * 0.5
            area = float(np.sum(areas))
            if area > 1e-8:
                center = np.average(
                    triangles.mean(axis=1), axis=0, weights=areas,
                )
                light_position_type = np.asarray((*center, 0.0), np.float32)
                light_color = (
                    np.asarray(emitter.material.emission, np.float32) * area
                )
            break
    camera_position = np.asarray(camera.position, np.float32)
    light_color_ambient = np.asarray(
        (*light_color, config.ambient_light), np.float32,
    )
    if not config.direct_lighting:
        light_color_ambient[:3] = 0.0
    visible_meshes = tuple(scene.visible_meshes)
    opaque_meshes = tuple(
        mesh for mesh in visible_meshes if mesh.material.alpha_mode != "blend"
    )
    optical_meshes = tuple(
        mesh for mesh in opaque_meshes
        if config.optical_quality == "screen-space" and (
            mesh.material.transmission > 0.0 or mesh.material.metallic >= 0.5
        )
    )
    # Opaque reflectors participate in both phases: the prepass makes them
    # visible to neighboring screen-space rays, while the optical pass
    # reshades their own pixels. Transmissive objects must remain absent from
    # the prepass or they would hide the background they are meant to refract.
    prepass_excluded_ids = {
        id(mesh) for mesh in optical_meshes
        if mesh.material.transmission > 0.0
    }
    prepass_meshes = tuple(
        mesh for mesh in opaque_meshes if id(mesh) not in prepass_excluded_ids
    )
    camera_forward = np.asarray(camera.target, np.float32) - camera_position
    camera_forward /= max(float(np.linalg.norm(camera_forward)), 1e-8)
    def optical_sort_key(mesh):
        vertices = np.asarray(mesh.world_vertices, np.float32)
        center = np.mean(vertices, axis=0)
        radius = float(np.max(np.linalg.norm(vertices - center, axis=1)))
        center_depth = float(np.dot(
            center - camera_position, camera_forward,
        ))
        # Back-to-front composition is governed by the nearest visible extent,
        # not the farthest point of the bounds.  This distinction matters for
        # concentric dielectrics: the inner shell's front surface is behind the
        # outer shell and must be composed first.  The radius also provides a
        # stable ordering when concentric centroids are numerically identical.
        return -(center_depth - radius)
    authored_transparent_meshes = tuple(
        mesh for mesh in visible_meshes if mesh.material.alpha_mode == "blend"
    )
    optical_transmissive_meshes = tuple(
        mesh for mesh in optical_meshes if mesh.material.transmission > 0.0
    )
    optical_transmissive_draw_meshes = tuple(sorted(
        optical_transmissive_meshes,
        key=optical_sort_key,
    ))
    optical_screen_bounds = []
    for optical_mesh in optical_transmissive_draw_meshes:
        optical_world = np.column_stack((
            optical_mesh.world_vertices,
            np.ones(len(optical_mesh.world_vertices), np.float32),
        ))
        optical_clip = optical_world @ matrix.T
        valid = optical_clip[:, 3] > 1e-6
        if not np.any(valid):
            optical_screen_bounds.append(None)
            continue
        optical_ndc = optical_clip[valid, :2] / optical_clip[valid, 3:4]
        optical_screen_bounds.append((
            float(np.min(optical_ndc[:, 0])),
            float(np.min(optical_ndc[:, 1])),
            float(np.max(optical_ndc[:, 0])),
            float(np.max(optical_ndc[:, 1])),
        ))
    # Two disjoint silhouettes cannot contribute to one another at their
    # visible pixels and may independently sample the immutable opaque scene.
    # With three or more refractors, remain conservative: an off-axis screen
    # ray can land inside another object's silhouette even when their primary
    # projections do not overlap pairwise.
    optical_transmissive_layers_overlap = (
        len(optical_screen_bounds) > 2
        or any(
            first is not None and second is not None
            and first[0] <= second[2] and second[0] <= first[2]
            and first[1] <= second[3] and second[1] <= first[3]
            for index, first in enumerate(optical_screen_bounds)
            for second in optical_screen_bounds[index + 1:]
        )
    )
    authored_transparent_draw_meshes = tuple(sorted(
        authored_transparent_meshes,
        # Source-alpha composition is order dependent. Camera-space depth,
        # unlike Euclidean center distance, reverses correctly when the camera
        # crosses a stack of laterally offset transparent layers.
        key=optical_sort_key,
    ))
    # Keep depth-resolved optical surfaces and genuinely alpha-composited
    # surfaces in contiguous index ranges.  Treating both as one transparent
    # draw makes opaque reflectors inherit order-dependent blending and is
    # especially unstable when several reflectors overlap on screen.
    optical_opaque_draw_meshes = tuple(sorted(
        (
            mesh for mesh in optical_meshes
            if mesh.material.transmission <= 0.0
        ),
        key=optical_sort_key,
    ))
    transparent_meshes = (
        *optical_transmissive_draw_meshes,
        *authored_transparent_draw_meshes,
    )
    optical_draw_meshes = (*optical_opaque_draw_meshes, *transparent_meshes)
    opaque_prepass_index_count = sum(mesh.indices.size for mesh in prepass_meshes)
    opaque_index_count = sum(mesh.indices.size for mesh in opaque_meshes)
    optical_opaque_index_count = sum(
        mesh.indices.size for mesh in optical_opaque_draw_meshes
    )
    transparent_index_count = sum(mesh.indices.size for mesh in transparent_meshes)
    optical_transmissive_index_count = sum(
        mesh.indices.size for mesh in optical_transmissive_draw_meshes
    )
    material_indices = {id(mesh): index for index, mesh in enumerate(visible_meshes)}
    for mesh in (*prepass_meshes, *optical_draw_meshes):
        material_index = material_indices[id(mesh)]
        world = np.column_stack((mesh.world_vertices, np.ones(len(mesh.vertices), np.float32)))
        clip = world if gpu_camera else world @ matrix.T
        (
            color, metallic, roughness, emission, transmission, occlusion,
        ) = material_channels(mesh, False)
        material_uvs = []
        for texture_name, transform_name, neutral, linear in _MATERIAL_TEXTURE_FIELDS:
            material_uvs.append(_atlas_uv(
                mesh, atlas_rectangles, atlas.shape,
                getattr(mesh.material, texture_name),
                getattr(mesh.material, transform_name),
                enabled=config.textures, neutral=neutral, linear=linear,
            ))
        if shadow_request is not None:
            shadow_world = world.copy()
            shadow_world[:, :3] += (
                mesh.world_normals * float(shadow_request.normal_bias)
            )
            shadow_coordinate = _shadow_atlas_coordinates(
                shadow_world, shadow_request.view_projection, shadow_rectangle,
            )
        else:
            shadow_coordinate = np.zeros((len(world), 4), np.float32)
        if config.shading_model == "diffuse":
            color = _vertex_lighting(scene, mesh, camera, config)
            metallic.fill(0.0); roughness.fill(1.0); transmission.fill(0.0)
            emission.fill(0.0)
        shadow_visibility = np.ones(len(world), np.float32)
        if config.shadows and shadow_request is None and np.any(light_color):
            if light_position_type[3] > 0.5:
                direction = -light_position_type[:3]
                direction /= max(float(np.linalg.norm(direction)), 1e-8)
                incoming = np.broadcast_to(direction, (len(world), 3))
                limit = np.full(len(world), np.inf, np.float32)
            else:
                delta = light_position_type[:3] - mesh.world_vertices
                limit = np.linalg.norm(delta, axis=1)
                incoming = delta / np.maximum(limit[:, None], 1e-8)
            shadow_visibility = _shadow_visibility(
                scene, mesh, mesh.world_vertices, incoming, limit,
            )
        object_id = np.full(len(world), float(mesh.id or 0), np.float32)
        rows.append(np.column_stack((
            clip, color, mesh.world_normals, mesh.world_vertices,
            metallic, roughness, transmission,
            np.full(len(world), mesh.material.opacity, np.float32), emission,
            np.broadcast_to(
                np.zeros(3, np.float32) if gpu_camera else camera_position,
                (len(world), 3),
            ),
            np.broadcast_to(light_position_type, (len(world), 4)),
            np.broadcast_to(light_color_ambient, (len(world), 4)),
            material_uvs[0],
            shadow_coordinate,
            shadow_visibility,
            object_id,
            mesh.world_tangents,
            *material_uvs[1:6],
            np.full(len(world), float(material_index), np.float32),
            *material_uvs[6:],
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
                rows.append(np.column_stack((
                    clip, rgba[:, :3], normal, world[:, :3],
                    np.zeros(len(local), np.float32),
                    np.ones(len(local), np.float32),
                    np.zeros(len(local), np.float32),
                    rgba[:, 3], rgba[:, :3],
                    np.broadcast_to(camera_position, (len(local), 3)),
                    np.zeros((len(local), 4), np.float32),
                    np.column_stack((
                        np.zeros((len(local), 3), np.float32),
                        np.ones(len(local), np.float32),
                    )),
                    np.zeros((len(local), 2), np.float32),
                    np.zeros((len(local), 4), np.float32),
                    np.ones(len(local), np.float32),
                    object_id,
                    np.zeros((len(local), 4), np.float32),
                    *[np.zeros((len(local), 2), np.float32) for _ in range(5)],
                    np.full(len(local), -1.0, np.float32),
                    *[np.zeros((len(local), 2), np.float32) for _ in range(5)],
                )))
                indices.append(base_indices + base)
                base += len(local)
    vertices = np.concatenate(rows).astype(np.float32) if rows else np.empty((0, 64), np.float32)
    index_data = np.concatenate(indices).astype(np.uint32) if indices else np.empty(0, np.uint32)
    layout = RasterVertexLayout(256, (
        RasterVertexAttribute(0, "float32x4", 0, "position"),
        RasterVertexAttribute(1, "float32x3", 16, "base_color"),
        RasterVertexAttribute(2, "float32x3", 28, "normal"),
        RasterVertexAttribute(3, "float32x3", 40, "world_position"),
        RasterVertexAttribute(4, "float32x4", 52, "material"),
        RasterVertexAttribute(5, "float32x3", 68, "emission"),
        RasterVertexAttribute(6, "float32x3", 80, "camera_position"),
        RasterVertexAttribute(7, "float32x4", 92, "light_position_type"),
        RasterVertexAttribute(8, "float32x4", 108, "light_color_ambient"),
        RasterVertexAttribute(9, "float32x2", 124, "base_color_uv"),
        RasterVertexAttribute(10, "float32x4", 132, "shadow_coordinate"),
        RasterVertexAttribute(11, "float32", 148, "shadow_visibility"),
        RasterVertexAttribute(12, "float32", 152, "object_id"),
        RasterVertexAttribute(13, "float32x4", 156, "tangent"),
        RasterVertexAttribute(14, "float32x2", 172, "metallic_roughness_uv"),
        RasterVertexAttribute(15, "float32x2", 180, "emissive_uv"),
        RasterVertexAttribute(16, "float32x2", 188, "normal_uv"),
        RasterVertexAttribute(17, "float32x2", 196, "occlusion_uv"),
        RasterVertexAttribute(18, "float32x2", 204, "transmission_uv"),
        RasterVertexAttribute(19, "float32", 212, "material_index"),
        RasterVertexAttribute(20, "float32x2", 216, "thickness_uv"),
        RasterVertexAttribute(21, "float32x2", 224, "clearcoat_uv"),
        RasterVertexAttribute(22, "float32x2", 232, "sheen_uv"),
        RasterVertexAttribute(23, "float32x2", 240, "anisotropy_uv"),
        RasterVertexAttribute(24, "float32x2", 248, "subsurface_uv"),
    ))
    shadow_vertices = prepared_resources["shadow_vertices"]
    shadow_indices = prepared_resources["shadow_indices"]
    from .resources import pack_raster_gpu_scene
    gpu_scene = pack_raster_gpu_scene(
        scene, camera, width, height,
        default_program=config.material_program,
        environment_rectangle=prepared_resources["environment_rectangle"],
        environment_log_range=prepared_resources["environment_log_range"],
        environment_parameters=prepared_resources["environment_parameters"],
        probe_parameters=prepared_resources["probe_parameters"],
        environment_rectangle_secondary=prepared_resources["environment_rectangle_secondary"],
        environment_log_range_secondary=prepared_resources["environment_log_range_secondary"],
        probe_parameters_secondary=prepared_resources["probe_parameters_secondary"],
    )
    return RasterMesh(vertices, index_data, layout, {
        "base_color_atlas": atlas,
        "shadow_vertices": shadow_vertices,
        "shadow_indices": shadow_indices,
        "shadow_rectangle": shadow_rectangle,
        "gpu_camera": bool(gpu_camera),
        "material_buffer": gpu_scene.materials.tobytes(),
        "material_programs": gpu_scene.programs,
        "transparent": bool(authored_transparent_meshes),
        "opaque_index_count": int(opaque_index_count),
        "opaque_prepass_index_count": int(opaque_prepass_index_count),
        "optical_index_count": int(index_data.size - opaque_prepass_index_count),
        "optical_opaque_index_count": int(optical_opaque_index_count),
        "transparent_index_count": int(transparent_index_count),
        "optical_transmissive_index_count": int(
            optical_transmissive_index_count
        ),
        "optical_transmissive_index_counts": tuple(
            int(item.indices.size)
            for item in optical_transmissive_draw_meshes
        ),
        "optical_transmissive_layers_overlap": bool(
            optical_transmissive_layers_overlap
        ),
        "camera_order_token": tuple(
            material_indices[id(mesh)] for mesh in optical_draw_meshes
        ),
    })


def prepare_scene_mesh_resources(scene, config=None, *, native_shadow_maps=False):
    """Prepare scene-static atlas and light-space shadow data once.

    Camera motion does not invalidate these resources.  Callers should retain
    the result until scene geometry/materials/lights or ``config`` change.
    """
    config = config or RasterConfig()
    from ..lights import EnvironmentLight
    from .shadows import plan_shadow_maps
    shadow_requests = plan_shadow_maps(
        scene, extent=(config.shadow_map_size, config.shadow_map_size),
        max_maps=1, normal_bias_texels=config.shadow_normal_bias,
    ) if config.shadows else ()
    shadow_request = (
        shadow_requests[0]
        if shadow_requests and shadow_requests[0].light_index == 0 else None
    )
    shadow_depth = (
        _rasterize_shadow_depth(
            scene, shadow_request.view_projection, config.shadow_map_size,
        ) if shadow_request is not None and not native_shadow_maps else None
    )
    atlas, rectangles, shadow_rectangle = _base_color_atlas(
        scene, config.textures, shadow_depth,
    )
    if native_shadow_maps and shadow_request is not None:
        size = int(config.shadow_map_size)
        shadow_rectangle = (0, 0, size, size, size, size)
    shadow_vertices, shadow_indices = (
        _shadow_geometry(scene, shadow_request.view_projection)
        if shadow_request is not None else
        (np.empty((0, 4), np.float32), np.empty(0, np.uint32))
    )
    return {
        "shadow_request": shadow_request,
        "base_color_atlas": atlas,
        "atlas_rectangles": rectangles,
        "environment_rectangle": tuple(
            rectangles.get(("environment", index), rectangles.get("environment"))
            for index, _mesh in enumerate(scene.visible_meshes)
        ),
        "environment_rectangle_secondary": tuple(
            rectangles.get(("environment", index, 1))
            for index, _mesh in enumerate(scene.visible_meshes)
        ),
        "environment_log_range": tuple(
            max(float(np.log2(1.0 + float(np.max(
                rectangles[("probe_selection", index)][0][0].image
            )) * rectangles[("probe_selection", index)][0][0].intensity)), 1e-6) if rectangles.get(("probe_selection", index)) else (
                scene._pack_environment_texture(scene.environment)[1]
                if scene.environment is not None else 0.0
            ) for index, _mesh in enumerate(scene.visible_meshes)
        ),
        "environment_log_range_secondary": tuple(
            (max(float(np.log2(1.0 + float(np.max(selected[1][0].image))
                 * selected[1][0].intensity)), 1e-6) if len(selected) > 1 else 0.0)
            for index, _mesh in enumerate(scene.visible_meshes)
            for selected in (rectangles.get(("probe_selection", index), ()),)
        ),
        "environment_parameters": tuple(
            (((1.0, 1.0, 1.0, 1.0),
              rectangles[("probe_selection", index)][0][0].rotation)
             if rectangles.get(("probe_selection", index)) else
             (((*scene.environment.color, scene.environment.intensity),
               scene.environment.rotation) if scene.environment is not None else
              None))
            for index, _mesh in enumerate(scene.visible_meshes)
        ),
        "probe_parameters": tuple(
            ((selected[0][0].position, selected[0][0].radius,
              selected[0][0].projection, selected[0][0].box_min,
              selected[0][0].box_max, tuple(weight for _probe, weight in selected))
             if (selected := rectangles.get(("probe_selection", index))) else None)
            for index, _mesh in enumerate(scene.visible_meshes)
        ),
        "probe_parameters_secondary": tuple(
            ((selected[1][0].position, selected[1][0].radius,
              selected[1][0].projection, selected[1][0].box_min,
              selected[1][0].box_max, selected[1][0].rotation,
              selected[1][1]) if len(selected) > 1 else None)
            for index, _mesh in enumerate(scene.visible_meshes)
            for selected in (rectangles.get(("probe_selection", index), ()),)
        ),
        "shadow_rectangle": shadow_rectangle,
        "shadow_vertices": shadow_vertices,
        "shadow_indices": shadow_indices,
    }


def rasterize_geometry_products(mesh: RasterMesh, width: int, height: int):
    """Rasterize portable depth/normal/object-ID products on the CPU.

    This correctness path gives every graphics backend identical named-product
    semantics. Native MRT implementations can replace it without API changes.
    """
    depth = np.full((height, width), np.inf, np.float32)
    normal = np.zeros((height, width, 3), np.float32)
    object_id = np.zeros((height, width), np.uint32)
    if mesh.indices is None or not mesh.indices.size:
        depth.fill(1.0)
        return {"depth": depth, "normal": normal, "object_id": object_id}
    vertices = mesh.vertices
    semantics = {
        item.semantic: item.offset // 4 for item in mesh.layout.attributes
    }
    normal_offset = semantics.get("normal")
    object_offset = semantics.get("object_id")
    if normal_offset is None or object_offset is None:
        depth.fill(1.0)
        return {"depth": depth, "normal": normal, "object_id": object_id}
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
                direction = np.sum(
                    vertices[triangle, normal_offset:normal_offset + 3]
                    * weights[:, None], axis=0,
                )
                length = np.linalg.norm(direction)
                normal[y, x] = direction / length if length > 1e-8 else 0.0
                object_id[y, x] = np.uint32(max(0, int(round(float(np.dot(vertices[triangle, object_offset], weights))))))
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
    """Describe the portable native forward-rendering pass graph."""
    from ..pipeline import RenderPipeline, RenderStage
    config = config or RasterConfig()
    stages = []
    shadow_resource = ()
    if config.shadows and config.direct_lighting:
        stages.append(RenderStage(
            "shadow_maps", reads={"scene"}, writes={"shadow_atlas"},
        ))
        shadow_resource = ("shadow_atlas",)
    if config.optical_quality == "screen-space":
        stages.append(RenderStage(
            "opaque_prepass", reads={"scene", "camera", *shadow_resource},
            writes={"opaque_color", "opaque_depth"},
        ))
        stages.append(RenderStage(
            "screen_space_optics",
            reads={"scene", "camera", "opaque_color", "opaque_depth", *shadow_resource},
            writes={"hdr_color", "depth", "normal", "object_id"},
        ))
        color = "hdr_color"
    elif config.direct_lighting:
        stages.append(RenderStage(
            "forward_lighting",
            reads={"scene", "camera", *shadow_resource},
            writes={"hdr_color", "depth", "normal", "object_id"},
        ))
        color = "hdr_color"
    else:
        stages.append(RenderStage(
            "geometry", reads={"scene", "camera"},
            writes={"hdr_color", "depth", "normal", "object_id"},
        ))
        color = "hdr_color"
    if config.temporal_history:
        stages.append(RenderStage("temporal", reads={color}, writes={"history_color"})); color = "history_color"
    stages.append(RenderStage("tone_map", reads={color}, writes={"output"}))
    return RenderPipeline(stages, initial_resources={"scene", "camera"})


def triangle_mesh() -> RasterMesh:
    return RasterMesh(np.array(((-0.7, -0.6), (0.7, -0.6), (0.0, 0.7)), np.float32))


__all__ = ["RasterConfig", "RasterMesh", "RasterPostProcessor", "RasterProgram", "RasterState", "RasterVertexAttribute", "RasterVertexLayout", "camera_matrix", "create_raster_pipeline", "prepare_scene_mesh_resources", "rasterize_geometry_products", "scene_mesh", "triangle_mesh"]
