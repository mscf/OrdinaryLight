"""Target-neutral shadow-map planning for raster implementations."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..lights import DirectionalLight, SpotLight


@dataclass(frozen=True, slots=True)
class ShadowMapRequest:
    """One depth-only shadow render requested from a native target.

    The graphics target owns the image and pass objects; this record keeps
    light selection, resolution and bias policy independent of Vulkan/WebGPU.
    """

    light_index: int
    kind: str
    extent: tuple[int, int] = (1024, 1024)
    depth_bias: float = 0.001
    normal_bias: float = 0.0
    view_projection: np.ndarray = field(
        default_factory=lambda: np.eye(4, dtype=np.float32),
        compare=False, repr=False,
    )

    def __post_init__(self):
        if self.kind not in {"directional", "spot"}:
            raise ValueError("shadow-map kind must be directional or spot")
        if self.light_index < 0:
            raise ValueError("shadow-map light index cannot be negative")
        if len(self.extent) != 2 or min(self.extent) < 1:
            raise ValueError("shadow-map extent must be positive")
        if self.depth_bias < 0.0 or self.normal_bias < 0.0:
            raise ValueError("shadow-map bias cannot be negative")
        matrix = np.ascontiguousarray(self.view_projection, dtype=np.float32)
        if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
            raise ValueError("shadow-map view_projection must be a finite mat4")
        matrix.flags.writeable = False
        object.__setattr__(self, "view_projection", matrix)


def _look_at(position, target):
    forward = np.asarray(target, np.float32) - np.asarray(position, np.float32)
    forward /= max(float(np.linalg.norm(forward)), 1e-8)
    up = np.array((0.0, 1.0, 0.0), np.float32)
    if abs(float(np.dot(forward, up))) > 0.98:
        up = np.array((1.0, 0.0, 0.0), np.float32)
    right = np.cross(forward, up); right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    view = np.eye(4, dtype=np.float32)
    view[:3, :3] = np.array((right, up, -forward), np.float32)
    view[:3, 3] = -view[:3, :3] @ np.asarray(position, np.float32)
    return view


def _scene_bounds(scene):
    points = [mesh.world_vertices for mesh in scene.visible_meshes if len(mesh.vertices)]
    if not points:
        return np.zeros(3, np.float32), 1.0
    points = np.concatenate(points)
    minimum, maximum = points.min(axis=0), points.max(axis=0)
    center = (minimum + maximum) * 0.5
    return center, max(float(np.linalg.norm(maximum - minimum)) * 0.5, 0.1)


def _light_matrix(scene, light):
    center, radius = _scene_bounds(scene)
    if isinstance(light, DirectionalLight):
        direction = np.asarray(light.direction, np.float32)
        direction /= np.linalg.norm(direction)
        view = _look_at(center - direction * radius * 2.0, center)
        near, far = 0.01, radius * 4.0
        projection = np.array((
            (1 / radius, 0, 0, 0), (0, 1 / radius, 0, 0),
            # Native Vulkan/WebGPU depth is [0, 1], not OpenGL [-1, 1].
            (0, 0, 1 / (near - far), near / (near - far)),
            (0, 0, 0, 1),
        ), np.float32)
    else:
        position = np.asarray(light.position, np.float32)
        direction = np.asarray(light.direction, np.float32)
        view = _look_at(position, position + direction)
        near = 0.01
        far = float(light.range) if light.range is not None else max(
            float(np.linalg.norm(center - position)) + radius, 1.0,
        )
        scale = 1.0 / np.tan(float(light.outer_cone_angle))
        projection = np.array((
            (scale, 0, 0, 0), (0, scale, 0, 0),
            (0, 0, far / (near - far), far * near / (near - far)),
            (0, 0, -1, 0),
        ), np.float32)
    return projection @ view


def plan_shadow_maps(
    scene, *, extent=(1024, 1024), max_maps=4, normal_bias_texels=1.5,
):
    """Select directional and spot lights supported by the first shadow tier."""
    if max_maps < 0:
        raise ValueError("max_maps cannot be negative")
    if not np.isfinite(normal_bias_texels) or normal_bias_texels < 0.0:
        raise ValueError("normal_bias_texels must be finite and non-negative")
    _center, radius = _scene_bounds(scene)
    world_texel = (2.0 * radius) / float(max(extent))
    requests = []
    for index, light in enumerate(scene.lights):
        if isinstance(light, DirectionalLight):
            kind = "directional"
        elif isinstance(light, SpotLight):
            kind = "spot"
        else:
            continue
        requests.append(ShadowMapRequest(
            index, kind, tuple(extent),
            normal_bias=float(normal_bias_texels) * world_texel,
            view_projection=_light_matrix(scene, light),
        ))
        if len(requests) == max_maps:
            break
    return tuple(requests)


__all__ = ["ShadowMapRequest", "plan_shadow_maps"]
