"""Backend-neutral picking and viewport-coordinate utilities."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from ..cameras import OrthographicCamera, PanoramicCamera, PerspectiveCamera
from ..scene import Mesh, Scene, Volume


@dataclass(frozen=True, slots=True)
class PickOptions:
    """Policies controlling which intersections may be selected.

    ``transmissive="surface"`` selects glass like any other surface;
    ``"through"`` ignores meshes whose material may transmit more than
    ``transmission_threshold``. Texture-driven transmission is conservatively
    treated as transmissive. Volume proxy geometry may be included, excluded,
    or selected exclusively with ``volumes``.
    """

    transmissive: str = "surface"
    transmission_threshold: float = 0.001
    volumes: str = "include"

    def __post_init__(self):
        if self.transmissive not in {"surface", "through"}:
            raise ValueError("transmissive must be 'surface' or 'through'")
        threshold = float(self.transmission_threshold)
        if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise ValueError("transmission_threshold must be in [0, 1]")
        if self.volumes not in {"include", "exclude", "only"}:
            raise ValueError("volumes must be 'include', 'exclude', or 'only'")
        object.__setattr__(self, "transmission_threshold", threshold)


@dataclass(frozen=True, slots=True)
class ViewportMapping:
    """Map UI pixels through a displayed content rectangle to render pixels.

    Sizes are ``(width, height)``. ``content_rect`` is ``(x, y, width, height)``
    in viewport coordinates and can describe letterboxing or pillarboxing.
    ``framebuffer_size`` accounts for high-DPI scaling; ``render_size`` then
    accounts for an internal dynamic-resolution target.
    """

    viewport_size: tuple[int, int]
    framebuffer_size: tuple[int, int] | None = None
    render_size: tuple[int, int] | None = None
    content_rect: tuple[float, float, float, float] | None = None

    def __post_init__(self):
        viewport = _positive_size(self.viewport_size, "viewport_size")
        framebuffer = _positive_size(
            self.framebuffer_size or viewport, "framebuffer_size"
        )
        render = _positive_size(self.render_size or framebuffer, "render_size")
        rect = self.content_rect or (0.0, 0.0, float(viewport[0]), float(viewport[1]))
        rect = tuple(float(value) for value in rect)
        if len(rect) != 4 or not np.isfinite(rect).all():
            raise ValueError("content_rect must contain four finite values")
        if rect[2] <= 0.0 or rect[3] <= 0.0:
            raise ValueError("content_rect dimensions must be positive")
        object.__setattr__(self, "viewport_size", viewport)
        object.__setattr__(self, "framebuffer_size", framebuffer)
        object.__setattr__(self, "render_size", render)
        object.__setattr__(self, "content_rect", rect)

    def map_pixel(self, pixel, *, target="render"):
        """Return a mapped pixel, or ``None`` when outside displayed content."""
        if target not in {"framebuffer", "render"}:
            raise ValueError("target must be 'framebuffer' or 'render'")
        x, y = (float(value) for value in pixel)
        if not np.isfinite((x, y)).all():
            raise ValueError("pixel coordinates must be finite")
        left, top, width, height = self.content_rect
        if x < left or y < top or x >= left + width or y >= top + height:
            return None
        normalized_x = (x - left) / width
        normalized_y = (y - top) / height
        size = self.framebuffer_size if target == "framebuffer" else self.render_size
        return normalized_x * size[0], normalized_y * size[1]


@dataclass(frozen=True, slots=True)
class PickResult:
    """The closest selectable scene object intersected by a screen-space ray."""

    object: Mesh | Volume
    object_id: int
    distance: float
    position: tuple[float, float, float]
    primitive_index: int
    triangle_index: int
    barycentric: tuple[float, float, float]


def _positive_size(value, name):
    if len(value) != 2:
        raise ValueError(f"{name} must contain width and height")
    result = tuple(int(item) for item in value)
    if min(result) < 1:
        raise ValueError(f"{name} dimensions must be positive")
    return result


def camera_ray(camera, viewport_size, pixel):
    """Return a world-space ``(origin, direction)`` ray through a pixel."""
    width, height = _positive_size(viewport_size, "viewport_size")
    x, y = (float(value) for value in pixel)
    if not np.isfinite((x, y)).all():
        raise ValueError("pixel coordinates must be finite")

    origin = np.asarray(camera.position, dtype=np.float64)
    forward = np.asarray(camera.target, dtype=np.float64) - origin
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.asarray(camera.up, dtype=np.float64))
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    ndc_x = 2.0 * ((x + 0.5) / width) - 1.0
    ndc_y = 1.0 - 2.0 * ((y + 0.5) / height)
    aspect = width / height

    if isinstance(camera, PerspectiveCamera):
        scale = math.tan(math.radians(camera.vertical_fov_degrees) * 0.5)
        direction = forward + right * (ndc_x * aspect * scale) + up * (
            ndc_y * scale
        )
        direction /= np.linalg.norm(direction)
    elif isinstance(camera, OrthographicCamera):
        origin = origin + right * (
            ndc_x * aspect * camera.vertical_size * 0.5
        ) + up * (ndc_y * camera.vertical_size * 0.5)
        direction = forward
    elif isinstance(camera, PanoramicCamera):
        yaw = math.radians(camera.horizontal_fov_degrees) * ndc_x * 0.5
        pitch = math.radians(camera.vertical_fov_degrees) * ndc_y * 0.5
        direction = (
            forward * (math.cos(pitch) * math.cos(yaw))
            + right * (math.cos(pitch) * math.sin(yaw))
            + up * math.sin(pitch)
        )
        direction /= np.linalg.norm(direction)
    else:
        raise TypeError("camera must be an Ordinary Light camera")
    return origin.astype(np.float32), direction.astype(np.float32)


def _object_allowed(value, options):
    if isinstance(value, Volume):
        return options.volumes != "exclude"
    if options.volumes == "only":
        return False
    if options.transmissive == "through":
        material = value.material
        if material.transmission_texture is not None:
            return False
        if material.transmission > options.transmission_threshold:
            return False
    return True


def _result_from_primitive(scene, origin, direction, primitive, distance, u, v):
    object_id = int(scene.triangle_instance_ids()[primitive])
    try:
        selected = scene.get_instance(object_id)
    except KeyError:
        selected = scene.get_volume(object_id)
    object_start, _object_end = scene.object_triangle_range(object_id)
    hit = origin.astype(np.float64) + direction.astype(np.float64) * distance
    return PickResult(
        selected, object_id, float(distance),
        tuple(float(value) for value in hit), primitive,
        primitive - object_start,
        (float(1.0 - u - v), float(u), float(v)),
    )


def pick_ray(scene, origin, direction, *, options=None):
    """Return the nearest policy-compatible hit along a world-space ray."""
    if not isinstance(scene, Scene):
        raise TypeError("scene must be a Scene")
    options = PickOptions() if options is None else options
    if not isinstance(options, PickOptions):
        raise TypeError("options must be PickOptions")
    origin = np.asarray(origin, dtype=np.float64)
    direction = np.asarray(direction, dtype=np.float64)
    if origin.shape != (3,) or direction.shape != (3,) or not (
        np.isfinite(origin).all() and np.isfinite(direction).all()
    ):
        raise ValueError("origin and direction must be finite three-vectors")
    length = np.linalg.norm(direction)
    if length <= 1e-12:
        raise ValueError("direction must be nonzero")
    direction /= length
    triangles = scene.render_triangles().astype(np.float64, copy=False)
    if not len(triangles):
        return None

    edge_a = triangles[:, 1] - triangles[:, 0]
    edge_b = triangles[:, 2] - triangles[:, 0]
    cross = np.cross(np.broadcast_to(direction, edge_b.shape), edge_b)
    determinant = np.einsum("ij,ij->i", edge_a, cross)
    valid = np.abs(determinant) > 1e-10
    inverse = np.zeros_like(determinant)
    inverse[valid] = 1.0 / determinant[valid]
    offset = origin - triangles[:, 0]
    u = np.einsum("ij,ij->i", offset, cross) * inverse
    valid &= (u >= 0.0) & (u <= 1.0)
    q = np.cross(offset, edge_a)
    v = np.einsum("j,ij->i", direction, q) * inverse
    valid &= (v >= 0.0) & ((u + v) <= 1.0)
    distance = np.einsum("ij,ij->i", edge_b, q) * inverse
    valid &= distance > 1e-5
    for primitive in np.argsort(np.where(valid, distance, np.inf)):
        primitive = int(primitive)
        if not valid[primitive]:
            break
        object_id = int(scene.triangle_instance_ids()[primitive])
        try:
            selected = scene.get_instance(object_id)
        except KeyError:
            selected = scene.get_volume(object_id)
        if _object_allowed(selected, options):
            return _result_from_primitive(
                scene, origin, direction, primitive, distance[primitive],
                u[primitive], v[primitive],
            )
    return None


def pick(scene, camera, viewport_size, pixel, *, options=None, mapping=None):
    """Return the nearest policy-compatible object beneath a UI pixel."""
    if mapping is not None:
        if not isinstance(mapping, ViewportMapping):
            raise TypeError("mapping must be ViewportMapping")
        pixel = mapping.map_pixel(pixel)
        if pixel is None:
            return None
        viewport_size = mapping.render_size
    origin, direction = camera_ray(camera, viewport_size, pixel)
    return pick_ray(scene, origin, direction, options=options)


__all__ = [
    "PickOptions", "PickResult", "ViewportMapping", "camera_ray", "pick",
    "pick_ray",
]
