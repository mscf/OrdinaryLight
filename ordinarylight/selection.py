"""Backend-neutral scene picking for interactive applications."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .cameras import OrthographicCamera, PanoramicCamera, PerspectiveCamera
from .scene import Mesh, Scene, Volume


@dataclass(frozen=True)
class PickResult:
    """The closest selectable scene object intersected by a screen-space ray."""

    object: Mesh | Volume
    object_id: int
    distance: float
    position: tuple[float, float, float]
    primitive_index: int
    triangle_index: int
    barycentric: tuple[float, float, float]


def camera_ray(camera, viewport_size, pixel):
    """Return a world-space ``(origin, direction)`` ray through a pixel.

    ``pixel`` uses the conventional GUI coordinate system: the origin is the
    top-left corner and coordinates may be fractional.
    """
    width, height = (int(value) for value in viewport_size)
    if width < 1 or height < 1:
        raise ValueError("viewport dimensions must be positive")
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


def pick(scene, camera, viewport_size, pixel):
    """Return the nearest visible mesh or volume beneath ``pixel``.

    Picking is intentionally backend-neutral and runs only on demand. It does
    not read a rendered image back from the GPU, so applications can use it
    with swapchain and external-video presentation paths alike.
    """
    if not isinstance(scene, Scene):
        raise TypeError("scene must be a Scene")
    origin, direction = camera_ray(camera, viewport_size, pixel)
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
    offset = origin.astype(np.float64) - triangles[:, 0]
    u = np.einsum("ij,ij->i", offset, cross) * inverse
    valid &= (u >= 0.0) & (u <= 1.0)
    q = np.cross(offset, edge_a)
    v = np.einsum("j,ij->i", direction.astype(np.float64), q) * inverse
    valid &= (v >= 0.0) & ((u + v) <= 1.0)
    distance = np.einsum("ij,ij->i", edge_b, q) * inverse
    valid &= distance > 1e-5
    if not np.any(valid):
        return None
    candidates = np.where(valid, distance, np.inf)
    primitive = int(np.argmin(candidates))
    object_id = int(scene.triangle_instance_ids()[primitive])
    try:
        selected = scene.get_instance(object_id)
    except KeyError:
        selected = scene.get_volume(object_id)
    object_start, _object_end = scene.object_triangle_range(object_id)
    hit = origin.astype(np.float64) + direction.astype(np.float64) * distance[primitive]
    return PickResult(
        selected, object_id, float(distance[primitive]),
        tuple(float(value) for value in hit), primitive,
        primitive - object_start,
        (float(1.0 - u[primitive] - v[primitive]),
         float(u[primitive]), float(v[primitive])),
    )


__all__ = ["PickResult", "camera_ray", "pick"]
