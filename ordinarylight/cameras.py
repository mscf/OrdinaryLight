"""Camera models exposed by the backend-neutral public API."""

from dataclasses import dataclass

import numpy as np


def _vec3(value, name):
    result = np.asarray(value, dtype=np.float32)
    if result.shape != (3,):
        raise ValueError(f"{name} must contain exactly three values")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain finite values")
    return result


@dataclass(frozen=True)
class PerspectiveCamera:
    """Pinhole camera described by a look-at transform and vertical field of view."""

    position: tuple[float, float, float]
    target: tuple[float, float, float]
    up: tuple[float, float, float] = (0.0, 1.0, 0.0)
    vertical_fov_degrees: float = 45.0

    def __post_init__(self):
        _validate_look_at(self.position, self.target, self.up)
        if not 1.0 <= self.vertical_fov_degrees < 179.0:
            raise ValueError("vertical_fov_degrees must be in [1, 179)")


@dataclass(frozen=True)
class OrthographicCamera:
    """Parallel-projection camera with a world-space vertical view size."""

    position: tuple[float, float, float]
    target: tuple[float, float, float]
    up: tuple[float, float, float] = (0.0, 1.0, 0.0)
    vertical_size: float = 2.0

    def __post_init__(self):
        _validate_look_at(self.position, self.target, self.up)
        if not np.isfinite(self.vertical_size) or self.vertical_size <= 0.0:
            raise ValueError("vertical_size must be finite and positive")

    @property
    def vertical_fov_degrees(self):
        """Compatibility scale used only by motion heuristics."""
        return 90.0


@dataclass(frozen=True)
class PanoramicCamera:
    """Equirectangular camera centered on its look-at orientation."""

    position: tuple[float, float, float]
    target: tuple[float, float, float]
    up: tuple[float, float, float] = (0.0, 1.0, 0.0)
    horizontal_fov_degrees: float = 360.0
    vertical_fov_degrees: float = 180.0

    def __post_init__(self):
        _validate_look_at(self.position, self.target, self.up)
        if not 1.0 <= self.horizontal_fov_degrees <= 360.0:
            raise ValueError("horizontal_fov_degrees must be in [1, 360]")
        if not 1.0 <= self.vertical_fov_degrees <= 180.0:
            raise ValueError("vertical_fov_degrees must be in [1, 180]")


def _validate_look_at(position, target, up):
    position = _vec3(position, "position")
    target = _vec3(target, "target")
    up = _vec3(up, "up")
    forward = target - position
    if np.linalg.norm(forward) < 1e-6:
        raise ValueError("camera position and target must differ")
    if np.linalg.norm(up) < 1e-6:
        raise ValueError("camera up vector cannot be zero")
    if np.linalg.norm(np.cross(forward, up)) < 1e-6:
        raise ValueError("camera up vector cannot be parallel to its view direction")


CAMERA_TYPES = (PerspectiveCamera, OrthographicCamera, PanoramicCamera)
Camera = PerspectiveCamera | OrthographicCamera | PanoramicCamera


__all__ = [
    "CAMERA_TYPES", "Camera", "OrthographicCamera", "PanoramicCamera",
    "PerspectiveCamera",
]
