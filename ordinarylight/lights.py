"""Analytic light resources exposed by the backend-neutral public API."""

from dataclasses import dataclass, field
import math

import numpy as np


POINT = 0
DIRECTIONAL = 1
SPOT = 2


def _vec3(value, name):
    result = np.asarray(value, dtype=np.float32)
    if result.shape != (3,):
        raise ValueError(f"{name} must contain exactly three values")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain finite values")
    return result


def _validate_color_intensity(color, intensity):
    color = _vec3(color, "color")
    if np.any(color < 0.0):
        raise ValueError("light color components cannot be negative")
    if not math.isfinite(intensity) or intensity < 0.0:
        raise ValueError("light intensity must be finite and non-negative")


def _validate_direction(direction):
    result = _vec3(direction, "direction")
    length = float(np.linalg.norm(result))
    if length < 1e-6:
        raise ValueError("light direction cannot be zero")
    return result / length


def _validate_range(value):
    if value is not None and (not math.isfinite(value) or value <= 0.0):
        raise ValueError("light range must be finite and positive, or None")


@dataclass
class PointLight:
    """Omnidirectional point emitter.

    ``intensity`` is radiant intensity; ``range=None`` leaves the physically
    based inverse-square falloff unbounded.
    """

    position: tuple[float, float, float]
    color: tuple[float, float, float] = (1.0, 1.0, 1.0)
    intensity: float = 1.0
    range: float | None = None
    id: int | None = field(default=None, init=False)

    def __post_init__(self):
        _vec3(self.position, "position")
        _validate_color_intensity(self.color, self.intensity)
        _validate_range(self.range)


@dataclass
class DirectionalLight:
    """Distant emitter whose rays travel along ``direction``."""

    direction: tuple[float, float, float]
    color: tuple[float, float, float] = (1.0, 1.0, 1.0)
    intensity: float = 1.0
    id: int | None = field(default=None, init=False)

    def __post_init__(self):
        _validate_direction(self.direction)
        _validate_color_intensity(self.color, self.intensity)


@dataclass
class SpotLight:
    """Conical point emitter whose rays travel along ``direction``."""

    position: tuple[float, float, float]
    direction: tuple[float, float, float]
    color: tuple[float, float, float] = (1.0, 1.0, 1.0)
    intensity: float = 1.0
    inner_cone_angle: float = 0.0
    outer_cone_angle: float = math.pi / 4.0
    range: float | None = None
    id: int | None = field(default=None, init=False)

    def __post_init__(self):
        _vec3(self.position, "position")
        _validate_direction(self.direction)
        _validate_color_intensity(self.color, self.intensity)
        _validate_range(self.range)
        if not math.isfinite(self.inner_cone_angle):
            raise ValueError("inner_cone_angle must be finite")
        if not math.isfinite(self.outer_cone_angle):
            raise ValueError("outer_cone_angle must be finite")
        if not 0.0 <= self.inner_cone_angle <= self.outer_cone_angle:
            raise ValueError("inner_cone_angle must be in [0, outer_cone_angle]")
        if not 0.0 < self.outer_cone_angle <= math.pi / 2.0:
            raise ValueError("outer_cone_angle must be in (0, pi/2]")


@dataclass(frozen=True)
class EnvironmentLight:
    """Constant or equirectangular image-based environment illumination.

    ``image`` accepts linear RGB/RGBA floating-point values and therefore
    preserves radiance above one. The image is sampled in latitude-longitude
    orientation; ``rotation`` is a yaw in radians.
    """

    image: np.ndarray | None = field(default=None, compare=False, repr=False)
    color: tuple[float, float, float] = (1.0, 1.0, 1.0)
    intensity: float = 1.0
    rotation: float = 0.0

    def __post_init__(self):
        _validate_color_intensity(self.color, self.intensity)
        if not math.isfinite(self.rotation):
            raise ValueError("environment rotation must be finite")
        if self.image is None:
            return
        image = np.array(self.image, dtype=np.float32, copy=True, order="C")
        if image.ndim != 3 or image.shape[2] not in (3, 4):
            raise ValueError("environment image must have shape (height, width, 3 or 4)")
        if image.shape[0] < 1 or image.shape[1] < 1:
            raise ValueError("environment image dimensions must be positive")
        image = image[..., :3]
        if not np.all(np.isfinite(image)) or np.any(image < 0.0):
            raise ValueError("environment image radiance must be finite and non-negative")
        image = np.ascontiguousarray(image, dtype=np.float32)
        image.flags.writeable = False
        object.__setattr__(self, "image", image)


Light = PointLight | DirectionalLight | SpotLight
LIGHT_TYPES = (PointLight, DirectionalLight, SpotLight)


__all__ = [
    "DIRECTIONAL",
    "LIGHT_TYPES",
    "Light",
    "POINT",
    "SPOT",
    "DirectionalLight",
    "EnvironmentLight",
    "PointLight",
    "SpotLight",
]
