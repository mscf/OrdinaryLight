"""Spot-light resource."""

from dataclasses import dataclass, field
import math

from ._validation import (
    validate_color_intensity, validate_direction, validate_range, vec3,
)


@dataclass
class SpotLight:
    position: tuple[float, float, float]
    direction: tuple[float, float, float]
    color: tuple[float, float, float] = (1.0, 1.0, 1.0)
    intensity: float = 1.0
    inner_cone_angle: float = 0.0
    outer_cone_angle: float = math.pi / 4.0
    range: float | None = None
    id: int | None = field(default=None, init=False)

    def __post_init__(self):
        vec3(self.position, "position")
        validate_direction(self.direction)
        validate_color_intensity(self.color, self.intensity)
        validate_range(self.range)
        if not math.isfinite(self.inner_cone_angle):
            raise ValueError("inner_cone_angle must be finite")
        if not math.isfinite(self.outer_cone_angle):
            raise ValueError("outer_cone_angle must be finite")
        if not 0.0 <= self.inner_cone_angle <= self.outer_cone_angle:
            raise ValueError("inner_cone_angle must be in [0, outer_cone_angle]")
        if not 0.0 < self.outer_cone_angle <= math.pi / 2.0:
            raise ValueError("outer_cone_angle must be in (0, pi/2]")
