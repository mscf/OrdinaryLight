"""Directional-light resource."""

from dataclasses import dataclass, field

from ._validation import validate_color_intensity, validate_direction


@dataclass
class DirectionalLight:
    direction: tuple[float, float, float]
    color: tuple[float, float, float] = (1.0, 1.0, 1.0)
    intensity: float = 1.0
    id: int | None = field(default=None, init=False)

    def __post_init__(self):
        validate_direction(self.direction)
        validate_color_intensity(self.color, self.intensity)
