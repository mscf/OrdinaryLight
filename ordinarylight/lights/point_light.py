"""Point-light resource."""

from dataclasses import dataclass, field

from ._validation import validate_color_intensity, validate_range, vec3


@dataclass
class PointLight:
    position: tuple[float, float, float]
    color: tuple[float, float, float] = (1.0, 1.0, 1.0)
    intensity: float = 1.0
    range: float | None = None
    id: int | None = field(default=None, init=False)

    def __post_init__(self):
        vec3(self.position, "position")
        validate_color_intensity(self.color, self.intensity)
        validate_range(self.range)
