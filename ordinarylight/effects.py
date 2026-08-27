"""Composable, renderer-side effects applied to individual scene objects."""

from __future__ import annotations

from dataclasses import dataclass
import math


class ObjectEffect:
    """Base class for transient visual effects attached to scene objects.

    Effects do not represent application selection state. Applications decide
    what a :class:`~ordinarylight.selection.PickResult` means and may apply an
    effect, mutate scene data, update another UI, or perform any other action.
    """


@dataclass(frozen=True, slots=True)
class Outline(ObjectEffect):
    """Draw a screen-space outline around an object's visible silhouette."""

    color: tuple[float, float, float] = (1.0, 0.45, 0.05)
    width: int = 2

    def __post_init__(self):
        color = tuple(float(value) for value in self.color)
        if len(color) != 3 or not all(math.isfinite(value) for value in color):
            raise ValueError("color must contain three finite values")
        if any(value < 0.0 or value > 1.0 for value in color):
            raise ValueError("color components must be in [0, 1]")
        if isinstance(self.width, bool) or not isinstance(self.width, int):
            raise TypeError("width must be an int")
        if not 1 <= self.width <= 32:
            raise ValueError("width must be between 1 and 32 pixels")
        object.__setattr__(self, "color", color)


__all__ = ["ObjectEffect", "Outline"]
