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


def _color(value):
    color = tuple(float(component) for component in value)
    if len(color) != 3 or not all(math.isfinite(component) for component in color):
        raise ValueError("color must contain three finite values")
    if any(component < 0.0 or component > 1.0 for component in color):
        raise ValueError("color components must be in [0, 1]")
    return color


def _unit(value, name):
    value = float(value)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return value


@dataclass(frozen=True, slots=True)
class Tint(ObjectEffect):
    """Blend the visible object surface toward ``color``."""

    color: tuple[float, float, float] = (1.0, 0.45, 0.05)
    strength: float = 0.5

    def __post_init__(self):
        object.__setattr__(self, "color", _color(self.color))
        object.__setattr__(self, "strength", _unit(self.strength, "strength"))


@dataclass(frozen=True, slots=True)
class EmissiveHighlight(ObjectEffect):
    """Add a display-space glow color to the visible object surface."""

    color: tuple[float, float, float] = (1.0, 0.45, 0.05)
    strength: float = 0.5

    def __post_init__(self):
        object.__setattr__(self, "color", _color(self.color))
        object.__setattr__(self, "strength", _unit(self.strength, "strength"))


@dataclass(frozen=True, slots=True)
class Isolation(ObjectEffect):
    """Keep the target unchanged while dimming all other visible surfaces."""

    dimming: float = 0.75

    def __post_init__(self):
        object.__setattr__(self, "dimming", _unit(self.dimming, "dimming"))


@dataclass(frozen=True, slots=True)
class BoundingBox(ObjectEffect):
    """Draw the object's projected world-space bounds through occluders."""

    color: tuple[float, float, float] = (1.0, 0.45, 0.05)
    width: int = 2

    def __post_init__(self):
        object.__setattr__(self, "color", _color(self.color))
        if isinstance(self.width, bool) or not isinstance(self.width, int):
            raise TypeError("width must be an int")
        if not 1 <= self.width <= 32:
            raise ValueError("width must be between 1 and 32 pixels")


@dataclass(frozen=True, slots=True)
class XRay(ObjectEffect):
    """Show translucent projected object bounds even while occluded."""

    color: tuple[float, float, float] = (0.1, 0.8, 1.0)
    strength: float = 0.2
    width: int = 2

    def __post_init__(self):
        object.__setattr__(self, "color", _color(self.color))
        object.__setattr__(self, "strength", _unit(self.strength, "strength"))
        if isinstance(self.width, bool) or not isinstance(self.width, int):
            raise TypeError("width must be an int")
        if not 1 <= self.width <= 32:
            raise ValueError("width must be between 1 and 32 pixels")


__all__ = [
    "BoundingBox", "EmissiveHighlight", "Isolation", "ObjectEffect",
    "Outline", "Tint", "XRay",
]
