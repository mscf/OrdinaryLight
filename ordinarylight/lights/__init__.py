"""Analytic and environment light resources."""

from .directional_light import DirectionalLight
from .environment_light import EnvironmentLight
from .point_light import PointLight
from .spot_light import SpotLight

POINT = 0
DIRECTIONAL = 1
SPOT = 2

Light = PointLight | DirectionalLight | SpotLight
LIGHT_TYPES = (PointLight, DirectionalLight, SpotLight)

__all__ = [
    "DIRECTIONAL", "LIGHT_TYPES", "Light", "POINT", "SPOT",
    "DirectionalLight", "EnvironmentLight", "PointLight", "SpotLight",
]
