"""Renderer-side object effects."""
from . import _core
globals().update({name: value for name, value in vars(_core).items() if not name.startswith("_")})
