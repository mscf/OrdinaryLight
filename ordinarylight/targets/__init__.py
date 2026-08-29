"""Execution targets used by renderer implementations.

Targets describe platform APIs and device support. Rendering algorithms live
under :mod:`ordinarylight.renderers` and may support one or more targets.
"""

from .base import ExecutionTargetInfo
from . import cpu, webgpu


def __getattr__(name):
    if name == "vulkan":
        from importlib import import_module

        module = import_module(f"{__name__}.vulkan")
        globals()[name] = module
        return module
    raise AttributeError(name)


__all__ = ["ExecutionTargetInfo", "cpu", "vulkan", "webgpu"]
