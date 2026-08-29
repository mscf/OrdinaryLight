"""WebGPU execution target."""

from .base import ExecutionTargetInfo

info = ExecutionTargetInfo(name="webgpu", shader_format="wgsl", gpu=True)

__all__ = ["info"]
