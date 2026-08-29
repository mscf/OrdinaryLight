"""Portable CPU execution target."""

from .base import ExecutionTargetInfo

info = ExecutionTargetInfo(name="cpu", shader_format=None, gpu=False)

__all__ = ["info"]
