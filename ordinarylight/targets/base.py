"""Execution-target metadata."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ExecutionTargetInfo:
    """Identity and shader representation for a graphics execution target."""

    name: str
    shader_format: str | None
    gpu: bool


__all__ = ["ExecutionTargetInfo"]
