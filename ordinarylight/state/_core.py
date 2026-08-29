"""Backend-neutral renderer state values."""

from enum import Enum


class AccumulationState(str, Enum):
    """Current relationship between scene motion and temporal accumulation."""

    DISABLED = "disabled"
    MOVING = "moving"
    SETTLING = "settling"
    ACCUMULATING = "accumulating"


__all__ = ["AccumulationState"]
