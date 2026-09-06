"""Shared presentation policies for rendering backends."""

from .acquisition import (
    DEFAULT_ACQUIRE_TIMEOUT_NS,
    acquire_image,
    validate_acquire_timeout,
)

__all__ = ["DEFAULT_ACQUIRE_TIMEOUT_NS", "acquire_image", "validate_acquire_timeout"]
