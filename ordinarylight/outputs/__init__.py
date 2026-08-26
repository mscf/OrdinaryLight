"""Headless image conversion and encoded output sinks."""

from .image import to_sdr
from .video import FFmpegVideoWriter

__all__ = ["FFmpegVideoWriter", "to_sdr"]
