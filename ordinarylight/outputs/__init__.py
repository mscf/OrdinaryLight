"""Headless image conversion and encoded output sinks."""

from .image import to_sdr
from .video import FFmpegVideoWriter


def __getattr__(name):
    if name == "NvencVideoWriter":
        from .nvenc import NvencVideoWriter
        return NvencVideoWriter
    raise AttributeError(name)

__all__ = ["FFmpegVideoWriter", "NvencVideoWriter", "to_sdr"]
