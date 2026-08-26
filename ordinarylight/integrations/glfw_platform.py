"""Select pyGLFW's native-library variant before importing GLFW."""

import importlib
import os
import sys


def load_glfw(*, default_linux="x11"):
    """Load GLFW with an explicit, environment-overridable window platform."""
    default = default_linux if sys.platform.startswith("linux") else "native"
    platform = os.environ.get("WAVE_RENDER_GLFW_PLATFORM", default).lower()
    if platform not in {"native", "x11", "wayland"}:
        raise ValueError(
            "WAVE_RENDER_GLFW_PLATFORM must be native, x11, or wayland"
        )
    if platform in {"x11", "wayland"}:
        os.environ.setdefault("PYGLFW_LIBRARY_VARIANT", platform)
    glfw = importlib.import_module("glfw")
    if platform == "x11":
        glfw.init_hint(glfw.PLATFORM, glfw.PLATFORM_X11)
    elif platform == "wayland":
        glfw.init_hint(glfw.PLATFORM, glfw.PLATFORM_WAYLAND)
    return glfw
