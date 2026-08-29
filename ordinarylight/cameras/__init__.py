"""Camera models."""

from .orthographic_camera import OrthographicCamera
from .panoramic_camera import PanoramicCamera
from .perspective_camera import PerspectiveCamera

CAMERA_TYPES = (PerspectiveCamera, OrthographicCamera, PanoramicCamera)
Camera = PerspectiveCamera | OrthographicCamera | PanoramicCamera

__all__ = [
    "CAMERA_TYPES", "Camera", "OrthographicCamera", "PanoramicCamera",
    "PerspectiveCamera",
]
