"""A script-discoverable Qt workbench showcase."""

from ordinarylight.integrations.workbench import OrbitCamera, Showcase
from examples.common import scene_and_camera


def build():
    scene, _camera = scene_and_camera()
    return scene


SHOWCASE = Showcase(
    id="example-triangle",
    title="Example triangle",
    description="Minimal third-party workbench extension.",
    build=build,
    camera=OrbitCamera(target=(0, 0, 0), radius=4.0, height=0.5),
)
