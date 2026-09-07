"""The glass motion controls move one subject and hold its end pose."""
import numpy as np
import pytest

from ordinarylight.showcases.catalog.rooms import SHOWCASES
from ordinarylight.showcases.glass_detail import animate_glass_detail


@pytest.mark.parametrize('identifier,moving_glass', [
    ('glass-detail-motion', True), ('glass-detail-target', False),
])
def test_motion_is_isolated_and_holds(identifier, moving_glass):
    item = next(item for item in SHOWCASES if item.id == identifier)
    scene = item.build()
    initial = [mesh.transform.matrix.copy() for mesh in scene.meshes]
    animate_glass_detail(scene, 1.0)
    moved = [mesh.transform.matrix.copy() for mesh in scene.meshes]
    revision = scene.revision
    animate_glass_detail(scene, 2.0)
    assert scene.revision == revision
    for mesh, before, after in zip(scene.meshes, initial, moved):
        np.testing.assert_array_equal(mesh.transform.matrix, after)
        selected = (mesh.name == 'glass-sphere') == moving_glass
        assert bool(np.any(before != after)) == selected
    assert item.animate is animate_glass_detail
    assert item.renderer['denoiser_enabled']


def test_camera_mode_has_static_geometry():
    item = next(item for item in SHOWCASES if item.id == 'glass-detail-camera')
    assert item.animate is None  # Viewer animates the camera only.
    assert not item.build().animations
