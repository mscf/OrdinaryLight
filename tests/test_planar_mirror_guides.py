import numpy as np

import ordinarylight as ol
from ordinarylight.raster import camera_matrix
from tools.denoiser_motion.planar_mirror import reflect_z, reflected_camera, prepare_signal
from tests.test_denoising_signals import signals


def test_reflection_camera_projection_reverses_only_screen_x():
    camera = ol.PerspectiveCamera(position=(0.4, 2, -7), target=(0, 1.8, 0))
    reflected = reflected_camera(camera)
    points = np.array([[-1, 1, -3], [1.4, 0.9, -1.8], [0.2, 2.5, -2]], np.float32)
    np.testing.assert_array_equal(reflect_z(reflect_z(points)), points)
    virtual = np.column_stack((reflect_z(points), np.ones(3)))
    physical = np.column_stack((points, np.ones(3)))
    a = virtual @ camera_matrix(camera, 160, 120).T
    b = physical @ camera_matrix(reflected, 160, 120).T
    a = a[:, :3] / a[:, 3:]
    b = b[:, :3] / b[:, 3:]
    np.testing.assert_allclose(a[:, 0], -b[:, 0], atol=1e-6)
    np.testing.assert_allclose(a[:, 1:], b[:, 1:], atol=1e-6)


def test_optical_guides_flip_motion_and_mask_without_changing_radiance():
    source = signals(width=5, height=3)
    source.motion[..., 0] = np.arange(5)
    source.motion[..., 1] = 2
    mask = np.ones((3, 5), bool)
    mask[0, 0] = False
    radiance = np.ones((3, 5, 3), np.float32)
    result = prepare_signal(source, radiance, mask, optical=True)
    np.testing.assert_array_equal(result.motion[1, :, 0], [-4, -3, -2, -1, 0])
    np.testing.assert_array_equal(result.motion[1, :, 1], 2)
    np.testing.assert_array_equal(result.specular_radiance_hit_distance[1, :, :3], 1)
    assert result.view_z[0, 0] == 0
    assert result.material_id[0, 0] == np.uint32(0xffffffff)
    np.testing.assert_array_equal(source.motion[1, :, 0], np.arange(5))


def test_object_motion_updates_scene_revision_and_stops_cleanly():
    from tools.denoiser_motion.planar_mirror import fixture, move_object
    scene, moving, _ = fixture(False)
    before = scene.transform_revision
    move_object(scene, moving, 1.5)
    assert scene.transform_revision > before
    np.testing.assert_array_equal(moving.transform.matrix[:3, 3], [1.5, 0, 0])
    stopped = scene.transform_revision
    move_object(scene, moving, 1.5)
    assert scene.transform_revision == stopped
