"""Canonical capture must reproject to old samples, including subpixel offsets."""
from unittest.mock import Mock

import numpy as np
import pytest

import ordinarylight as ol
from ordinarylight.raster import camera_matrix
from ordinarylight.targets.vulkan.api import _VulkanGlobalIlluminationEngine


@pytest.mark.parametrize("camera_shift,object_shift", [(0, 0), (0.3, 0), (0, 0.2)])
def test_capture_motion_projects_previous_world_position(camera_shift, object_shift):
    engine = object.__new__(_VulkanGlobalIlluminationEngine)
    scene = ol.Scene()
    vertices = np.array([[-1, -1, 0], [1, -1, 0], [0, 1, 0]], np.float32)
    mesh = scene.add_mesh(vertices, np.array([[0, 1, 2]], np.uint32), ol.Material())
    old_camera = ol.PerspectiveCamera(position=(0, 0, -7), target=(0, 0, 0))
    camera = ol.PerspectiveCamera(position=(camera_shift, 0, -7),
                                  target=(camera_shift, 0, 0))
    width, height = 2, 1
    engine._output_history = engine._capture_motion_state(scene, old_camera, (width, height))
    scene.update_instance_transforms((mesh,), (ol.Transform.translation((object_shift, 0, 0)),))
    barycentric = np.array([[[0.2, 0.3], [0.3, 0.4]]], np.float32)
    weights = np.concatenate((1-barycentric.sum(-1, keepdims=True), barycentric), -1)
    old_positions = weights @ vertices
    positions = old_positions + [object_shift, 0, 0]
    raw = dict(primitive_id=np.zeros((height, width), np.uint32),
               normal=np.tile([0, 0, 1], (height, width, 1)),
               primary_position=positions, primary_barycentric=barycentric,
               path_state={key: np.ones((height, width, 4), np.float32) for key in
                           ("diffuse_radiance_hit_distance", "specular_radiance_hit_distance")})
    engine.capture_denoiser_raw = Mock(return_value=raw)
    signal = engine.capture_denoiser_signals(scene, camera, width, height, frame_index=1)
    points = np.concatenate((old_positions, np.ones((height, width, 1))), -1)
    clip = points @ camera_matrix(old_camera, width, height).T
    ndc = clip[..., :2] / clip[..., 3:4]
    old_pixels = (ndc * [0.5, -0.5] + 0.5) * [width, height] - 0.5
    np.testing.assert_allclose(signal.motion + [[[0, 0], [1, 0]]], old_pixels, atol=1e-6)
    assert not signal.frame.camera_cut
