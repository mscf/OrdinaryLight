"""Regression: nearby orthographic geometry must survive WebGPU clipping."""
import os
import numpy as np
import pytest
import ordinarylight as ol
from ordinarylight.raster import camera_matrix
from ordinarylight.renderers.raster.webgpu import _webgpu_projection


@pytest.mark.parametrize("camera", [
    ol.OrthographicCamera((0, 0, 0), (0, 0, -1), vertical_size=2),
    ol.PerspectiveCamera((0, 0, 0), (0, 0, -1)),
])
def test_webgpu_projection_maps_near_far_and_preserves_xy(camera):
    original = camera_matrix(camera, 32, 24)
    before = original.copy()
    matrix = _webgpu_projection(original)
    points = np.array(((0, 0, -0.01, 1), (0, 0, -10000, 1), (0.1, 0.1, -2, 1)))
    clip = points @ matrix.T
    ndc = clip[:, :3] / clip[:, 3:]
    np.testing.assert_allclose(ndc[:2, 2], (0, 1), atol=1e-6)
    old_clip = points @ original.T
    np.testing.assert_allclose(ndc[:, :2], old_clip[:, :2] / old_clip[:, 3:])
    assert 0 < ndc[2, 2] < 1
    np.testing.assert_array_equal(original, before)
    recovered = clip @ np.linalg.inv(matrix).T
    np.testing.assert_allclose(recovered[2] / recovered[2, 3], points[2], atol=1e-4)


@pytest.mark.skipif(os.environ.get("ORDINARYLIGHT_WEBGPU_TESTS") != "1", reason="opt-in WebGPU device test")
def test_orthographic_color_and_geometry_products_agree():
    scene = ol.Scene()
    scene.add_mesh(np.array(((-1, -1, 0), (1, -1, 0), (1, 1, 0), (-1, 1, 0)), np.float32),
                   np.array(((0, 1, 2), (0, 2, 3)), np.uint32),
                   ol.Material(base_color=(0.2, 0.4, 0.6)))
    camera = ol.OrthographicCamera((0, 0, 10), (0, 0, 0), vertical_size=4)
    config = ol.RasterConfig(shadows=False, direct_lighting=False, material_program=ol.unlit_material)
    renderer = ol.renderers.raster.WebGpuRasterRenderer(
        ol.RasterProgram.scene(target="wgsl", validate=False, material_programs=(ol.unlit_material,)), config=config)
    try:
        frame = renderer.render_frame(scene, camera, 16, 16)
        np.testing.assert_allclose(frame[8, 8, :3], (0.2, 0.4, 0.6), atol=1e-3)
        products = renderer.render_products(scene, camera, 16, 16, outputs=("color", "depth", "normal", "motion"))
        np.testing.assert_allclose(products["color"], frame, atol=1e-3)
        assert 9.9 < products["depth"][8, 8] < 10.1
        np.testing.assert_allclose(products["normal"][8, 8], (0, 0, 1), atol=1e-4)
        np.testing.assert_allclose(products["motion"][8, 8], (0, 0), atol=1e-4)
        again = renderer.render_products(scene, camera, 16, 16, outputs=("motion",))
        np.testing.assert_allclose(again["motion"][8, 8], (0, 0), atol=1e-4)
    finally:
        renderer.close()
