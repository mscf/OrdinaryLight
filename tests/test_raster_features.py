import unittest

import numpy as np

import ordinarylight as ol


class _Backend:
    available_outputs = ("color",)
    config = None
    device = None
    last_timings = {"total_ms": 1.0}

    def __init__(self, value): self.value = value; self.closed = False
    def render_frame(self, _scene, _camera, width, height, **_kwargs):
        return np.full((height, width, 4), self.value, np.float32)
    def close(self): self.closed = True


def _camera():
    return ol.PerspectiveCamera((0, 0, 4), (0, 0, 0))


class RasterFeatureTests(unittest.TestCase):
    def test_shared_material_texture_is_evaluated(self):
        pixels = np.array([[[255, 0, 0, 255], [0, 255, 0, 255]]], np.uint8)
        material = ol.Material(
            base_color=(1, 1, 1), base_color_texture=ol.Texture(pixels),
        )
        mesh = ol.Mesh(
            [[-1, -1, 0], [1, -1, 0], [0, 1, 0]], [[0, 1, 2]], material,
            texcoords=[[0, 0], [0.99, 0], [0, 0]],
        )
        packed = ol.scene_mesh(
            ol.Scene([mesh]), _camera(), 64, 64,
            ol.RasterConfig(direct_lighting=False),
        )
        self.assertGreater(packed.vertices[0, 4], 0.9)
        self.assertGreater(packed.vertices[1, 5], 0.9)

    def test_direct_light_and_hard_shadow_are_composable(self):
        receiver = ol.Mesh(
            [[-1, -1, 0], [1, -1, 0], [0, 1, 0]], [[0, 1, 2]],
            ol.Material(base_color=(1, 1, 1)),
        )
        blocker = ol.Mesh(
            [[-2, -2, 1], [2, -2, 1], [0, 2, 1]], [[0, 1, 2]],
            ol.Material(),
        )
        light = ol.PointLight((0, 0, 2), intensity=8)
        scene = ol.Scene([receiver, blocker], [light])
        lit = ol.scene_mesh(scene, _camera(), 64, 64, ol.RasterConfig(ambient_light=0, shadows=False))
        shadowed = ol.scene_mesh(scene, _camera(), 64, 64, ol.RasterConfig(ambient_light=0, shadows=True))
        self.assertGreater(float(lit.vertices[:3, 4:7].mean()), 0.1)
        self.assertLess(float(shadowed.vertices[:3, 4:7].mean()), 1e-5)

    def test_temporal_history_only_accumulates_compatible_frames(self):
        config = ol.RasterConfig(temporal_history=True, temporal_weight=0.5)
        post = ol.RasterPostProcessor(config)
        scene = ol.Scene(); camera = _camera()
        first = post.process(np.zeros((2, 2, 4), np.float32), scene, camera)
        second = post.process(np.ones((2, 2, 4), np.float32), scene, camera)
        self.assertEqual(post.accumulated_frames, 2)
        np.testing.assert_allclose(second, 0.5)
        moved = ol.PerspectiveCamera((1, 0, 4), (0, 0, 0))
        third = post.process(np.ones((2, 2, 4), np.float32), scene, moved)
        self.assertEqual(post.accumulated_frames, 1)
        np.testing.assert_allclose(third, 1.0)

    def test_render_graph_declares_multipass_dependencies(self):
        pipeline = ol.create_raster_pipeline(ol.RasterConfig(temporal_history=True))
        self.assertEqual(pipeline.stage_names, ("geometry", "shadows", "lighting", "temporal", "post"))
        self.assertIn("output", pipeline.output_resources)

    def test_geometry_products_preserve_depth_normal_and_object_id(self):
        mesh = ol.Mesh(
            [[-0.5, -0.5, 0], [0.5, -0.5, 0], [0, 0.5, 0]], [[0, 1, 2]],
        )
        scene = ol.Scene([mesh])
        packed = ol.scene_mesh(scene, _camera(), 32, 24)
        products = ol.rasterize_geometry_products(packed, 32, 24)
        mask = products["depth"] < 1.0
        self.assertGreater(np.count_nonzero(mask), 10)
        self.assertTrue(np.all(products["object_id"][mask] == mesh.id))
        self.assertGreater(float(np.linalg.norm(products["normal"][mask], axis=1).mean()), 0.9)

    def test_volume_slice_geometry_uses_transfer_function_alpha(self):
        data = np.ones((2, 2, 2), np.float32)
        volume = ol.Volume(data)
        mesh = ol.scene_mesh(
            ol.Scene(volumes=[volume]), _camera(), 64, 64,
            ol.RasterConfig(volume_slices=3),
        )
        self.assertEqual(mesh.vertices.shape, (12, 12))
        self.assertGreater(float(mesh.vertices[:, 7].max()), 0.0)

    def test_hybrid_backend_composes_child_renderers(self):
        raster, lighting = _Backend(0.25), _Backend(0.5)
        renderer = ol.Renderer(backend=ol.HybridBackend(raster, lighting, weight=0.5))
        try:
            result = renderer.render(ol.Scene(), _camera(), (3, 2))
            np.testing.assert_allclose(result[..., :3], 0.5)
            np.testing.assert_allclose(result[..., 3], 0.25)
        finally:
            renderer.close()
        self.assertTrue(raster.closed and lighting.closed)

    def test_renderer_presents_to_array_surface(self):
        renderer = ol.Renderer(backend=_Backend(0.5))
        surface = ol.ArraySurface(3, 2)
        try:
            result = renderer.render_to(ol.Scene(), _camera(), surface)
        finally:
            renderer.close()
        self.assertEqual(result.dtype, np.uint8)
        self.assertEqual(int(result[0, 0, 0]), 128)


if __name__ == "__main__":
    unittest.main()
