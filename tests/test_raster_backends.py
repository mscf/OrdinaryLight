import sys
import shutil
import importlib.util
import os
import unittest
from pathlib import Path

import numpy as np

import ordinarylight as ol


shade_root = Path(__file__).parents[2] / "ordinaryshade"
if shade_root.exists():
    sys.path.insert(0, str(shade_root))
try:
    import ordinaryshade as osh
except ImportError as error:
    raise unittest.SkipTest("raster compiler tests require sibling ordinaryshade") from error


@osh.vertex
def raster_vertex(position: osh.location(osh.vec2, 0)) -> osh.builtin(osh.vec4, "position"):
    return osh.vec4(position, 0.0, 1.0)


@osh.fragment
def raster_fragment() -> osh.location(osh.vec4, 0):
    return osh.vec4(0.95, 0.45, 0.15, 1.0)


class RasterBackendTests(unittest.TestCase):
    @staticmethod
    def _overlap_scene():
        vertices = [[-0.8, -0.8, 0], [0.8, -0.8, 0], [0, 0.8, 0]]
        near = ol.Mesh(vertices, [[0, 1, 2]], ol.Material(base_color=(1, 0, 0)),
                       transform=ol.Transform.translation((0, 0, 1)))
        far = ol.Mesh(vertices, [[0, 1, 2]], ol.Material(base_color=(0, 0, 1)))
        return ol.Scene([near, far]), ol.PerspectiveCamera((0, 0, 4), (0, 0, 0))
    def test_raster_program_compiles_for_both_backends(self):
        target = "spirv" if shutil.which("glslangValidator") else "glsl"
        vulkan = ol.RasterProgram.compile(raster_vertex, raster_fragment, target=target)
        webgpu = ol.RasterProgram.compile(raster_vertex, raster_fragment, target="wgsl", validate=False)
        self.assertTrue(vulkan.vertex.binary if target == "spirv" else vulkan.vertex.source)
        self.assertTrue(webgpu.vertex.source.startswith("@vertex"))
        self.assertEqual(vulkan.reflection.vertex.stage, "vertex")

    def test_builtin_scene_program_compiles_for_both_backends(self):
        target = "spirv" if shutil.which("glslangValidator") else "glsl"
        native = ol.RasterProgram.scene(target=target)
        web = ol.RasterProgram.scene(target="wgsl", validate=False)
        self.assertEqual(len(native.reflection.varyings), 1)
        self.assertIn("@location(1)", web.vertex.source)

    def test_raster_mesh_owns_contiguous_typed_arrays(self):
        mesh = ol.RasterMesh([[0, 0], [1, 0], [0, 1]], [0, 1, 2])
        self.assertEqual(mesh.vertices.dtype, np.float32)
        self.assertEqual(mesh.indices.dtype, np.uint32)
        self.assertTrue(mesh.vertices.flags.c_contiguous)

    def test_general_interleaved_vertex_layout(self):
        layout = ol.RasterVertexLayout(28, (
            ol.RasterVertexAttribute(0, "float32x4", 0, "position"),
            ol.RasterVertexAttribute(1, "float32x3", 16, "color"),
        ))
        mesh = ol.RasterMesh(np.zeros((3, 7), np.float32), [0, 1, 2], layout)
        self.assertEqual(mesh.layout.stride, 28)
        self.assertEqual(mesh.layout.attributes[1].location, 1)

    def test_scene_adapter_applies_camera_and_object_transform(self):
        source = ol.Mesh(
            [[-0.5, -0.5, 0], [0.5, -0.5, 0], [0, 0.5, 0]],
            [[0, 1, 2]], ol.Material(base_color=(0.2, 0.4, 0.8)),
            transform=ol.Transform.translation((1, 0, 0)),
        )
        scene = ol.Scene([source])
        camera = ol.PerspectiveCamera((0, 0, 4), (0, 0, 0))
        mesh = ol.scene_mesh(scene, camera, 100, 100)
        self.assertEqual(mesh.vertices.shape, (3, 7))
        np.testing.assert_allclose(
            mesh.vertices[:, 4:], np.tile((0.2, 0.4, 0.8), (3, 1)),
        )
        self.assertGreater(mesh.vertices[:, 0].mean(), 0.0)

    def test_raster_state_validation(self):
        state = ol.RasterState(cull_mode="none", depth_compare="less-equal")
        self.assertTrue(state.depth_test)
        with self.assertRaises(ValueError):
            ol.RasterState(blend_mode="multiply")

    @unittest.skipUnless(
        os.environ.get("ORDINARYLIGHT_RUN_GPU_GATES") == "1",
        "GPU raster validation is opt-in",
    )
    def test_webgpu_backend_draws_offscreen_when_available(self):
        if importlib.util.find_spec("wgpu") is None:
            self.skipTest("wgpu is unavailable")
        program = ol.RasterProgram.compile(
            raster_vertex, raster_fragment, target="wgsl", validate=False,
        )
        backend = ol.WebGpuRasterBackend(program)
        try:
            image = backend.render(ol.triangle_mesh(), 64, 48)
            self.assertEqual(image.shape, (48, 64, 4))
            self.assertGreater(np.count_nonzero(image[..., 0] > 128), 100)
        finally:
            backend.close()

    @unittest.skipUnless(
        os.environ.get("ORDINARYLIGHT_RUN_GPU_GATES") == "1",
        "GPU raster validation is opt-in",
    )
    def test_vulkan_backend_draws_offscreen_when_available(self):
        if importlib.util.find_spec("vulkan") is None:
            self.skipTest("vulkan is unavailable")
        program = ol.RasterProgram.compile(
            raster_vertex, raster_fragment, target="spirv",
        )
        backend = ol.VulkanRasterBackend(program)
        try:
            image = backend.render(ol.triangle_mesh(), 64, 48)
            self.assertEqual(image.shape, (48, 64, 4))
            self.assertGreater(np.count_nonzero(image[..., 0] > 128), 100)
        finally:
            backend.close()

    @unittest.skipUnless(
        os.environ.get("ORDINARYLIGHT_RUN_GPU_GATES") == "1",
        "GPU raster validation is opt-in",
    )
    def test_scene_rendering_and_depth_match_backend_contract(self):
        scene, camera = self._overlap_scene()
        choices = []
        if importlib.util.find_spec("vulkan") is not None:
            choices.append(ol.VulkanRasterBackend(
                ol.RasterProgram.scene(target="spirv"),
                state=ol.RasterState(cull_mode="none"),
            ))
        if importlib.util.find_spec("wgpu") is not None:
            choices.append(ol.WebGpuRasterBackend(
                ol.RasterProgram.scene(target="wgsl", validate=False),
                state=ol.RasterState(cull_mode="none"),
            ))
        if not choices:
            self.skipTest("no GPU raster backend is available")
        for backend in choices:
            with self.subTest(backend=backend.capabilities.backend):
                renderer = ol.Renderer(backend=backend)
                try:
                    image = renderer.render(scene, camera, (96, 64))
                    self.assertEqual(image.dtype, np.float32)
                    self.assertEqual(image.shape, (64, 96, 4))
                    center = image[32, 48, :3]
                    self.assertGreater(center[0], 0.8)
                    self.assertLess(center[2], 0.2)
                finally:
                    renderer.close()
