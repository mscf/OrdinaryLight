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
    def test_raster_program_compiles_for_both_backends(self):
        target = "spirv" if shutil.which("glslangValidator") else "glsl"
        vulkan = ol.RasterProgram.compile(raster_vertex, raster_fragment, target=target)
        webgpu = ol.RasterProgram.compile(raster_vertex, raster_fragment, target="wgsl", validate=False)
        self.assertTrue(vulkan.vertex.binary if target == "spirv" else vulkan.vertex.source)
        self.assertTrue(webgpu.vertex.source.startswith("@vertex"))
        self.assertEqual(vulkan.reflection.vertex.stage, "vertex")

    def test_raster_mesh_owns_contiguous_typed_arrays(self):
        mesh = ol.RasterMesh([[0, 0], [1, 0], [0, 1]], [0, 1, 2])
        self.assertEqual(mesh.vertices.dtype, np.float32)
        self.assertEqual(mesh.indices.dtype, np.uint32)
        self.assertTrue(mesh.vertices.flags.c_contiguous)

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
