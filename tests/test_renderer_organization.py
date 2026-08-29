import unittest
from pathlib import Path

import ordinarylight as ol


class RendererOrganizationTests(unittest.TestCase):
    def test_concrete_renderers_have_explicit_family_and_target(self):
        implementations = (
            ol.renderers.reference.CpuReferenceRenderer,
            ol.renderers.hybrid.HybridRenderer,
            ol.renderers.raster.VulkanRasterRenderer,
            ol.renderers.raster.WebGpuRasterRenderer,
            ol.renderers.gi.VulkanGlobalIlluminationRenderer,
        )
        for implementation in implementations:
            with self.subTest(implementation=implementation.__name__):
                self.assertTrue(
                    issubclass(implementation, ol.RendererImplementation)
                )
                info = implementation.implementation_info()
                self.assertTrue(info.name)
                self.assertIn(
                    info.family,
                    {"global_illumination", "raster", "hybrid", "reference"},
                )
                self.assertIn(
                    info.graphics_api,
                    {"vulkan", "webgpu", "cpu", "composite"},
                )

    def test_execution_targets_are_distinct_from_renderers(self):
        self.assertEqual(ol.targets.vulkan.info.shader_format, "spirv")
        self.assertEqual(ol.targets.webgpu.info.shader_format, "wgsl")
        self.assertFalse(ol.targets.cpu.info.gpu)

    def test_removed_implementation_namespace_is_not_exported(self):
        self.assertFalse(hasattr(ol, "backends"))

    def test_root_contains_only_package_entrypoints(self):
        root = Path(ol.__file__).parent
        root_modules = {path.name for path in root.glob("*.py")}
        self.assertEqual(root_modules, {"__init__.py", "__main__.py"})

    def test_camera_models_have_concrete_modules(self):
        self.assertEqual(
            ol.PerspectiveCamera.__module__,
            "ordinarylight.cameras.perspective_camera",
        )
        self.assertEqual(
            ol.OrthographicCamera.__module__,
            "ordinarylight.cameras.orthographic_camera",
        )


if __name__ == "__main__":
    unittest.main()
