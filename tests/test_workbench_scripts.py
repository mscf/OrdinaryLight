import textwrap
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import ordinarylight as ol
from ordinarylight.integrations.workbench import (
    OrbitCamera, Showcase, ShowcaseCatalog, discover_showcases,
)
from ordinarylight.integrations.qt_workbench import _default_showcase_paths


class WorkbenchScriptTests(unittest.TestCase):
    def test_packaged_catalog_is_discoverable(self):
        catalog = discover_showcases(_default_showcase_paths())
        self.assertGreaterEqual(len(catalog), 12)
        self.assertEqual(catalog["area-lights"].title, "Area lights")
        directional = catalog["raster-directional-shadows"]
        spot = catalog["raster-spot-shadows"]
        self.assertIn("raster-feature", directional.tags)
        self.assertEqual(directional.renderer["shadow_map_size"], 512)
        self.assertIsInstance(
            directional.create_scene().lights[0], ol.DirectionalLight,
        )
        self.assertIsInstance(spot.create_scene().lights[0], ol.SpotLight)

    def test_showcase_is_lazy_and_camera_is_fitted(self):
        calls = []

        def build():
            calls.append(True)
            scene = ol.Scene()
            scene.add_mesh(((0, 0, 0), (1, 0, 0), (0, 1, 0)), ((0, 1, 2),))
            return scene

        showcase = Showcase("triangle", "Triangle", build)
        self.assertFalse(calls)
        scene = showcase.create_scene()
        self.assertEqual(len(calls), 1)
        self.assertIsInstance(showcase.camera.camera(scene), ol.PerspectiveCamera)

    def test_scripts_are_discovered_without_sys_path_mutation(self):
        with TemporaryDirectory() as directory:
            script = Path(directory, "sample.py")
            script.write_text(textwrap.dedent("""
                import ordinarylight as ol
                from ordinarylight.integrations.workbench import Showcase

                def build():
                    scene = ol.Scene()
                    scene.add_mesh(((0, 0, 0), (1, 0, 0), (0, 1, 0)), ((0, 1, 2),))
                    return scene

                SHOWCASE = Showcase("sample", "Sample", build)
            """))
            catalog = discover_showcases((directory,))
            self.assertEqual(len(catalog), 1)
            self.assertEqual(catalog["sample"].title, "Sample")
            self.assertIsInstance(catalog[0].create_scene(), ol.Scene)

    def test_catalog_rejects_duplicate_ids(self):
        first = Showcase("same", "First", ol.Scene)
        second = Showcase("same", "Second", ol.Scene)
        with self.assertRaisesRegex(ValueError, "duplicate showcase id"):
            ShowcaseCatalog((first, second))


if __name__ == "__main__":
    unittest.main()
