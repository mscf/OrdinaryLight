import unittest

from scripts.compile_shaders import build_plan, load_manifest, validate_plan


class ShaderManifestTests(unittest.TestCase):
    def test_manifest_expands_to_unique_existing_inventory(self):
        manifest = load_manifest()
        plan = build_plan(manifest)
        self.assertEqual(len(plan), len({build.name for build in plan}))
        report = validate_plan(plan)
        self.assertFalse(report["missing_sources"])
        self.assertFalse(report["missing_outputs"])
        self.assertFalse(report["unmanaged_outputs"])
        self.assertIn("raster_scene.vert.spv", manifest["managed_outputs"])
        self.assertIn("raster_scene.frag.spv", manifest["managed_outputs"])

    def test_plan_contains_production_specializations(self):
        names = {build.name for build in build_plan()}
        self.assertIn("wavefront_primary.comp.spv", names)
        self.assertIn("wavefront_shade_ordinaryshade.comp.spv", names)
        self.assertIn("wavefront_reconstruct_bgra.comp.spv", names)
        self.assertIn(
            "wavefront_megakernel_native_opaque_untextured_production.comp.spv",
            names,
        )
        self.assertIn("ser_probe_ser.rgen.spv", names)


if __name__ == "__main__":
    unittest.main()
