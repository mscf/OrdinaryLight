import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class ShaderRngTests(unittest.TestCase):
    def test_stochastic_frame_seed_comes_from_live_camera_buffer(self):
        cases = {
            "ordinarylight/shaders/wavefront_generate.comp":
                "uint frame_index = uint((camera.camera_origin.w + 0.5));",
            "ordinarylight/shaders/wavefront_primary_impl.glsl":
                "uint frame_index = uint(camera.origin.w + 0.5);",
            "ordinarylight/shaders/wavefront_indirect_candidates.comp":
                "uint frame_index = uint((camera.origin.w + 0.5));",
        }
        for relative, expected in cases.items():
            with self.subTest(shader=relative):
                source = (ROOT / relative).read_text()
                self.assertIn(expected, source)

    def test_stochastic_seed_does_not_use_resident_push_frame(self):
        for relative in (
            "ordinarylight/shaders/wavefront_generate.comp",
            "ordinarylight/shaders/wavefront_primary_impl.glsl",
            "ordinarylight/shaders/wavefront_indirect_candidates.comp",
        ):
            with self.subTest(shader=relative):
                source = (ROOT / relative).read_text()
                self.assertNotIn("waveHash(push.tile_frame.z)", source)
                self.assertNotIn("hashValue(push.tile_frame.z)", source)
                self.assertNotIn("candidateRandom(pixel, push.frame_index", source)


if __name__ == "__main__":
    unittest.main()
