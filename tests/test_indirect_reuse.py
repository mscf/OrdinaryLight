import math
import unittest
from pathlib import Path

from ordinarylight.integrations.indirect_reuse import (
    IndirectLightReservoir,
    IndirectLightSample,
    IndirectReservoirPlan,
    pack_indirect_reservoir,
    unpack_indirect_reservoir,
)
from ordinarylight.targets.vulkan.core import _motion_adaptive_history_limit


class IndirectReuseTests(unittest.TestCase):
    def test_history_budget_tracks_screen_space_motion(self):
        self.assertEqual(_motion_adaptive_history_limit(32, 0.0, 16.0), 32)
        self.assertEqual(_motion_adaptive_history_limit(32, 2.0, 16.0), 8)
        self.assertEqual(_motion_adaptive_history_limit(32, 8.0, 16.0), 2)
        self.assertEqual(_motion_adaptive_history_limit(32, 20.0, 16.0), 1)

    def test_half_resolution_4k_plan_fits_default_budget(self):
        plan = IndirectReservoirPlan(3840, 2160)
        self.assertEqual((plan.width, plan.height), (1920, 1080))
        self.assertAlmostEqual(plan.estimated_mib, 110.7421875)

    def test_full_resolution_4k_plan_is_rejected(self):
        with self.assertRaises(MemoryError):
            IndirectReservoirPlan(3840, 2160, scale=1.0)

    def test_reconnection_merge_applies_jacobian(self):
        sample = IndirectLightSample(
            2.0, 0.25, (1.0, 0.5, 0.25),
            (0.0, 1.0, 2.0), (0.0, 1.0, 0.0))
        source = IndirectLightReservoir()
        source.update(sample, 6.0, 0.0, represented_samples=3)
        current = IndirectLightReservoir()
        self.assertTrue(current.merge_reconnected(source, 1.0, 0.5, 0.0))
        self.assertEqual(current.weight_sum, 1.5)
        self.assertEqual(current.sample_count, 3)
        self.assertEqual(current.normalization, 0.5)

    def test_history_limit_preserves_normalization(self):
        sample = IndirectLightSample(
            2.0, 0.5, (1.0, 1.0, 1.0),
            (0.0, 1.0, 2.0), (0.0, 1.0, 0.0))
        reservoir = IndirectLightReservoir()
        reservoir.update(sample, 80.0, 0.0, represented_samples=80)
        normalization = reservoir.normalization
        self.assertTrue(reservoir.limit_history(32))
        self.assertEqual(reservoir.sample_count, 32)
        self.assertEqual(reservoir.weight_sum, 32.0)
        self.assertEqual(reservoir.normalization, normalization)
        self.assertFalse(reservoir.limit_history(32))
        with self.assertRaises(ValueError):
            reservoir.limit_history(128)

    def test_invalid_candidate_and_plan_are_rejected(self):
        with self.assertRaises(ValueError):
            IndirectLightSample(
                1.0, 0.0, (1.0, 1.0, 1.0),
                (0.0, 0.0, 0.0), (0.0, 1.0, 0.0))
        with self.assertRaises(ValueError):
            IndirectLightSample(
                1.0, 1.0, (1.0, 1.0, 1.0),
                (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
        with self.assertRaises(ValueError):
            IndirectReservoirPlan(1, 1, bytes_per_reservoir=20)
        with self.assertRaises(ValueError):
            IndirectReservoirPlan(1, 1, bytes_per_seed=3)
        with self.assertRaises(ValueError):
            IndirectReservoirPlan(1, 1, scale=0.1)

    def test_compact_abi_round_trip_stays_within_error_bounds(self):
        sample = IndirectLightSample(
            3.25, 0.375, (8.0, 1.5, 0.125),
            (102.25, -18.5, 4.75), (0.2, 0.9, -0.3))
        reservoir = IndirectLightReservoir()
        reservoir.update(sample, 7.5, 0.0, represented_samples=9)
        origin = (100.0, -20.0, 5.0)
        packed = pack_indirect_reservoir(reservoir, origin)
        self.assertEqual(len(packed), 24)
        decoded = unpack_indirect_reservoir(packed, origin)
        self.assertEqual(decoded.sample_count, 9)
        self.assertAlmostEqual(decoded.weight_sum, 7.5, places=2)
        self.assertAlmostEqual(decoded.sample.target, 3.25, places=2)
        self.assertAlmostEqual(decoded.sample.proposal_pdf, 0.375, places=3)
        for actual, expected in zip(
            decoded.sample.secondary_position, sample.secondary_position
        ):
            self.assertAlmostEqual(actual, expected, places=2)
        normal_dot = sum(
            actual * expected
            for actual, expected in zip(
                decoded.sample.secondary_normal, sample.secondary_normal)
        ) / math.sqrt(sum(value * value for value in sample.secondary_normal))
        self.assertGreater(normal_dot, 0.999)
        for actual, expected in zip(decoded.sample.radiance, sample.radiance):
            self.assertLess(abs(actual - expected), max(expected * 0.01, 0.002))

    def test_empty_compact_reservoir_round_trips(self):
        packed = pack_indirect_reservoir(IndirectLightReservoir())
        decoded = unpack_indirect_reservoir(packed)
        self.assertIsNone(decoded.sample)
        self.assertEqual(decoded.sample_count, 0)

    def test_glsl_abi_uses_six_raw_words(self):
        shader = (
            Path(__file__).parents[1]
            / "ordinarylight" / "shaders" / "wavefront_indirect_reuse.glsl"
        ).read_text()
        self.assertRegex(shader, r"reservoir_index\s*\*\s*(?:6u|uint\(6\))")
        self.assertIn("packHalf2x16(relative_position.xy)", shader)
        self.assertIn("packUnorm2x16", shader)
        self.assertIn("indirectPackRgb9e5", shader)
        self.assertIn("indirectUnpackRgb9e5", shader)
        self.assertIn("loadIndirectLightReservoir", shader)
        self.assertIn("storeIndirectLightReservoir", shader)

    def test_vulkan_storage_is_budgeted_and_resize_owned(self):
        backend = (
            Path(__file__).parents[1] / "ordinarylight" / "targets" / "vulkan" / "core.py"
        ).read_text()
        self.assertIn("IndirectReservoirPlan(", backend)
        self.assertIn('"wavefront_indirect_reservoir_buffer"', backend)
        self.assertIn('"wavefront_indirect_reservoir_extent"', backend)
        self.assertIn("wavefront_indirect_reservoir_bytes", backend)
        self.assertIn('"wavefront_indirect_seed_buffer"', backend)
        self.assertIn("wavefront_indirect_seed_bytes", backend)

    def test_vulkan_storage_has_isolated_one_time_clear_pass(self):
        root = Path(__file__).parents[1]
        backend = (root / "ordinarylight" / "targets" / "vulkan" / "core.py").read_text()
        shader = (
            root / "ordinarylight" / "shaders"
            / "wavefront_indirect_clear.comp"
        ).read_text()
        self.assertIn("indirect_reuse_clear_layout", backend)
        self.assertIn("record_indirect_reuse_clear", backend)
        self.assertIn("wavefront_indirect_reservoir_initialized", backend)
        self.assertRegex(
            shader, r"reservoir_index\s*\*\s*(?:6u|uint\(6\))",
        )
        self.assertIn("reservoir_count", shader)

    def test_candidate_generation_is_separate_and_screen_space_bounded(self):
        root = Path(__file__).parents[1]
        backend = (root / "ordinarylight" / "targets" / "vulkan" / "core.py").read_text()
        shader = (
            root / "ordinarylight" / "shaders"
            / "wavefront_indirect_candidates.comp"
        ).read_text()
        self.assertIn("wavefront_indirect_reuse_candidates", backend)
        self.assertIn("record_indirect_reuse_candidates", backend)
        self.assertIn(
            "or self.core.config.wavefront_indirect_reuse_candidates",
            backend,
        )
        self.assertIn("push.reservoir_width", shader)
        self.assertIn("candidatePrimaryWorldPosition", shader)
        self.assertIn("storeIndirectLightReservoir", shader)
        self.assertIn("candidateReprojectPrevious", shader)
        self.assertIn("previous_material", shader)
        self.assertIn("reuse_weight", shader)
        self.assertIn("candidateSpatialCompatibility", shader)
        self.assertIn("for (int neighbor = 0; neighbor < 4", shader)
        self.assertIn("uint counters[]", shader)
        self.assertIn("candidateInstrumentedPixel", shader)
        self.assertIn("reservoir.sample_count > push.history_limit", shader)
        self.assertIn("read_indirect_reuse_counters", backend)
        debug_shader = (
            root / "ordinarylight" / "shaders"
            / "wavefront_indirect_debug.comp"
        ).read_text()
        self.assertIn("indirectAcceptanceColor", debug_shader)
        self.assertIn("reservoir.debug_flags", debug_shader)
        self.assertIn("candidateRejectionDebugFlag", shader)
        self.assertIn("0x007fff00u", (
            root / "ordinarylight" / "shaders"
            / "wavefront_indirect_reuse.glsl"
        ).read_text())
        self.assertIn("record_indirect_reuse_debug", backend)
        self.assertIn("wavefront_indirect_reuse_apply", backend)
        self.assertIn("float normalization", debug_shader)
        self.assertIn("reservoir.selected.target", debug_shader)
        self.assertIn("indirectCorrection", debug_shader)
        self.assertIn("base + ivec2(x, y)", debug_shader)
        self.assertIn("result.correction * filter_weight", debug_shader)
        self.assertIn("result.confidence", debug_shader)
        self.assertNotIn("mix(current, reconstructed", debug_shader)


if __name__ == "__main__":
    unittest.main()
