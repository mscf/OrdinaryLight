import unittest
from pathlib import Path

from ordinarylight.integrations.restir_di import (
    DirectLightReservoir,
    DirectLightSample,
)
from ordinarylight.targets.vulkan.core import _restir_reservoir_storage_bytes


class DirectLightReservoirTests(unittest.TestCase):
    def test_independent_reservoir_storage_scales_by_stream_count(self):
        self.assertEqual(
            _restir_reservoir_storage_bytes(10, 20, 4),
            10 * 20 * 4 * 12,
        )
        self.assertEqual(
            _restir_reservoir_storage_bytes(
                10, 20, 4, stratified=True,
            ),
            10 * 20 * 4 * 20,
        )

    def test_weighted_stream_tracks_sum_count_and_normalization(self):
        reservoir = DirectLightReservoir()
        first = DirectLightSample(0, 0.2, 0.3, 2.0)
        second = DirectLightSample(1, 0.4, 0.1, 4.0)
        self.assertTrue(reservoir.update(first, 2.0, 0.9))
        self.assertTrue(reservoir.update(second, 6.0, 0.2))
        self.assertEqual(reservoir.sample, second)
        self.assertEqual(reservoir.weight_sum, 8.0)
        self.assertEqual(reservoir.sample_count, 2)
        self.assertEqual(reservoir.normalization, 1.0)

    def test_zero_weight_candidate_is_counted_but_not_selected(self):
        reservoir = DirectLightReservoir()
        sample = DirectLightSample(3, 0.1, 0.2, 0.0)
        self.assertFalse(reservoir.update(sample, 0.0, 0.0))
        self.assertIsNone(reservoir.sample)
        self.assertEqual(reservoir.sample_count, 1)
        self.assertEqual(reservoir.normalization, 0.0)

    def test_temporal_merge_reevaluates_target_and_preserves_m(self):
        previous = DirectLightReservoir()
        previous.update(DirectLightSample(2, 0.25, 0.5, 2.0), 6.0, 0.0)
        current = DirectLightReservoir()
        current.update(DirectLightSample(0, 0.1, 0.1, 1.0), 1.0, 0.0)
        self.assertTrue(current.merge(previous, 1.0, 0.0))
        self.assertEqual(current.sample.light_index, 2)
        self.assertEqual(current.sample.target, 1.0)
        self.assertEqual(current.weight_sum, 4.0)
        self.assertEqual(current.sample_count, 2)
        self.assertEqual(current.normalization, 2.0)

    def test_weighted_selection_matches_candidate_probability(self):
        selected_second = 0
        trials = 1000
        for index in range(trials):
            reservoir = DirectLightReservoir()
            reservoir.update(
                DirectLightSample(0, 0.0, 0.0, 1.0), 1.0, 0.5
            )
            reservoir.update(
                DirectLightSample(1, 0.0, 0.0, 3.0), 3.0,
                (index + 0.5) / trials,
            )
            selected_second += reservoir.sample.light_index == 1
        self.assertEqual(selected_second, 750)

    def test_pairwise_merge_matches_canonical_for_equal_targets(self):
        source = DirectLightReservoir()
        source.update(DirectLightSample(2, 0.25, 0.5, 2.0), 6.0, 0.0,
                      represented_samples=3)
        canonical = DirectLightReservoir()
        pairwise = DirectLightReservoir()
        canonical.merge_canonical(source, 2.0, 0.0)
        pairwise.merge_pairwise(source, 2.0, 0.0)
        self.assertEqual(pairwise.weight_sum, canonical.weight_sum)
        self.assertEqual(pairwise.sample_count, 1)

    def test_pairwise_merge_applies_source_balance_weight(self):
        source = DirectLightReservoir()
        source.update(DirectLightSample(1, 0.2, 0.4, 1.0), 4.0, 0.0,
                      represented_samples=2)
        destination = DirectLightReservoir()
        destination.merge_pairwise(source, 3.0, 0.0)
        # Canonical weight is 6; source balance is 2*1/(3+1) = 0.5.
        self.assertEqual(destination.weight_sum, 3.0)
        self.assertEqual(destination.sample.target, 3.0)

    def test_invalid_inputs_are_rejected(self):
        with self.assertRaises(ValueError):
            DirectLightSample(-1, 0.0, 0.0, 1.0)
        reservoir = DirectLightReservoir()
        sample = DirectLightSample(0, 0.0, 0.0, 1.0)
        with self.assertRaises(ValueError):
            reservoir.update(sample, -1.0, 0.5)
        with self.assertRaises(ValueError):
            reservoir.update(sample, 1.0, 1.0)

    def test_glsl_layout_and_operations_match_reference_contract(self):
        shader = (
            Path(__file__).parents[1]
            / "ordinarylight" / "shaders" / "wavefront_restir.glsl"
        ).read_text()
        self.assertIn("struct DirectLightReservoir", shader)
        self.assertIn("uvec4 data", shader)
        self.assertIn("uint current_reservoir_words[]", shader)
        self.assertIn("uint previous_reservoir_words[]", shader)
        self.assertIn("reservoir_index * 3u", shader)
        self.assertIn("DIRECT_LIGHT_STORAGE_INDEX_BITS = 25u", shader)
        self.assertIn("loadPreviousDirectLightReservoir", shader)
        self.assertIn("storeCurrentDirectLightReservoir", shader)
        self.assertIn("packHalf2x16", shader)
        self.assertIn("unpackHalf2x16", shader)
        self.assertIn("updateDirectLightReservoir", shader)
        self.assertIn("mergeDirectLightReservoir", shader)
        self.assertIn("mergeCanonicalDirectLightReservoir", shader)
        self.assertIn("mergePairwiseDirectLightReservoir", shader)
        self.assertIn("mergeBalancedDirectLightReservoir", shader)
        self.assertIn("directLightReservoirNormalization", shader)

        lighting = (
            Path(__file__).parents[1]
            / "ordinarylight" / "shaders" / "wavefront_lighting.glsl"
        ).read_text()
        primary = (
            Path(__file__).parents[1]
            / "ordinarylight" / "shaders" / "wavefront_primary_impl.glsl"
        ).read_text()
        self.assertIn("AreaLightCandidate generateAreaLightCandidate", lighting)
        self.assertIn("float areaLightCandidateVisibility", lighting)
        self.assertIn("volumeShadowTransmittance", lighting)
        self.assertIn("uint restir_reservoir_count", primary)
        self.assertIn("uint restir_reservoir_index", primary)
        self.assertIn(
            "storeCurrentDirectLightReservoir(\n"
            "                restir_reservoir_index, reservoir)", primary
        )
        self.assertIn(
            "previous_index * restir_reservoir_count", primary
        )
        self.assertIn(
            "proposal_flat_index\n"
            "                                                * restir_reservoir_count",
            primary,
        )
        self.assertIn("directLightReservoirNormalization(reservoir)", primary)
        self.assertIn("reprojectRestir(position, previous_pixel)", primary)
        self.assertIn("mergeDirectLightReservoir(", primary)
        self.assertIn("profileWork(11u, 1u)", primary)
        self.assertIn("profileWork(12u, 1u)", primary)
        self.assertIn("profileWork(13u, 1u)", primary)
        self.assertIn("restirSpatialOffset", primary)
        self.assertIn("push.restir_spatial_neighbors", primary)
        self.assertIn("push.restir_pairwise_mis", primary)
        self.assertIn("&& !material_textured", primary)
        self.assertIn("push.restir_generalized_mis", primary)
        self.assertIn("push.restir_generalized_balance_cap", primary)
        self.assertIn("active_proposals / target_sum", primary)
        self.assertIn("cancellation-free form", shader)
        self.assertIn("restirHistorySurfaceCompatible", primary)
        self.assertIn("source_history_limit", primary)
        self.assertIn("!history_source_present", primary)
        self.assertIn("restirMaterialSignature", primary)
        self.assertIn("previous_material_image", primary)
        self.assertIn("old_material == material_signature", primary)
        self.assertIn(
            "binding = 8, r32f", primary,
        )
        self.assertIn("restirPreviousWorldPosition", primary)
        self.assertIn("binding = 9, r32ui", primary)
        self.assertIn("restirPackNormalClass", primary)
        self.assertIn("restirUnpackNormalClass", primary)
        backend = (
            Path(__file__).parents[1] / "ordinarylight" / "targets" / "vulkan" / "core.py"
        ).read_text()
        self.assertIn("or self.config.wavefront_restir_di", backend)
        self.assertIn("vk.VK_FORMAT_R32_UINT", backend)
        self.assertIn('"wavefront_material_image"', backend)
        self.assertIn("vk.VK_FORMAT_R32_SFLOAT", backend)


if __name__ == "__main__":
    unittest.main()
