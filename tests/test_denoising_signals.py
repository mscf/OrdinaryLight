import tempfile
import unittest
from pathlib import Path

import numpy as np

import ordinarylight as ol
from tests.gates.denoiser_reference_quality import main as reference_gate_main
from ordinarylight.denoising.reference import (
    NrdRelaxReference,
    ReferenceDenoiserUnavailable,
)


def signals(width=5, height=3):
    normal_roughness = np.zeros((height, width, 4), np.float32)
    normal_roughness[..., 2] = 1.0
    normal_roughness[..., 3] = 0.4
    radiance = np.zeros((height, width, 4), np.float32)
    radiance[..., :3] = (0.2, 0.4, 0.8)
    radiance[..., 3] = 2.5
    frame = ol.DenoiserFrameInfo(
        np.eye(4, dtype=np.float32), np.eye(4, dtype=np.float32), 7,
        jitter=(0.25, -0.25), previous_jitter=(-0.25, 0.25),
    )
    return ol.DenoiserSignals(
        radiance, radiance.copy(), normal_roughness,
        np.ones((height, width), np.float32),
        np.zeros((height, width, 2), np.float32),
        np.full((height, width), 11, np.uint32), frame,
    )


def noisy_signals(seed, width=24, height=16, *, camera_cut=False):
    rng = np.random.default_rng(seed)
    value = signals(width, height)
    diffuse = value.diffuse_radiance_hit_distance.copy()
    diffuse[..., :3] = np.maximum(
        0.5 + rng.normal(0.0, 0.3, (height, width, 1)), 0.0
    )
    # A material/depth discontinuity must survive spatial filtering.
    diffuse[:, width // 2:, :3] += 1.0
    material = value.material_id.copy()
    material[:, width // 2:] = 12
    depth = value.view_z.copy()
    depth[:, width // 2:] = 2.0
    frame = ol.DenoiserFrameInfo(
        value.frame.world_to_clip, value.frame.previous_world_to_clip, seed,
        camera_cut=camera_cut,
    )
    return ol.DenoiserSignals(
        diffuse, value.specular_radiance_hit_distance.copy(),
        value.normal_roughness.copy(), depth, value.motion.copy(), material,
        frame,
    )


class DenoiserSignalTests(unittest.TestCase):
    def test_contract_round_trip_preserves_exact_signals(self):
        expected = signals()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "frame.npz"
            expected.save(path)
            actual = ol.DenoiserSignals.load(path)
        self.assertEqual(actual.extent, (5, 3))
        self.assertEqual(actual.frame.frame_index, 7)
        for name in (
            "diffuse_radiance_hit_distance",
            "specular_radiance_hit_distance",
            "normal_roughness", "view_z", "motion", "material_id",
        ):
            np.testing.assert_array_equal(getattr(actual, name), getattr(expected, name))

    def test_contract_rejects_non_unit_foreground_normals(self):
        value = signals()
        normals = value.normal_roughness.copy()
        normals[..., :3] *= 0.5
        with self.assertRaisesRegex(ol.SignalValidationError, "unit length"):
            ol.DenoiserSignals(
                value.diffuse_radiance_hit_distance,
                value.specular_radiance_hit_distance, normals, value.view_z,
                value.motion, value.material_id, value.frame,
            )

    def test_contract_rejects_negative_hit_distance(self):
        value = signals()
        diffuse = value.diffuse_radiance_hit_distance.copy()
        diffuse[0, 0, 3] = -1.0
        with self.assertRaisesRegex(ol.SignalValidationError, "cannot be negative"):
            ol.DenoiserSignals(
                diffuse, value.specular_radiance_hit_distance,
                value.normal_roughness, value.view_z, value.motion,
                value.material_id, value.frame,
            )

    def test_nrd_adapter_is_optional(self):
        reference = NrdRelaxReference()
        if not reference.available:
            with self.assertRaisesRegex(
                ReferenceDenoiserUnavailable, "tools/nrd_reference"
            ):
                reference.denoise(signals())

    def test_nrd_adapter_validates_bridge_result(self):
        class Bridge:
            @staticmethod
            def version():
                return "test"

            @staticmethod
            def denoise_relax(value, settings):
                height, width = value.view_z.shape
                amount = settings.get("amount", 1.0)
                return (
                    np.full((height, width, 3), amount, np.float32),
                    np.full((height, width, 3), 2.0, np.float32),
                )

        result = NrdRelaxReference(Bridge()).denoise(signals(), amount=3.0)
        self.assertEqual(result.implementation_version, "test")
        self.assertAlmostEqual(float(result.combined[0, 0, 0]), 5.0)

    def test_sequence_evaluation_compares_portable_and_reference_outputs(self):
        class Bridge:
            @staticmethod
            def version():
                return "nrd-test"

            @staticmethod
            def denoise_relax(value, settings):
                return (
                    value.diffuse_radiance_hit_distance[..., :3],
                    value.specular_radiance_hit_distance[..., :3],
                )

        sequence = (signals(), signals())
        truth = (
            sequence[0].diffuse_radiance_hit_distance[..., :3]
            + sequence[0].specular_radiance_hit_distance[..., :3]
        )
        result = ol.evaluate_denoiser_sequence(
            sequence, truth,
            portable_config=ol.PortableDenoiserConfig(spatial_iterations=0),
            reference_denoiser=NrdRelaxReference(Bridge()),
        )
        self.assertEqual(result.reference_implementation, "nrd-test")
        self.assertLess(result.portable.relative_rmse, 1e-6)
        self.assertLess(result.reference.relative_rmse, 1e-6)
        self.assertLess(
            result.portable_against_reference.relative_rmse, 1e-6
        )

    def test_sequence_evaluation_validates_ground_truth_extent(self):
        with self.assertRaisesRegex(ValueError, "ground_truth must have shape"):
            ol.evaluate_denoiser_sequence(
                (signals(),), np.zeros((1, 1, 3), np.float32),
            )

    def test_reference_quality_gate_consumes_saved_signal_sequence(self):
        with tempfile.TemporaryDirectory() as directory:
            capture = Path(directory) / "capture"
            capture.mkdir()
            sequence = (signals(), signals())
            for index, value in enumerate(sequence):
                value.save(capture / f"frame-{index:04d}.npz")
            truth = (
                sequence[0].diffuse_radiance_hit_distance[..., :3]
                + sequence[0].specular_radiance_hit_distance[..., :3]
            )
            np.save(capture / "ground_truth.npy", truth)
            output = Path(directory) / "metrics.json"
            self.assertEqual(reference_gate_main([
                str(capture), "--output", str(output),
            ]), 0)
            payload = __import__("json").loads(output.read_text())
        self.assertEqual(payload["frames"], 2)
        self.assertEqual(payload["extent"], [5, 3])
        self.assertIsNone(payload["reference"])

    def test_portable_oracle_reduces_temporal_noise_and_preserves_edge(self):
        denoiser = ol.PortableDenoiser(ol.PortableDenoiserConfig(
            spatial_iterations=1, max_history_frames=16,
        ))
        raw_frames = []
        filtered_frames = []
        result = None
        for index in range(12):
            value = noisy_signals(index)
            raw_frames.append(value.diffuse_radiance_hit_distance[..., 0])
            result = denoiser.denoise(value)
            filtered_frames.append(result.diffuse[..., 0])
        raw_variance = float(np.var(np.stack(raw_frames), axis=0).mean())
        filtered_variance = float(
            np.var(np.stack(filtered_frames[-4:]), axis=0).mean()
        )
        self.assertLess(filtered_variance, raw_variance * 0.35)
        left = float(result.diffuse[:, 10, 0].mean())
        right = float(result.diffuse[:, 13, 0].mean())
        self.assertGreater(right - left, 0.75)
        self.assertGreater(result.temporal_acceptance, 0.9)

    def test_portable_oracle_camera_cut_discards_history(self):
        denoiser = ol.PortableDenoiser(ol.PortableDenoiserConfig(
            spatial_iterations=0,
        ))
        denoiser.denoise(noisy_signals(1))
        result = denoiser.denoise(noisy_signals(2, camera_cut=True))
        self.assertEqual(result.temporal_acceptance, 0.0)
        np.testing.assert_array_equal(
            result.history_length, np.ones_like(result.history_length)
        )

    def test_quality_baseline_requires_explicit_regression_override(self):
        reference = np.ones((4, 5, 3), np.float32)
        good = ol.DenoiserQualityMetrics.measure(
            np.stack((reference, reference)), reference,
        )
        bad = ol.DenoiserQualityMetrics.measure(
            np.stack((reference * 0.5, reference * 1.5)), reference,
        )
        baseline = ol.DenoiserQualityBaseline("test-scene", good)
        with self.assertRaisesRegex(AssertionError, "explicit override reason"):
            baseline.require(bad)
        failures = baseline.require(bad, override_reason="intentional test")
        self.assertIn("relative_rmse", failures)

    def test_ordinary_shade_denoiser_kernels_compile_for_both_targets(self):
        import ordinaryshade as osh
        from ordinarylight.denoising.kernels import (
            prepare_decode_normal, prepare_previous_pixel,
            prepare_relax_signals, prepare_unpack_normal, relax_atrous,
            relax_compose, relax_temporal,
        )

        for target in ("glsl", "wgsl"):
            for kernel in (
                prepare_relax_signals, relax_temporal, relax_atrous,
                relax_compose,
            ):
                helpers = ()
                if kernel is prepare_relax_signals:
                    helpers = (
                        prepare_decode_normal, prepare_unpack_normal,
                        prepare_previous_pixel,
                    )
                source = osh.compile(
                    kernel, target=target, validate=False, helpers=helpers,
                ).source
                self.assertIn("main", source)

    def test_atrous_iteration_constants_are_portable_push_constants(self):
        import ordinaryshade as osh
        from ordinarylight.denoising.kernels import relax_atrous

        glsl = osh.compile(
            relax_atrous, target="glsl", validate=False,
        ).source
        wgsl = osh.compile(
            relax_atrous, target="wgsl", validate=False,
        ).source
        self.assertIn("layout(push_constant)", glsl)
        self.assertIn("@group(0) @binding(5) var<uniform>", wgsl)

    def test_prepare_kernel_emits_canonical_signal_outputs(self):
        import ordinaryshade as osh
        from ordinarylight.denoising.kernels import (
            prepare_decode_normal, prepare_previous_pixel,
            prepare_relax_signals, prepare_unpack_normal,
        )

        source = osh.compile(
            prepare_relax_signals, target="glsl", validate=False,
            helpers=(
                prepare_decode_normal, prepare_unpack_normal,
                prepare_previous_pixel,
            ),
        ).source
        for name in (
            "diffuse_output", "specular_output", "normal_roughness_output",
            "view_z_output", "motion_output", "previous_camera",
        ):
            self.assertIn(name, source)

    def test_temporal_kernel_clamps_reprojected_history(self):
        import ordinaryshade as osh
        from ordinarylight.denoising.kernels import relax_temporal

        source = osh.compile(
            relax_temporal, target="glsl", validate=False,
        ).source
        self.assertIn("neighborhood_variance", source)
        self.assertIn("constants.rejection.z", source)
        self.assertIn("clamp(history.rgb", source)

    def test_temporal_kernel_requires_explicitly_valid_history(self):
        import ordinaryshade as osh
        from ordinarylight.denoising.kernels import relax_temporal

        source = osh.compile(
            relax_temporal, target="glsl", validate=False,
        ).source
        self.assertIn("constants.extent_history.w > 0.5", source)


if __name__ == "__main__":
    unittest.main()
