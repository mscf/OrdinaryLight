import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import ordinarylight as ol
from ordinarylight.integrations.temporal_quality import summarize_temporal_quality
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
    def test_surface_presenter_exposes_signal_capture_on_its_existing_core(self):
        from ordinarylight.targets.vulkan.api import (
            VulkanSurfacePresenter, _VulkanGlobalIlluminationEngine,
        )

        presenter = object.__new__(VulkanSurfacePresenter)
        sentinel = object()
        with mock.patch.object(
            _VulkanGlobalIlluminationEngine,
            "capture_denoiser_signals",
            autospec=True,
            return_value=sentinel,
        ) as capture:
            result = presenter.capture_denoiser_signals(
                "scene", "camera", 64, 32, frame_index=9,
            )

        self.assertIs(result, sentinel)
        capture.assert_called_once_with(
            presenter, "scene", "camera", 64, 32, frame_index=9,
        )

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

    def test_nrd_adapter_preserves_sequence_in_one_bridge_call(self):
        class Bridge:
            calls = 0

            @staticmethod
            def version():
                return "sequence-test"

            @classmethod
            def denoise_relax_sequence(cls, values, settings):
                cls.calls += 1
                return [(
                    value.diffuse_radiance_hit_distance[..., :3],
                    value.specular_radiance_hit_distance[..., :3],
                ) for value in values]

        values = (signals(), signals())
        results = NrdRelaxReference(Bridge()).denoise_sequence(values)
        self.assertEqual(Bridge.calls, 1)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].implementation_version, "sequence-test")

    def test_nrd_adapter_validates_gpu_benchmark_telemetry(self):
        class Bridge:
            @staticmethod
            def version():
                return "nrd-test"

            @staticmethod
            def benchmark_relax(values, settings, *, warmup, iterations):
                return {
                    "median_gpu_ms": 1.25,
                    "p95_gpu_ms": 1.5,
                    "wall_ms": 2.0,
                    "persistent_mib": 48.0,
                    "transient_mib": 12.0,
                    "measured_frames": iterations,
                }

        result = NrdRelaxReference(Bridge()).benchmark(
            (signals(), signals()), warmup=2, iterations=7,
        )
        self.assertEqual(result.implementation_version, "nrd-test")
        self.assertEqual(result.measured_frames, 7)
        self.assertAlmostEqual(result.median_gpu_ms, 1.25)
        self.assertAlmostEqual(result.persistent_mib, 48.0)

    def test_nrd_benchmark_rejects_wall_clock_substitution(self):
        class Bridge:
            @staticmethod
            def version():
                return "invalid"

            @staticmethod
            def benchmark_relax(values, settings, *, warmup, iterations):
                return {"wall_ms": 2.0, "measured_frames": iterations}

        with self.assertRaisesRegex(RuntimeError, "missing"):
            NrdRelaxReference(Bridge()).benchmark((signals(),))

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

    def test_deterministic_relax_integration_has_no_band_or_finite_regression(self):
        """Compare enabled ReLAX with the exact same fixed-seed raw sequence."""
        denoiser = ol.PortableDenoiser(ol.PortableDenoiserConfig(
            spatial_iterations=2, max_history_frames=16,
        ))
        raw = []
        filtered = []
        for seed in range(16):
            value = noisy_signals(seed, width=32, height=24)
            raw.append(
                value.diffuse_radiance_hit_distance[..., :3]
                + value.specular_radiance_hit_distance[..., :3]
            )
            filtered.append(denoiser.denoise(value).combined)
        raw = np.asarray(raw, np.float32)
        filtered = np.asarray(filtered, np.float32)
        truth = np.empty_like(raw)
        truth[..., :16, :] = (0.7, 0.9, 1.3)
        truth[..., 16:, :] = (1.7, 1.9, 2.3)
        self.assertTrue(np.isfinite(filtered).all())
        raw_quality = summarize_temporal_quality(truth, raw)
        relax_quality = summarize_temporal_quality(truth, filtered)
        self.assertLess(
            relax_quality["relative_rmse_mean"],
            raw_quality["relative_rmse_mean"] * 0.65,
        )
        self.assertLess(
            relax_quality["horizontal_band_rms_p95"],
            raw_quality["horizontal_band_rms_p95"] * 0.75,
        )
        self.assertGreater(relax_quality["edge_gradient_gain_mean"], 0.75)

    def test_visual_metrics_detect_repeating_horizontal_corruption(self):
        height, width = 32, 40
        reference = np.ones((2, height, width, 3), np.float32)
        candidate = reference.copy()
        candidate[:, 7::8, :, :] = 0.0
        metrics = summarize_temporal_quality(reference, candidate)
        self.assertGreater(metrics["horizontal_band_rms_max"], 0.25)
        self.assertGreater(metrics["band_anisotropy_max"], 10.0)

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
        self.assertIn("firefly_limit", glsl)

    def test_portable_specular_firefly_clamp_preserves_diffuse(self):
        value = signals(7, 7)
        specular = value.specular_radiance_hit_distance.copy()
        specular[3, 3, :3] = 100.0
        value = ol.DenoiserSignals(
            value.diffuse_radiance_hit_distance, specular,
            value.normal_roughness, value.view_z, value.motion,
            value.material_id, value.frame,
        )
        result = ol.PortableDenoiser(ol.PortableDenoiserConfig(
            spatial_iterations=1,
        )).denoise(value)
        np.testing.assert_allclose(
            result.diffuse,
            value.diffuse_radiance_hit_distance[..., :3],
            atol=1e-6,
        )
        self.assertLess(float(result.specular[3, 3, 0]), 10.0)

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
            "previous_vertices", "identity_output",
        ):
            self.assertIn(name, source)
        self.assertIn("primary_geometry.w", source)
        self.assertIn("identity_output", source)

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

    def test_temporal_kernel_validates_expected_previous_camera_depth(self):
        import ordinaryshade as osh
        from ordinarylight.denoising.kernels import relax_temporal

        source = osh.compile(
            relax_temporal, target="glsl", validate=False,
        ).source
        self.assertIn("expected_old_depth", source)
        self.assertIn("expected_old_depth > 0.0", source)
        self.assertIn("expected_old_depth - old_depth", source)

    def test_temporal_kernel_rejects_different_primitive_identity(self):
        import ordinaryshade as osh
        from ordinarylight.denoising.kernels import relax_temporal

        source = osh.compile(
            relax_temporal, target="glsl", validate=False,
        ).source
        self.assertIn("previous_identity", source)
        self.assertIn("old_primitive == current_primitive", source)

    def test_temporal_kernel_rejects_reactive_luminance_history(self):
        import ordinaryshade as osh
        from ordinarylight.denoising.kernels import relax_temporal

        source = osh.compile(
            relax_temporal, target="glsl", validate=False,
        ).source
        self.assertIn("history_luma", source)
        self.assertIn("reactive_limit", source)
        self.assertIn("constants.rejection.w", source)


if __name__ == "__main__":
    unittest.main()
