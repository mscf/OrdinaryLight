import tempfile
import unittest
from pathlib import Path

import numpy as np

from ordinarylight.integrations.temporal_quality import (
    load_hdr_sequence,
    save_hdr_sequence,
    summarize_temporal_quality,
    temporal_quality_rows,
    write_temporal_quality_csv,
)


class TemporalQualityTests(unittest.TestCase):
    def test_identical_sequence_has_zero_error(self):
        frames = np.ones((3, 2, 4, 4), np.float32)
        summary = summarize_temporal_quality(frames, frames)
        self.assertEqual(summary["rmse_max"], 0.0)
        self.assertEqual(summary["temporal_residual_rmse_max"], 0.0)

    def test_temporal_residual_and_lag_detect_stale_frame(self):
        reference = np.zeros((3, 1, 1, 3), np.float32)
        reference[1:] = 1.0
        candidate = reference.copy()
        candidate[1] = reference[0]
        rows = temporal_quality_rows(reference, candidate)
        self.assertGreater(rows[1]["temporal_residual_rmse"], 0.0)
        self.assertGreater(rows[1]["history_lag_ratio"], 1.0)

    def test_band_metric_detects_horizontal_structure(self):
        reference = np.zeros((2, 16, 32, 3), np.float32)
        horizontal = reference.copy()
        horizontal[:, ::2, :, :] = 1.0
        vertical = reference.copy()
        vertical[:, :, ::2, :] = 1.0
        horizontal_score = summarize_temporal_quality(
            reference, horizontal
        )["band_anisotropy_mean"]
        vertical_score = summarize_temporal_quality(
            reference, vertical
        )["band_anisotropy_mean"]
        self.assertGreater(horizontal_score, 10.0)
        self.assertLess(vertical_score, 0.1)

    def test_spectral_metric_detects_low_frequency_noise_structure(self):
        reference = np.zeros((2, 64, 128, 3), np.float32)
        x = np.arange(128, dtype=np.float32)[None, :]
        structured = np.sin(x * (2.0 * np.pi / 32.0))
        structured = np.broadcast_to(structured, (64, 128))
        candidate = reference.copy()
        candidate[..., :] = structured[None, ..., None]
        score = summarize_temporal_quality(
            reference, candidate
        )["low_frequency_energy_ratio_mean"]
        self.assertGreater(score, 10.0)

    def test_round_trip_and_csv(self):
        frames = np.arange(48, dtype=np.float32).reshape(2, 2, 4, 3)
        with tempfile.TemporaryDirectory() as directory:
            capture = Path(directory) / "capture.npz"
            report = Path(directory) / "report.csv"
            save_hdr_sequence(capture, frames, metadata={"samples": 8, "mode": "ref"})
            loaded, metadata = load_hdr_sequence(capture)
            np.testing.assert_array_equal(loaded, frames)
            self.assertEqual(metadata, {"samples": 8, "mode": "ref"})
            write_temporal_quality_csv(report, {"same": (frames, loaded)})
            self.assertIn("temporal_residual_rmse", report.read_text())

    def test_memory_mapped_npy_load(self):
        frames = np.ones((2, 3, 4, 3), np.float32)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "capture.npy"
            np.save(path, frames)
            loaded, metadata = load_hdr_sequence(path)
            np.testing.assert_array_equal(loaded, frames)
            self.assertEqual(metadata, {})

    def test_rejects_bad_shapes(self):
        with self.assertRaisesRegex(ValueError, "shape"):
            temporal_quality_rows(np.zeros((2, 3)), np.zeros((2, 3)))


if __name__ == "__main__":
    unittest.main()
