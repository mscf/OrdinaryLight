"""HDR sequence storage and objective temporal-quality measurements."""

from pathlib import Path
import csv

import numpy as np


def _sequence(value, name):
    sequence = np.asarray(value, dtype=np.float32)
    if sequence.ndim != 4 or sequence.shape[-1] < 3:
        raise ValueError(f"{name} must have shape (frames, height, width, channels)")
    if not np.all(np.isfinite(sequence[..., :3])):
        raise ValueError(f"{name} contains non-finite HDR values")
    return sequence[..., :3]


def save_hdr_sequence(path, frames, *, metadata=None):
    """Store a deterministic HDR sequence and optional scalar metadata."""
    frames = _sequence(frames, "frames")
    metadata = dict(metadata or {})
    unsupported = [
        key for key, value in metadata.items()
        if not isinstance(value, (str, bool, int, float, np.number))
    ]
    if unsupported:
        raise TypeError(f"metadata values must be scalar: {unsupported}")
    payload = {"frames": frames}
    payload.update({f"metadata_{key}": np.asarray(value) for key, value in metadata.items()})
    np.savez_compressed(Path(path), **payload)


def load_hdr_sequence(path):
    """Load frames and metadata written by :func:`save_hdr_sequence`."""
    path = Path(path)
    if path.suffix == ".npy":
        return _sequence(np.load(path, mmap_mode="r"), "frames"), {}
    with np.load(path, allow_pickle=False) as payload:
        frames = _sequence(payload["frames"], "frames").copy()
        metadata = {
            name.removeprefix("metadata_"): payload[name].item()
            for name in payload.files if name.startswith("metadata_")
        }
    return frames, metadata


def temporal_quality_rows(reference, candidate):
    """Return per-frame HDR accuracy, stability, bias, and history-lag metrics."""
    reference = _sequence(reference, "reference")
    candidate = _sequence(candidate, "candidate")
    if reference.shape != candidate.shape:
        raise ValueError("reference and candidate sequences must have matching shapes")
    rows = []
    epsilon = 1e-8
    for index in range(len(reference)):
        error = candidate[index] - reference[index]
        rmse = float(np.sqrt(np.mean(error * error)))
        reference_rms = float(np.sqrt(np.mean(reference[index] ** 2)))
        luminance_error = np.tensordot(
            error, np.asarray((0.2126, 0.7152, 0.0722), np.float32), axes=1
        )
        centered = luminance_error - float(np.mean(luminance_error))
        row_bias = np.mean(centered, axis=1)
        column_bias = np.mean(centered, axis=0)
        horizontal_band_rms = float(np.sqrt(np.mean(row_bias * row_bias)))
        vertical_band_rms = float(np.sqrt(np.mean(column_bias * column_bias)))
        height, width = luminance_error.shape
        normalized_row = horizontal_band_rms * np.sqrt(float(width))
        normalized_column = vertical_band_rms * np.sqrt(float(height))
        spectrum = np.fft.rfft2(centered)
        power = np.abs(spectrum) ** 2
        power[0, 0] = 0.0
        frequencies_y = np.fft.fftfreq(height)[:, None]
        frequencies_x = np.fft.rfftfreq(width)[None, :]
        low_frequency = (
            frequencies_x * frequencies_x + frequencies_y * frequencies_y
            <= (1.0 / 16.0) ** 2
        )
        low_frequency[0, 0] = False
        low_bin_fraction = float(np.mean(low_frequency))
        low_frequency_energy_ratio = (
            float(np.sum(power[low_frequency]) / max(np.sum(power), epsilon))
            / max(low_bin_fraction, epsilon)
        )
        row = {
            "frame": index,
            "mae": float(np.mean(np.abs(error))),
            "rmse": rmse,
            "relative_rmse": rmse / max(reference_rms, epsilon),
            "bias": float(np.mean(error)),
            "temporal_residual_rmse": 0.0,
            "history_lag_ratio": 0.0,
            "horizontal_band_rms": horizontal_band_rms,
            "vertical_band_rms": vertical_band_rms,
            "band_anisotropy": normalized_row / max(normalized_column, epsilon),
            "low_frequency_energy_ratio": low_frequency_energy_ratio,
        }
        if index:
            reference_delta = reference[index] - reference[index - 1]
            candidate_delta = candidate[index] - candidate[index - 1]
            residual = candidate_delta - reference_delta
            row["temporal_residual_rmse"] = float(
                np.sqrt(np.mean(residual * residual))
            )
            previous_error = candidate[index] - reference[index - 1]
            previous_rmse = float(np.sqrt(np.mean(previous_error * previous_error)))
            # Values above one mean the result is closer to previous-frame
            # truth than current-frame truth, a useful ghosting warning.
            row["history_lag_ratio"] = rmse / max(previous_rmse, epsilon)
        rows.append(row)
    return rows


def summarize_temporal_quality(reference, candidate):
    """Aggregate sequence metrics without hiding worst-frame behavior."""
    rows = temporal_quality_rows(reference, candidate)
    names = tuple(name for name in rows[0] if name != "frame")
    summary = {}
    for name in names:
        values = np.asarray([row[name] for row in rows], dtype=np.float64)
        summary[f"{name}_mean"] = float(np.mean(values))
        summary[f"{name}_p95"] = float(np.percentile(values, 95.0))
        summary[f"{name}_max"] = float(np.max(values))
    return summary


def write_temporal_quality_csv(path, comparisons):
    """Write rows for ``{mode: (reference, candidate)}`` comparisons."""
    output = []
    for mode, (reference, candidate) in comparisons.items():
        for row in temporal_quality_rows(reference, candidate):
            output.append({"mode": mode, **row})
    fields = (
        "mode", "frame", "mae", "rmse", "relative_rmse", "bias",
        "temporal_residual_rmse", "history_lag_ratio",
        "horizontal_band_rms", "vertical_band_rms", "band_anisotropy",
        "low_frequency_energy_ratio",
    )
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output)
