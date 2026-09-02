"""Optional local bridge to NVIDIA NRD's offline benchmark executable.

This module is intentionally not imported by Ordinary Light unless an NRD
reference is explicitly requested.  It does not make NRD a runtime dependency.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import struct
import subprocess
import tempfile

import numpy as np


_ROOT = Path(__file__).resolve().parent
_DEFAULT_EXECUTABLE = (
    _ROOT / "tools" / "nrd_reference" / "_build" /
    "ordinarylight_nrd_benchmark"
)


def version() -> str:
    return "NRD 4.18.0 bridge-1"


def _executable() -> Path:
    override = os.environ.get("ORDINARYLIGHT_NRD_BENCHMARK")
    path = Path(override).expanduser() if override else _DEFAULT_EXECUTABLE
    if not path.is_file():
        raise RuntimeError(
            "NRD benchmark executable is unavailable; run "
            "`python tools/nrd_reference/bootstrap.py`"
        )
    return path


def benchmark_relax(signals, settings, *, warmup: int, iterations: int):
    del settings  # Native RELAX settings are pinned for reproducible comparisons.
    if not signals:
        raise ValueError("signals must not be empty")
    width, height = signals[0].extent
    completed = subprocess.run(
        (
            str(_executable()),
            "--width", str(width),
            "--height", str(height),
            "--warmup", str(warmup),
            "--iterations", str(iterations),
        ),
        check=True,
        capture_output=True,
        text=True,
    )
    start = completed.stdout.find("{")
    if start < 0:
        raise RuntimeError("NRD benchmark did not emit JSON telemetry")
    return json.loads(completed.stdout[start:])


_CAPTURE_MAGIC = b"OLNRDIN1"
_RESULT_MAGIC = b"OLNRDOU1"


def _packed_normal_roughness(signal):
    """Match NRD_FrontEnd_PackNormalAndRoughness for encoding 2."""
    value = signal.normal_roughness
    normal = value[..., :3].astype(np.float32, copy=True)
    denominator = np.sum(np.abs(normal), axis=-1, keepdims=True)
    normal /= np.maximum(denominator, np.float32(1e-9))
    roughness = np.square(value[..., 3], dtype=np.float32)
    roughness = np.maximum(roughness, np.float32(1.5 / 512.0))
    packed = np.empty(value.shape, dtype=np.float32)
    packed[..., 1] = normal[..., 1] * 0.5 + 0.5
    packed[..., 0] = normal[..., 0] * 0.5 + packed[..., 1]
    packed[..., 1] -= normal[..., 0] * 0.5
    packed[..., 2] = np.where(normal[..., 2] < 0.0, -roughness, roughness) * 0.5 + 0.5
    packed[..., 3] = np.minimum(signal.material_id, 3).astype(np.float32) / 3.0
    return np.ascontiguousarray(packed, dtype=np.float16)


def _write_sequence(path, signals):
    width, height = signals[0].extent
    with Path(path).open("wb") as stream:
        stream.write(_CAPTURE_MAGIC)
        stream.write(struct.pack("<IIII", 1, width, height, len(signals)))
        for signal in signals:
            frame = signal.frame
            stream.write(struct.pack(
                "<Q?3x4f", frame.frame_index, frame.camera_cut,
                *frame.jitter, *frame.previous_jitter,
            ))
            stream.write(np.asarray(frame.world_to_clip, dtype="<f4").tobytes())
            stream.write(np.asarray(frame.previous_world_to_clip, dtype="<f4").tobytes())
            stream.write(np.asarray(signal.motion, dtype="<f2").tobytes())
            stream.write(_packed_normal_roughness(signal).astype("<f2", copy=False).tobytes())
            stream.write(np.asarray(signal.view_z, dtype="<f4").tobytes())
            stream.write(np.asarray(signal.diffuse_radiance_hit_distance, dtype="<f2").tobytes())
            stream.write(np.asarray(signal.specular_radiance_hit_distance, dtype="<f2").tobytes())


def _read_results(path, expected_frames, width, height):
    with Path(path).open("rb") as stream:
        if stream.read(8) != _RESULT_MAGIC:
            raise RuntimeError("NRD bridge returned an invalid result file")
        version, result_width, result_height, frame_count = struct.unpack(
            "<IIII", stream.read(16)
        )
        if (version, result_width, result_height, frame_count) != (
            1, width, height, expected_frames,
        ):
            raise RuntimeError("NRD bridge returned incompatible result metadata")
        count = height * width * 4
        results = []
        for _ in range(frame_count):
            diffuse = np.frombuffer(stream.read(count * 2), dtype="<f2").reshape(height, width, 4)
            specular = np.frombuffer(stream.read(count * 2), dtype="<f2").reshape(height, width, 4)
            results.append((
                np.ascontiguousarray(diffuse[..., :3], dtype=np.float32),
                np.ascontiguousarray(specular[..., :3], dtype=np.float32),
            ))
        if stream.read(1):
            raise RuntimeError("NRD bridge result contains trailing data")
    return results


def denoise_relax_sequence(signals, settings):
    del settings  # Settings remain pinned until convention parity is gated.
    sequence = tuple(signals)
    if not sequence:
        raise ValueError("signals must not be empty")
    width, height = sequence[0].extent
    if any(signal.extent != (width, height) for signal in sequence):
        raise ValueError("all NRD signal frames must share one extent")
    with tempfile.TemporaryDirectory(prefix="ordinarylight-nrd-") as directory:
        capture = Path(directory) / "signals.bin"
        result = Path(directory) / "result.bin"
        _write_sequence(capture, sequence)
        subprocess.run(
            (str(_executable()), "--input", str(capture), "--output", str(result)),
            check=True,
        )
        return _read_results(result, len(sequence), width, height)


def denoise_relax(signals, settings):
    return denoise_relax_sequence((signals,), settings)[0]


__all__ = [
    "benchmark_relax", "denoise_relax", "denoise_relax_sequence", "version",
]
