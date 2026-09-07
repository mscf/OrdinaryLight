"""Separate mirror temporal rejection from spatial filtering on saved signals."""
import argparse
import json
from pathlib import Path

import numpy as np

from ordinarylight.denoising import DenoiserSignals
from tools.denoiser_motion.replay import ShaderReplay
from tools.denoiser_motion.run import region_error, write_gallery


def analyze(capture, output):
    original = np.load(capture / "arrays.npz")
    source_report = json.loads((capture / "report.json").read_text())
    reference, mask = original["reference"], original["mask"]
    count, height, width = mask.shape
    signals = {name: [DenoiserSignals.load(capture / f"{name}-{i:03d}.npz")
                      for i in range(count)] for name in ("primary", "reflected")}
    depths = {name: [np.load(capture / f"depth-{i:03d}.npz")[name]
                     for i in range(count)] for name in signals}
    finite = np.stack([s.view_z > 0 for s in signals["reflected"]]) & mask
    cases = {
        "primary": ("primary", 3, 2.5, 32, 4),
        "reflected": ("reflected", 3, 2.5, 32, 4),
        "primary_temporal": ("primary", 0, 2.5, 32, 4),
        "reflected_temporal": ("reflected", 0, 2.5, 32, 4),
        "no_reactive": ("reflected", 3, 0, 32, 4),
        "broader_normals": ("reflected", 3, 2.5, 8, 4),
        "broader_color": ("reflected", 3, 2.5, 32, 2),
        "wider_depth": ("reflected", 3, 2.5, 32, 4),
        "depth_footprint": ("reflected", 3, 2.5, 32, 4),
    }
    arrays = {"reference": reference}
    report = {
        "capture": str(capture), "finite_reflected_pixels": int(finite.sum()),
        "mirror_pixels": int(mask.sum()), "variants": {},
    }
    for name, (guide, iterations, reactive, normal_power, color_weight) in cases.items():
        replay = ShaderReplay(width, height, depth_footprint=name == "depth_footprint")
        policy = dict(source_report["policy"], reactive_sigma=reactive)
        if name == "wider_depth":
            policy["depth_threshold"] = 0.02
        frames, histories, accepts = [], [], []
        try:
            for index in range(count):
                lobes, history = replay.denoise(
                    signals[guide][index], depths[guide][index], policy,
                    iterations=iterations, spatial_normal_power=normal_power,
                    color_weight=color_weight,
                )
                frames.append(lobes[0] + lobes[1])
                histories.append(history[1])
                accepts.append(replay.last_acceptance[1])
        finally:
            replay.close()
        arrays[name] = np.asarray(frames)
        histories, accepts = np.asarray(histories), np.asarray(accepts)
        metrics = {
            "all_mirror_error": region_error(reference, arrays[name], mask),
            "finite_surface_error": region_error(reference, arrays[name], finite),
            "revealed_error": region_error(reference, arrays[name], original["revealed"]),
            "phases": {},
        }
        for phase, indices in (("after_initial", slice(1, None)),
                               ("moving", slice(2, 6)), ("stopped", slice(6, None))):
            selection = finite[indices]
            metrics["phases"][phase] = {
                "finite_history_mean": float(histories[indices][selection].mean()),
                "finite_acceptance_fraction": float(accepts[indices][selection].mean()),
                "finite_error": region_error(reference[indices], arrays[name][indices], selection),
            }
        if name in ("primary", "reflected"):
            expected = original[f"{name}_guides"]
            metrics["original_replay_max_difference"] = float(np.max(np.abs(expected - arrays[name])))
            np.testing.assert_array_equal(expected, arrays[name])
        report["variants"][name] = metrics
    output.mkdir(parents=True, exist_ok=False)
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    np.savez_compressed(output / "arrays.npz", **arrays, finite=finite)
    write_gallery(output, {key: arrays[key] for key in (
        "reference", "primary", "reflected", "no_reactive", "broader_normals",
    )})
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    analyze(args.capture, args.output)


if __name__ == "__main__":
    main()
