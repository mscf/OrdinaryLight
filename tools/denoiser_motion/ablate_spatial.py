"""Replay spatial filtering from preserved live temporal outputs."""
import argparse
import json
from pathlib import Path

import numpy as np

from tests.gates.relax_motion_quality import evaluate_sequence, _luminance
from tools.denoiser_motion.analyze_edges import geometry_edges
from tools.denoiser_motion.replay import ShaderReplay


def analyze(capture, output):
    reference = np.load(capture / "reference.npy")
    live = np.load(capture / "floor3.npy")
    height, width = reference.shape[1:3]
    replay = ShaderReplay(width, height)
    boundaries = []
    variants = {name: [] for name in (
        "temporal_only", "production", "no_firefly", "combined",
        "broader_color",
    )}
    try:
        for index in range(len(reference)):
            with np.load(capture / f"floor3-guides-{index:03d}.npz") as guide:
                boundaries.append(geometry_edges(guide))
                normal = replay.texture(guide["normal_roughness"], "rgba16float")
                depth = replay.texture(guide["view_z"], "r32float")
                material = replay.texture(guide["material"], "r32uint")
                valid = guide["view_z"][..., 0] > 0
                values = [guide[f"temporal_{name}"].astype(np.float32)
                          for name in ("diffuse", "specular")]
                for name, frames in variants.items():
                    channels = values
                    iterations = 0 if name == "temporal_only" else 3
                    if name == "combined":
                        channels = [values[0] + values[1]]
                    results = []
                    for lobe, value in enumerate(channels):
                        source = replay.texture(value, "rgba16float")
                        for iteration in range(iterations):
                            target = replay.texture(np.zeros_like(value), "rgba16float")
                            replay.dispatch(
                                "atrous",
                                {0: source, 1: normal, 2: depth, 3: material, 4: target},
                                [width, height, 1 << iteration, 0, 32, 0.02,
                                 2 if name == "broader_color" else 4,
                                 float(lobe == 1 and iteration == 0
                                       and name not in ("no_firefly", "combined"))],
                                5,
                            )
                            source.destroy()
                            source = target
                        results.append(replay.read(source)[..., :3])
                        source.destroy()
                    # Match the live rgba16f composition store and preserve sky.
                    image = np.sum(results, axis=0).astype(np.float16).astype(np.float32)
                    image[~valid] = live[index, ..., :3][~valid]
                    frames.append(image)
                for texture in (normal, depth, material):
                    texture.destroy()
    finally:
        replay.close()
    arrays = {name: np.asarray(frames) for name, frames in variants.items()}
    difference = arrays["production"] - live[..., :3]
    edges = np.stack(boundaries)
    bands = edges.copy()
    bands[:, 1:] |= edges[:, :-1]
    bands[:, :-1] |= edges[:, 1:]
    bands[:, :, 1:] |= edges[:, :, :-1]
    bands[:, :, :-1] |= edges[:, :, 1:]
    metrics = {}
    for name, frames in arrays.items():
        error = _luminance(frames) - _luminance(reference)
        metrics[name] = {
            **evaluate_sequence(reference, frames),
            "geometry_band_log_luma_rmse": float(np.sqrt(np.mean(error[bands] ** 2))),
        }
    report = {
        "capture": str(capture),
        "production_replay_max_difference": float(np.max(np.abs(difference))),
        "production_replay_rmse": float(np.sqrt(np.mean(difference ** 2))),
        "metrics": metrics,
    }
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output / "variants.npz", **arrays)
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    analyze(args.capture, args.output)


if __name__ == "__main__":
    main()
