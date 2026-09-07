"""Compare image error on captured geometry boundaries and noise-selected edges."""

import argparse
import json
from pathlib import Path
import numpy as np
from tests.gates.relax_motion_quality import _gradient, _luminance, evaluate_sequence


def geometry_edges(guide):
    normal = guide["normal_roughness"][..., :3].astype(np.float32)
    depth = guide["view_z"][..., 0]
    identity = guide["identity"][..., 0]
    edge = np.zeros(depth.shape, bool)
    for axis in (0, 1):
        other = np.roll(normal, 1, axis=axis)
        z = np.roll(depth, 1, axis=axis)
        active = (depth != 0) | (z != 0)
        changed = active & (
            (identity != np.roll(identity, 1, axis=axis))
            | (np.sum(normal * other, axis=-1) < 0.95)
            | (np.abs(depth - z) > np.maximum(0.02 * np.abs(depth), 0.001))
        )
        if axis == 0:
            changed[0] = False
        else:
            changed[:, 0] = False
        edge |= changed | np.roll(changed, -1, axis=axis)
    return edge


def analyze(path):
    reference = np.load(path / "reference.npy")
    halves = np.load(path / "reference-split.npy")
    candidates = {f"floor{n}": np.load(path / f"floor{n}.npy") for n in (1, 3)}
    guides = [
        np.load(path / f"floor3-guides-{i:03d}.npz") for i in range(len(reference))
    ]
    edges = np.stack([geometry_edges(g) for g in guides])
    bands = edges.copy()
    bands[:, 1:] |= edges[:, :-1]
    bands[:, :-1] |= edges[:, 1:]
    bands[:, :, 1:] |= edges[:, :, :-1]
    bands[:, :, :-1] |= edges[:, :, 1:]
    ref = _luminance(reference)
    selected = np.stack(
        [_gradient(r) > max(np.percentile(_gradient(r), 85), 1e-5) for r in ref]
    )
    report = {
        "reference_samples": json.loads((path / "configuration.json").read_text())[
            "reference_samples"
        ],
        "reference_split_log_luma_rmse": float(
            np.sqrt(np.mean((_luminance(halves[:, 0]) - _luminance(halves[:, 1])) ** 2))
        ),
        "selected_edge_fraction_on_geometry": float(np.mean(bands[selected])),
        "metrics": {},
    }
    for name, frames in candidates.items():
        error = _luminance(frames) - ref
        report["metrics"][name] = {
            **evaluate_sequence(reference, frames),
            "geometry_band_log_luma_rmse": float(np.sqrt(np.mean(error[bands] ** 2))),
            "per_frame_geometry_band_rmse": [
                float(np.sqrt(np.mean(e[m] ** 2))) for e, m in zip(error, bands)
            ],
            "interior_log_luma_rmse": float(np.sqrt(np.mean(error[~bands] ** 2))),
            "geometry_gradient_rmse": float(
                np.sqrt(
                    np.mean(
                        (
                            np.stack([_gradient(r) for r in _luminance(frames)])
                            - np.stack([_gradient(r) for r in ref])
                        )[edges]
                        ** 2
                    )
                )
            ),
        }
    history = np.stack([g["diffuse_history"][..., 0] for g in guides])
    report["history_over_one"] = {
        "geometry": float(np.mean(history[bands] > 1)),
        "interior": float(np.mean(history[~bands] > 1)),
    }
    np.savez_compressed(
        path / "edge-masks.npz", geometry=edges, band=bands, selected=selected
    )
    (path / "edge-report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    print(json.dumps(analyze(parser.parse_args().capture), indent=2))
