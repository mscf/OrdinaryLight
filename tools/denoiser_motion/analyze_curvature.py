"""Audit the smooth-surface curvature assumption on saved faceted captures."""
import argparse
import gc
import json
from pathlib import Path

import numpy as np

from ordinarylight.denoising import DenoiserSignals
from tools.denoiser_motion.replay import ShaderReplay


def analyze(capture, output):
    rows = []
    for width in (64, 160, 320):
        old = DenoiserSignals.load(capture / f"old-{width}.npz")
        current = DenoiserSignals.load(capture / f"current-{width}.npz")
        with np.load(capture / f"guides-{width}.npz") as guide:
            positions = guide["positions"]
            valid, invalid = guide["valid"], guide["invalid"]
        height = width // 2
        yy, xx = np.indices((height, width))
        q = (np.stack((xx, yy), -1) + current.motion + 0.5).astype(int)
        qx, qy = np.clip(q[..., 0], 0, width-1), np.clip(q[..., 1], 0, height-1)
        delta = positions[1] - positions[0][qy, qx]
        normal = old.normal_roughness[qy, qx, :3]
        distance = np.abs(np.sum(delta * normal, axis=-1))
        change = np.linalg.norm(current.normal_roughness[..., :3] - normal, axis=-1)
        tangent = np.sqrt(np.maximum(np.sum(delta**2, axis=-1) - distance**2, 0))
        # The fixture uses the default 45-degree vertical FOV.
        footprint = 2 * old.view_z[qy, qx] * np.tan(np.pi / 8) / height
        base = old.view_z[qy, qx] * 1e-5
        allowance = change * np.maximum(footprint, tangent)
        policy = dict(history_limit=8, normal_threshold=0.95,
                      depth_threshold=0.005, clamp_sigma=1, reactive_sigma=2.5)
        replay = ShaderReplay(width, height, depth_footprint=True)
        try:
            replay.denoise(old, old.view_z, policy, iterations=0)
            # Camera translates only in X; view depth is unchanged by that transform.
            replay.denoise(current, current.view_z, policy, iterations=0)
            accept = replay.last_acceptance[1] > 0
        finally:
            replay.close()
            del replay
            gc.collect()
        rejected = valid & accept & (distance > base + 0.5 * allowance)
        rows.append(dict(width=width, accepted_valid=int((valid & accept).sum()),
            accepted_invalid=int((invalid & accept).sum()),
            half_factor_rejections=int(rejected.sum()),
            rejected_with_changed_normal=int((rejected & (change > 1e-5)).sum()),
            coefficients={str(f): dict(
                extra_valid_rejections=int((valid & accept & (distance > base + f*allowance)).sum()),
                remaining_false_accepts=int((invalid & accept & (distance <= base + f*allowance)).sum()),
            ) for f in (0.5, 1.0)},
        ))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    analyze(args.capture, args.output)
