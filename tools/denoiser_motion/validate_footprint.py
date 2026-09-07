"""GPU stress tests with known valid correspondence and hidden-surface labels."""
import argparse
import gc
import json
from pathlib import Path

import numpy as np

from ordinarylight.denoising import DenoiserFrameInfo, DenoiserSignals
from tools.denoiser_motion.replay import ShaderReplay


def fixture(width, slope, hidden=False, different_identity=False):
    height = width // 2
    y, x = np.indices((height, width))
    depth = (10 + slope * x / width).astype(np.float32)
    normal = np.zeros((height, width, 4), np.float32)
    normal[..., 2:] = (1, 0.4)
    radiance = np.ones_like(normal)
    motion = np.zeros((height, width, 2), np.float32)
    ids = np.ones((height, width), np.uint32)
    frame = DenoiserFrameInfo(np.eye(4, dtype=np.float32),
                             np.eye(4, dtype=np.float32), 0)
    old = DenoiserSignals(radiance, radiance.copy(), normal, depth, motion, ids, frame)
    # A subpixel sample on the same plane is valid. A narrow newly exposed
    # parallel surface is invalid even when its material/instance/normal match.
    expected = depth + 0.4 * slope / width
    invalid = (x >= width // 2) & (x < width // 2 + max(1, width // 32)) if hidden else np.zeros_like(x, bool)
    expected = expected.copy()
    expected[invalid] = depth[invalid] + 0.12
    new_ids = ids.copy()
    if different_identity:
        new_ids[invalid] = 2
    current = DenoiserSignals(radiance, radiance.copy(), normal, expected,
                             motion, new_ids, frame)
    interior = (x > 1) & (x < width - 2) & (y > 1) & (y < height - 2)
    return old, current, expected, invalid & interior, (~invalid) & interior


def run(output):
    results = []
    for width in (64, 160, 320):
        for name, slope, hidden, distinct in (
            ("flat", 0, False, False), ("slope", 12, False, False),
            ("same_object_occlusion", 12, True, False),
            ("distinct_object_occlusion", 12, True, True),
        ):
            old, current, expected, invalid, valid = fixture(width, slope, hidden, distinct)
            for mode in ("original", "blanket", "adaptive"):
                replay = ShaderReplay(width, width // 2, depth_footprint=mode == "adaptive")
                policy = dict(history_limit=8, normal_threshold=0.95,
                              depth_threshold=0.02 if mode == "blanket" else 0.005,
                              clamp_sigma=1, reactive_sigma=2.5)
                try:
                    replay.denoise(old, old.view_z, policy, iterations=0)
                    replay.denoise(current, expected, policy, iterations=0)
                    accept = replay.last_acceptance[1] > 0
                    results.append(dict(width=width, fixture=name, mode=mode,
                        valid_pixels=int(valid.sum()), invalid_pixels=int(invalid.sum()),
                        valid_acceptance=float(accept[valid].mean()),
                        false_acceptance=float(accept[invalid].mean()) if invalid.any() else None))
                finally:
                    replay.close()
                    del replay
                    gc.collect()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
