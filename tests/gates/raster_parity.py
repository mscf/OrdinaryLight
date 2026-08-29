"""Gate Vulkan/WebGPU raster parity on shared Ordinary Light scenes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from tools.raster_showcase import _comparison, _render


def _metrics(reference, candidate, tolerance):
    difference = np.abs(
        np.asarray(reference, np.int16) - np.asarray(candidate, np.int16)
    )
    rgb = difference[..., :3]
    return {
        "maximum_channel_error": int(rgb.max(initial=0)),
        "mean_absolute_error": float(rgb.mean()),
        "root_mean_square_error": float(
            np.sqrt(np.mean(np.square(rgb.astype(np.float64))))
        ),
        "differing_pixel_fraction": float(
            np.mean(np.max(rgb, axis=-1) > int(tolerance))
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=180)
    parser.add_argument(
        "--modes", nargs="+", choices=("scene", "triangle", "volume"),
        default=("scene", "triangle", "volume"),
    )
    parser.add_argument("--max-channel-error", type=int, default=1)
    parser.add_argument("--max-differing-fraction", type=float, default=0.001)
    parser.add_argument(
        "--output", type=Path, default=Path("/tmp/ordinarylight_raster_parity"),
    )
    args = parser.parse_args()
    if args.width < 1 or args.height < 1:
        parser.error("width and height must be positive")
    if not 0 <= args.max_channel_error <= 255:
        parser.error("max-channel-error must be between 0 and 255")
    if not 0.0 <= args.max_differing_fraction <= 1.0:
        parser.error("max-differing-fraction must be in [0, 1]")

    args.output.mkdir(parents=True, exist_ok=True)
    report = {
        "extent": [args.width, args.height],
        "max_channel_error": args.max_channel_error,
        "max_differing_fraction": args.max_differing_fraction,
        "modes": {},
    }
    failures = []
    for mode in args.modes:
        print(f"Rendering {mode} through Vulkan raster...")
        vulkan = _render("vulkan", args.width, args.height, mode)
        print(f"Rendering {mode} through WebGPU raster...")
        webgpu = _render("webgpu", args.width, args.height, mode)
        metrics = _metrics(vulkan, webgpu, args.max_channel_error)
        report["modes"][mode] = metrics
        print(
            f"  max={metrics['maximum_channel_error']} "
            f"mae={metrics['mean_absolute_error']:.6g} "
            f"different={metrics['differing_pixel_fraction']:.6%}"
        )
        comparison = _comparison([
            ("VULKAN", vulkan), ("WEBGPU", webgpu),
        ])
        comparison.save(args.output / f"{mode}.png")
        difference = np.abs(
            vulkan[..., :3].astype(np.int16)
            - webgpu[..., :3].astype(np.int16)
        ).astype(np.uint8)
        Image.fromarray(difference, "RGB").save(
            args.output / f"{mode}_difference.png"
        )
        if metrics["differing_pixel_fraction"] > args.max_differing_fraction:
            failures.append(mode)

    (args.output / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    if failures:
        raise SystemExit(
            "FAIL: raster parity exceeded tolerance for " + ", ".join(failures)
        )
    print("PASS: Vulkan and WebGPU raster paths are within tolerance")


if __name__ == "__main__":
    main()
