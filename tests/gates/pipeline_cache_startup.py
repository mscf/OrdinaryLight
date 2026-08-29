"""Validate cross-process Vulkan pipeline-cache startup improvement."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def _run(args, output, cache, driver_cache):
    command = [
        sys.executable,
        str(Path(__file__).with_name("ordinaryshade_shade_parity.py")),
        "--gate", "--scene", args.scene,
        "--width", str(args.width), "--height", str(args.height),
        "--samples", str(args.samples), "--bounces", str(args.bounces),
        "--timing-runs", "1", "--pipeline-cache-path", str(cache),
        "--output", str(output),
    ]
    environment = os.environ.copy()
    environment["XDG_CACHE_HOME"] = str(driver_cache)
    started = time.perf_counter()
    completed = subprocess.run(command, env=environment, check=False)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    if completed.returncode:
        raise SystemExit(
            f"FAIL: parity subprocess exited with {completed.returncode}"
        )
    report = json.loads((output / "report.json").read_text())
    return report, elapsed_ms


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=180)
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--bounces", type=int, default=6)
    parser.add_argument("--scene", default="area_lights")
    parser.add_argument("--minimum-speedup", type=float, default=1.5)
    parser.add_argument(
        "--output", type=Path,
        default=Path("/tmp/ordinarylight_pipeline_cache_startup"),
    )
    parser.add_argument("--gate", action="store_true")
    args = parser.parse_args()
    if args.width < 1 or args.height < 1 or args.samples < 1 or args.bounces < 1:
        parser.error("dimensions, samples, and bounces must be positive")
    if args.minimum_speedup <= 1.0:
        parser.error("minimum speedup must be greater than one")
    args.output.mkdir(parents=True, exist_ok=True)
    token = time.time_ns()
    cache = args.output / f"pipeline-{token}.bin"
    driver_cache = args.output / f"driver-{token}"
    cold, cold_elapsed = _run(
        args, args.output / "cold", cache, driver_cache
    )
    warm, warm_elapsed = _run(
        args, args.output / "warm", cache, driver_cache
    )
    cold_generated = cold["generated_statistics"]
    warm_generated = warm["generated_statistics"]
    cold_activation = (
        cold_generated["construction_ms"]
        + cold_generated["first_render_ms"]
    )
    warm_activation = (
        warm_generated["construction_ms"]
        + warm_generated["first_render_ms"]
    )
    speedup = cold_activation / max(warm_activation, 1e-9)
    report = {
        "extent": [args.width, args.height],
        "scene": args.scene,
        "cache": str(cache),
        "cache_bytes": cache.stat().st_size if cache.exists() else 0,
        "cold_process_ms": cold_elapsed,
        "warm_process_ms": warm_elapsed,
        "cold_generated_activation_ms": cold_activation,
        "warm_generated_activation_ms": warm_activation,
        "activation_speedup": speedup,
        "minimum_speedup": args.minimum_speedup,
        "cold_metrics": cold["metrics"],
        "warm_metrics": warm["metrics"],
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if args.gate:
        if cold["metrics"]["max_absolute"] != 0.0:
            raise SystemExit("FAIL: cold candidate is not HDR-bit-identical")
        if warm["metrics"]["max_absolute"] != 0.0:
            raise SystemExit("FAIL: cached candidate is not HDR-bit-identical")
        if speedup < args.minimum_speedup:
            raise SystemExit(
                "FAIL: persistent pipeline-cache activation speedup is below "
                f"{args.minimum_speedup:.2f}x"
            )
    print("PASS: persistent Vulkan pipeline-cache startup gate")


if __name__ == "__main__":
    main()
