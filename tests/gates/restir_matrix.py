"""Run the ReSTIR quality gate across scenes and camera-motion cases."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from ordinarylight.showcases.rooms import SCENES


CASES = {"stationary": 0.0, "moving": 0.35, "fast": 2.8}


def build_command(args, scene, case, prefix):
    command = [
        sys.executable, str(Path(__file__).with_name("restir_quality.py")),
        "--gate", "--gate-quality-only", "--scene", scene,
        "--candidate-modes", "canonical", "pairwise",
        "--motion-radians", str(CASES[case]),
        "--width", str(args.width), "--height", str(args.height),
        "--frames", str(args.frames),
        "--reference-samples", str(args.reference_samples),
        "--bounces", str(args.bounces),
        "--strategy", getattr(args, "strategy", "megakernel"),
        "--gate-max-abs-bias", str(args.gate_max_abs_bias),
        "--gate-max-mae-ratio", str(args.gate_max_mae_ratio),
        "--generalized-balance-cap", str(args.generalized_balance_cap),
        "--gate-max-generalized-gpu-ratio",
        str(args.gate_max_generalized_gpu_ratio),
        "--output", str(prefix),
    ]
    if case == "fast":
        command.append("--require-motion-rejection")
    if args.include_generalized:
        insert_at = command.index("--motion-radians")
        command.insert(insert_at, "generalized")
    if getattr(args, "unified_secondary_nee", False):
        command.append("--unified-secondary-nee")
    if getattr(args, "unified_primary_restir", False):
        command.append("--unified-primary-restir")
    if getattr(args, "stratified_primary_restir", False):
        command.append("--stratified-primary-restir")
    if getattr(args, "material_bucketing", False):
        command.extend((
            "--material-bucketing",
            "--material-bucketing-start-bounce",
            str(args.material_bucketing_start_bounce),
        ))
    if getattr(args, "persistent_coarse_tiles", False):
        command.append("--persistent-coarse-tiles")
    if getattr(args, "persistent_continuations", False):
        command.append("--persistent-continuations")
    if getattr(args, "no_scene_specialization", False):
        command.append("--no-scene-specialization")
    return command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenes", nargs="+", choices=tuple(SCENES),
                        default=list(SCENES))
    parser.add_argument("--cases", nargs="+", choices=tuple(CASES),
                        default=list(CASES))
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=180)
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--reference-samples", type=int, default=16)
    parser.add_argument("--bounces", type=int, default=8)
    parser.add_argument(
        "--strategy", choices=(
            "wavefront", "hybrid", "megakernel", "persistent"
        ),
        default="megakernel",
    )
    parser.add_argument("--material-bucketing", action="store_true")
    parser.add_argument(
        "--material-bucketing-start-bounce", type=int, default=2,
    )
    parser.add_argument("--persistent-coarse-tiles", action="store_true")
    parser.add_argument("--persistent-continuations", action="store_true")
    parser.add_argument("--no-scene-specialization", action="store_true")
    parser.add_argument(
        "--include-generalized", action="store_true",
        help="also gate the experimental generalized-MIS estimator",
    )
    parser.add_argument(
        "--unified-secondary-nee", action="store_true",
        help="gate the unified secondary area/environment sampler",
    )
    parser.add_argument(
        "--unified-primary-restir", action="store_true",
        help="gate unified primary area/environment ReSTIR",
    )
    parser.add_argument(
        "--stratified-primary-restir", action="store_true",
        help="gate independent primary area/environment reservoirs",
    )
    parser.add_argument("--gate-max-abs-bias", type=float, default=0.0125)
    parser.add_argument("--gate-max-mae-ratio", type=float, default=1.04)
    parser.add_argument("--generalized-balance-cap", type=float, default=2.0)
    parser.add_argument(
        "--gate-max-generalized-gpu-ratio", type=float, default=2.0,
        help="experimental generalized/canonical GPU budget",
    )
    parser.add_argument(
        "--output-dir", default="/tmp/ordinarylight_restir_matrix"
    )
    args = parser.parse_args()
    if args.frames < 2:
        parser.error("--frames must be at least 2")
    if args.width < 1 or args.height < 1:
        parser.error("capture dimensions must be positive")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for scene in args.scenes:
        for case in args.cases:
            prefix = output_dir / f"{scene}_{case}"
            command = build_command(args, scene, case, prefix)
            print(f"\n=== {scene} / {case} ===", flush=True)
            completed = subprocess.run(command, check=False)
            summary_path = prefix.with_name(prefix.name + "_summary.json")
            summary = None
            if summary_path.exists():
                summary = json.loads(summary_path.read_text())
            results.append({
                "scene": scene, "case": case,
                "status": "pass" if completed.returncode == 0 else "fail",
                "returncode": completed.returncode,
                "summary": str(summary_path),
                "gate": None if summary is None else summary.get("gate"),
                "quality": None if summary is None else summary.get("quality"),
                "gpu_ms_mean": None if summary is None else summary.get("gpu_ms_mean"),
            })

    failures = [item for item in results if item["status"] != "pass"]
    report = {
        "status": "fail" if failures else "pass",
        "configuration": {
            "width": args.width, "height": args.height,
            "frames": args.frames,
            "reference_samples": args.reference_samples,
            "bounces": args.bounces,
            "include_generalized": args.include_generalized,
            "gate_max_abs_bias": args.gate_max_abs_bias,
            "gate_max_mae_ratio": args.gate_max_mae_ratio,
            "generalized_balance_cap": args.generalized_balance_cap,
            "gate_max_generalized_gpu_ratio":
                args.gate_max_generalized_gpu_ratio,
        },
        "cases": results,
    }
    report_path = output_dir / "matrix_summary.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nMatrix summary -> {report_path}")
    if failures:
        print(f"FAIL: {len(failures)} of {len(results)} matrix cases failed")
        raise SystemExit(1)
    print(f"PASS: all {len(results)} ReSTIR matrix cases passed")


if __name__ == "__main__":
    main()
