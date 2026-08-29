"""Run the formal cross-scene quality, parity, memory, and performance gates."""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from ordinarylight.showcases.rooms import SCENES


STAGES = (
    "indirect", "restir", "parity", "shade_parity",
    "shade_material_parity", "shade_attribute_parity",
    "shade_surface_parity", "termination",
    "pipeline_cache", "strategy_transition", "performance",
)


def _run(command, *, env=None):
    started = time.perf_counter()
    completed = subprocess.run(command, env=env, check=False)
    return completed.returncode, time.perf_counter() - started


def _load(path):
    return json.loads(path.read_text()) if path.exists() else None


def _result(scene, stage, returncode, elapsed, summary_path):
    summary = _load(summary_path)
    return {
        "scene": scene,
        "stage": stage,
        "status": "pass" if returncode == 0 else "fail",
        "returncode": returncode,
        "elapsed_seconds": elapsed,
        "summary": str(summary_path),
        "metrics": summary,
    }


def _indirect(args, scene, output):
    summary = output / "indirect" / scene / "summary.json"
    command = [
        sys.executable, str(Path(__file__).with_name("indirect_quality.py")),
        "--gate", "--scene", scene,
        "--width", str(args.width), "--height", str(args.height),
        "--frames", str(args.frames),
        "--reference-samples", str(args.reference_samples),
        "--bounces", str(args.bounces),
        "--output", str(summary.parent),
    ]
    code, elapsed = _run(command)
    return _result(scene, "indirect", code, elapsed, summary)


def _restir(args, scene, output):
    prefix = output / "restir" / scene
    prefix.parent.mkdir(parents=True, exist_ok=True)
    summary = prefix.with_name(prefix.name + "_summary.json")
    command = [
        sys.executable, str(Path(__file__).with_name("restir_quality.py")),
        "--gate", "--gate-quality-only", "--scene", scene,
        "--candidate-modes", "canonical", "pairwise",
        "--width", str(args.width), "--height", str(args.height),
        "--frames", str(args.frames),
        "--reference-samples", str(args.reference_samples),
        "--bounces", str(args.bounces),
        "--output", str(prefix),
    ]
    code, elapsed = _run(command)
    return _result(scene, "restir", code, elapsed, summary)


def _parity(args, scene, output):
    directory = output / "parity" / scene
    summary = directory / "summary.json"
    command = [
        sys.executable, str(Path(__file__).with_name("execution_parity.py")),
        "--scene", scene, "--width", str(args.width),
        "--height", str(args.height), "--samples", str(args.parity_samples),
        "--bounces", str(args.bounces), "--summary", str(summary),
        "--output-prefix", str(directory / "capture"),
    ]
    code, elapsed = _run(command)
    return _result(scene, "parity", code, elapsed, summary)


def _performance(args, scene, output):
    summary = output / "performance" / f"{scene}.json"
    trace = output / "performance" / f"{scene}.csv"
    summary.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update({
        "WAVE_RENDER_GLFW_PLATFORM": args.performance_platform,
        "WAVE_RENDER_GATE_LOGICAL_WIDTH": str(args.performance_logical_width),
        "WAVE_RENDER_GATE_LOGICAL_HEIGHT": str(args.performance_logical_height),
        "WAVE_RENDER_MAXIMIZE_AFTER_FRAME": "0",
        "WAVE_RENDER_DECORATED": "0",
        "WAVE_RENDER_SCENE": scene,
        "WAVE_RENDER_BENCHMARK_FRAMES": str(args.performance_frames),
        "WAVE_RENDER_BENCHMARK_WARMUP_FRAMES": str(
            args.performance_warmup_frames),
        "WAVE_RENDER_BENCHMARK_ORBITS": str(args.performance_orbits),
        "WAVE_RENDER_BENCHMARK_SUMMARY": str(summary),
        "WAVE_RENDER_BENCHMARK_CSV": str(trace),
        "WAVE_RENDER_INDIRECT_REUSE_STORAGE": "1",
        "WAVE_RENDER_INDIRECT_REUSE_CANDIDATES": "1",
        "WAVE_RENDER_INDIRECT_REUSE_TEMPORAL": "1",
        "WAVE_RENDER_INDIRECT_REUSE_SPATIAL": "1",
        "WAVE_RENDER_INDIRECT_REUSE_APPLY": "1",
        "WAVE_RENDER_DIRECT_SWAPCHAIN": "0",
        "WAVE_RENDER_EXECUTION_STRATEGY": args.performance_strategy,
    })
    command = [str(Path(__file__).with_name("run_4k_performance.sh"))]
    code, elapsed = _run(command, env=env)
    return _result(scene, "performance", code, elapsed, summary)


def _shade_parity(args, scene, output):
    directory = output / "shade_parity" / scene
    summary = directory / "report.json"
    command = [
        sys.executable,
        str(Path(__file__).with_name("ordinaryshade_shade_parity.py")),
        "--gate", "--scene", scene,
        "--width", str(args.width), "--height", str(args.height),
        "--samples", str(args.parity_samples),
        "--bounces", str(args.bounces), "--output", str(directory),
    ]
    if scene == "textured":
        command.append("--native-textures")
    code, elapsed = _run(command)
    return _result(scene, "shade_parity", code, elapsed, summary)


def _shade_material_parity(args, scene, output):
    directory = output / "shade_material_parity" / scene
    summary = directory / "report.json"
    command = [
        sys.executable,
        str(Path(__file__).with_name("ordinaryshade_shade_parity.py")),
        "--gate", "--custom-material", "--scene", scene,
        "--width", str(args.width), "--height", str(args.height),
        "--samples", str(args.parity_samples),
        "--bounces", str(args.bounces), "--output", str(directory),
    ]
    if scene == "textured":
        command.append("--native-textures")
    code, elapsed = _run(command)
    return _result(scene, "shade_material_parity", code, elapsed, summary)


def _shade_attribute_parity(args, scene, output):
    directory = output / "shade_attribute_parity" / scene
    summary = directory / "report.json"
    command = [
        sys.executable,
        str(Path(__file__).with_name("ordinaryshade_shade_parity.py")),
        "--gate", "--custom-attributes", "--scene", scene,
        "--width", str(args.width), "--height", str(args.height),
        "--samples", str(args.parity_samples),
        "--bounces", str(args.bounces), "--output", str(directory),
    ]
    code, elapsed = _run(command)
    return _result(scene, "shade_attribute_parity", code, elapsed, summary)


def _shade_surface_parity(args, scene, output):
    directory = output / "shade_surface_parity"
    summary = directory / "report.json"
    command = [
        sys.executable,
        str(Path(__file__).with_name("ordinaryshade_shade_parity.py")),
        "--gate", "--scene", "feature_parity",
        "--width", str(args.width), "--height", str(args.height),
        "--samples", str(args.parity_samples),
        "--bounces", str(args.bounces), "--output", str(directory),
    ]
    code, elapsed = _run(command)
    return _result(scene, "shade_surface_parity", code, elapsed, summary)


def _pipeline_cache(args, scene, output):
    directory = output / "pipeline_cache"
    summary = directory / "report.json"
    command = [
        sys.executable,
        str(Path(__file__).with_name("pipeline_cache_startup.py")),
        "--gate", "--scene", scene,
        "--width", str(args.width), "--height", str(args.height),
        "--samples", "1", "--bounces", str(args.bounces),
        "--output", str(directory),
    ]
    code, elapsed = _run(command)
    return _result(scene, "pipeline_cache", code, elapsed, summary)


def _strategy_transition(args, scene, output):
    summary = output / "strategy_transition" / "report.json"
    command = [
        sys.executable,
        str(Path(__file__).with_name("strategy_transition.py")),
        "--gate", "--width", str(args.width), "--height", str(args.height),
        "--output", str(summary),
    ]
    code, elapsed = _run(command)
    return _result(scene, "strategy_transition", code, elapsed, summary)


def _termination(args, scene, output):
    directory = output / "termination" / scene
    summary = directory / "summary.json"
    command = [
        sys.executable,
        str(Path(__file__).with_name("path_termination_quality.py")),
        "--gate", "--scene", scene,
        "--width", str(args.width), "--height", str(args.height),
        "--frames", str(args.frames),
        "--reference-samples", str(args.reference_samples),
        "--bounces", str(args.bounces),
        "--roulette-start", "4",
        "--output", str(directory),
    ]
    code, elapsed = _run(command)
    return _result(scene, "termination", code, elapsed, summary)


RUNNERS = {
    "indirect": _indirect,
    "restir": _restir,
    "parity": _parity,
    "shade_parity": _shade_parity,
    "shade_material_parity": _shade_material_parity,
    "shade_attribute_parity": _shade_attribute_parity,
    "shade_surface_parity": _shade_surface_parity,
    "pipeline_cache": _pipeline_cache,
    "strategy_transition": _strategy_transition,
    "termination": _termination,
    "performance": _performance,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenes", nargs="+", choices=tuple(SCENES),
                        default=list(SCENES))
    parser.add_argument("--stages", nargs="+", choices=STAGES,
                        default=list(STAGES))
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=180)
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--reference-samples", type=int, default=16)
    parser.add_argument("--parity-samples", type=int, default=8)
    parser.add_argument("--performance-frames", type=int, default=120)
    parser.add_argument("--performance-warmup-frames", type=int, default=30)
    parser.add_argument(
        "--performance-orbits", type=float, default=1.0,
        help="camera revolutions completed during the performance frames",
    )
    parser.add_argument(
        "--performance-strategy",
        choices=("auto", "wavefront", "hybrid", "megakernel", "persistent"),
        default="auto",
    )
    parser.add_argument(
        "--performance-platform", choices=("x11", "wayland", "native"),
        default="wayland",
    )
    parser.add_argument("--performance-logical-width", type=int, default=3072)
    parser.add_argument("--performance-logical-height", type=int, default=1728)
    parser.add_argument("--bounces", type=int, default=8)
    parser.add_argument(
        "--output", type=Path,
        default=Path("/tmp/ordinarylight_validation_matrix"),
    )
    args = parser.parse_args()
    if args.width < 1 or args.height < 1 or args.frames < 3:
        parser.error("positive dimensions and at least three frames are required")
    if args.reference_samples < 2 or args.parity_samples < 1:
        parser.error("sample counts are too small")
    if args.performance_frames <= args.performance_warmup_frames + 1:
        parser.error("performance capture needs two frames after warmup")
    if args.performance_orbits <= 0.0:
        parser.error("performance orbits must be positive")
    if args.performance_logical_width < 1 or args.performance_logical_height < 1:
        parser.error("performance logical dimensions must be positive")

    args.output.mkdir(parents=True, exist_ok=True)
    results = []
    for scene in args.scenes:
        for stage in args.stages:
            # The custom-attribute gate builds its own purpose-specific scene;
            # run it once rather than once per room-scene label.
            if stage in {"shade_attribute_parity", "shade_surface_parity"} \
                    and scene != args.scenes[0]:
                continue
            if stage == "pipeline_cache" and scene != args.scenes[0]:
                continue
            if stage == "strategy_transition" and scene != args.scenes[0]:
                continue
            print(f"\n=== {scene} / {stage} ===", flush=True)
            result = RUNNERS[stage](args, scene, args.output)
            results.append(result)
            print(f"{result['status'].upper()} in {result['elapsed_seconds']:.1f}s")
    failures = [result for result in results if result["status"] != "pass"]
    report = {
        "status": "fail" if failures else "pass",
        "configuration": {
            "scenes": args.scenes, "stages": args.stages,
            "quality_extent": [args.width, args.height],
            "frames": args.frames,
            "reference_samples": args.reference_samples,
            "parity_samples": args.parity_samples,
            "performance_frames": args.performance_frames,
            "performance_warmup_frames": args.performance_warmup_frames,
            "performance_orbits": args.performance_orbits,
            "performance_strategy": args.performance_strategy,
            "performance_platform": args.performance_platform,
            "performance_logical_extent": [
                args.performance_logical_width,
                args.performance_logical_height,
            ],
            "bounces": args.bounces,
        },
        "results": results,
    }
    report_path = args.output / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nWrote {report_path}")
    if failures:
        raise SystemExit(
            "FAIL: " + ", ".join(
                f"{item['scene']}/{item['stage']}" for item in failures)
        )
    print("PASS: renderer validation matrix")


if __name__ == "__main__":
    main()
