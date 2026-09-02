#!/usr/bin/env python3
"""Fetch and build the pinned, optional NRD/NRI reference SDK.

This helper deliberately does not install anything into Ordinary Light.  It
creates a private SDK build that can be consumed by the native
``ordinarylight_nrd`` bridge once that bridge is enabled.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parent


def _run(command, *, cwd=None):
    print("+", " ".join(map(str, command)), flush=True)
    subprocess.run(tuple(map(str, command)), cwd=cwd, check=True)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", type=Path, default=ROOT / "_build")
    parser.add_argument("--config", default="Release")
    parser.add_argument("--jobs", type=int, default=0)
    parser.add_argument("--fetch-only", action="store_true")
    args = parser.parse_args(argv)

    pins = json.loads((ROOT / "pins.json").read_text())
    source = args.work_dir.resolve() / "NRD"
    build = args.work_dir.resolve() / "nrd-build"
    pin = pins["nrd"]
    args.work_dir.mkdir(parents=True, exist_ok=True)
    if not source.exists():
        source.mkdir()
        _run(("git", "init"), cwd=source)
        _run(("git", "remote", "add", "origin", pin["repository"]), cwd=source)
        _run(("git", "fetch", "--depth", "1", "origin", pin["revision"]), cwd=source)
        _run(("git", "checkout", "--detach", "FETCH_HEAD"), cwd=source)
        _run((
            "git", "submodule", "update", "--init", "--recursive",
            "--depth", "1",
        ), cwd=source)
    else:
        actual = subprocess.run(
            ("git", "rev-parse", "HEAD"), cwd=source, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
        if actual != pin["revision"]:
            raise RuntimeError(
                f"existing NRD checkout is {actual}, expected {pin['revision']}"
            )
        _run(("git", "submodule", "update", "--init", "--recursive"), cwd=source)
    if args.fetch_only:
        return 0

    _run((
        "cmake", "-S", source, "-B", build,
        "-DNRD_NRI=ON",
        "-DNRD_STATIC_LIBRARY=ON",
        "-DNRD_EMBEDS_SPIRV_SHADERS=ON",
        "-DNRD_EMBEDS_DXBC_SHADERS=OFF",
        "-DNRD_EMBEDS_DXIL_SHADERS=OFF",
        f"-DCMAKE_BUILD_TYPE={args.config}",
    ))
    command = ["cmake", "--build", build, "--config", args.config]
    if args.jobs:
        command.extend(("--parallel", str(args.jobs)))
    _run(command)
    nri_source = build / "_deps" / "nri-src"
    shadermake = build / "_deps" / "shadermake-src"
    shadermake_library = build / "_deps" / "shadermake-build" / "libShaderMakeBlob.a"
    output = args.work_dir.resolve() / "ordinarylight_nrd_benchmark"
    definitions = (
        "NOMINMAX", "NRD_EMBEDS_DXBC_SHADERS=0",
        "NRD_EMBEDS_DXIL_SHADERS=0", "NRD_EMBEDS_SPIRV_SHADERS=1",
        "NRD_STATIC_LIBRARY=1", "SPIRV_BREG_OFFSET=2",
        "SPIRV_SREG_OFFSET=0", "SPIRV_TREG_OFFSET=20",
        "SPIRV_UREG_OFFSET=3",
    )
    compile_command = ["c++", "-std=c++20", "-O3"]
    compile_command.extend(f"-D{item}" for item in definitions)
    compile_command.extend((
        f"-I{source / 'Include'}", f"-I{source / 'Integration'}",
        f"-I{source / '_Shaders'}", f"-I{nri_source / 'Include'}",
        f"-I{build / '_deps' / 'mathlib-src'}", f"-I{shadermake}",
        str(ROOT / "native" / "benchmark.cpp"),
        str(source / "_Bin" / "libNRD.a"), str(shadermake_library),
        f"-L{source / '_Bin'}", "-lNRI", "-lvulkan", "-ldl", "-lpthread",
        "-Wl,-rpath,$ORIGIN/NRD/_Bin", "-o", str(output),
    ))
    _run(compile_command)
    print(f"Pinned NRD/NRI build is available at {build}")
    print(f"NRD benchmark bridge is available at {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
