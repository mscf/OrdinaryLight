"""Compile portable denoiser kernels from Ordinary Shade source."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import ordinaryshade as osh

from ordinarylight.denoising.kernels import (
    prepare_decode_normal, prepare_previous_pixel, prepare_relax_signals,
    prepare_unpack_normal, relax_atrous, relax_compose, relax_temporal,
)


OUTPUT = ROOT / "ordinarylight" / "shaders"
SOURCE = ROOT / "ordinarylight" / "denoising" / "kernels.py"


def _digest(payload):
    return hashlib.sha256(payload).hexdigest()


def build_artifacts():
    files = {}
    manifest = {
        "schema": 1,
        "source": str(SOURCE.relative_to(ROOT)),
        "source_sha256": _digest(SOURCE.read_bytes()),
        "targets": {},
    }
    for target in ("spirv", "wgsl"):
        options = {"target": target, "validate": True}
        if target == "spirv":
            from ordinarylight.shaders.compiler import find_glsl_compiler

            compiler = find_glsl_compiler()
            if compiler is not None:
                options["spirv_compiler"] = compiler
        records = {}
        for name, kernel, helpers in (
            (
                "prepare", prepare_relax_signals,
                (
                    prepare_decode_normal, prepare_unpack_normal,
                    prepare_previous_pixel,
                ),
            ),
            ("temporal", relax_temporal, ()),
            ("atrous", relax_atrous, ()),
            ("compose", relax_compose, ()),
        ):
            shader = osh.compile(kernel, helpers=helpers, **options)
            suffix = "spv" if target == "spirv" else "wgsl"
            filename = f"denoiser_relax_{name}.comp.{suffix}"
            payload = (
                bytes(shader.binary) if target == "spirv"
                else shader.source.encode("utf-8")
            )
            files[filename] = payload
            records[name] = {
                "file": filename,
                "sha256": _digest(payload),
                "cache_key": shader.cache_key,
                "reflection": asdict(shader.reflection),
            }
        manifest["targets"][target] = records
    files["denoiser_relax.json"] = (
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    failures = []
    for filename, payload in build_artifacts().items():
        path = OUTPUT / filename
        if args.check:
            if not path.is_file() or path.read_bytes() != payload:
                failures.append(filename)
        else:
            path.write_bytes(payload)
            print(f"Wrote {path.relative_to(ROOT)}")
    if failures:
        raise SystemExit(
            "generated denoiser artifacts are stale: " + ", ".join(failures)
        )
    if args.check:
        print("Verified packaged denoiser shader artifacts")


if __name__ == "__main__":
    main()
