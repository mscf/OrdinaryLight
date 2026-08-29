"""Compile built-in Python raster shaders into packaged backend artifacts."""

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
from ordinarylight.shaders.raster_programs import scene_fragment, scene_vertex


OUTPUT = ROOT / "ordinarylight" / "shaders"


def _digest(payload):
    return hashlib.sha256(payload).hexdigest()


def _compile(target):
    options = {"target": target, "validate": True}
    if target == "spirv":
        from ordinarylight.shaders.compiler import find_glsl_compiler

        compiler = find_glsl_compiler()
        if compiler is not None:
            options["spirv_compiler"] = compiler
    vertex = osh.compile(scene_vertex, **options)
    fragment = osh.compile(scene_fragment, **options)
    reflection = osh.link_graphics(vertex, fragment)
    return vertex, fragment, reflection


def build_artifacts():
    manifest = {
        "schema": 1,
        "source": "ordinarylight/shaders/raster_programs.py",
        "source_sha256": _digest(
            (ROOT / "ordinarylight" / "shaders" / "raster_programs.py").read_bytes()
        ),
        "targets": {},
    }
    files = {}
    linked_reflection = None
    for target in ("spirv", "wgsl"):
        vertex, fragment, reflection = _compile(target)
        linked_reflection = reflection
        records = {}
        for name, shader, suffix in (
            ("vertex", vertex, "vert"), ("fragment", fragment, "frag"),
        ):
            filename = (
                f"raster_scene.{suffix}.spv" if target == "spirv"
                else f"raster_scene.{suffix}.wgsl"
            )
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
    manifest["reflection"] = {
        "varyings": [asdict(item) for item in linked_reflection.varyings],
    }
    files["raster_scene.json"] = (
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    expected = build_artifacts()
    failures = []
    for filename, payload in expected.items():
        path = OUTPUT / filename
        if args.check:
            if not path.is_file() or path.read_bytes() != payload:
                failures.append(filename)
        else:
            path.write_bytes(payload)
            print(f"Wrote {path.relative_to(ROOT)}")
    if failures:
        raise SystemExit(
            "generated raster artifacts are stale: " + ", ".join(failures)
        )
    if args.check:
        print("Verified packaged raster shader artifacts")


if __name__ == "__main__":
    main()
