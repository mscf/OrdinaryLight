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
from ordinarylight.shaders.raster_programs import (
    blend_raster_surfaces, default_raster_material_hook,
    geometry_product_fragment, geometry_product_vertex,
    scene_fragment, scene_vertex,
    shadow_fragment, shadow_vertex,
)
from ordinarylight.shaders.raster_volume_programs import (
    volume_fragment, volume_vertex,
)


OUTPUT = ROOT / "ordinarylight" / "shaders"


def _digest(payload):
    return hashlib.sha256(payload).hexdigest()


def _compile(target, vertex_program, fragment_program, *, helpers=()):
    options = {"target": target, "validate": True}
    if target == "spirv":
        from ordinarylight.shaders.compiler import find_glsl_compiler

        compiler = find_glsl_compiler()
        if compiler is not None:
            options["spirv_compiler"] = compiler
    vertex = osh.compile(vertex_program, **options)
    fragment = osh.compile(fragment_program, helpers=helpers, **options)
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
        vertex, fragment, reflection = _compile(
            target, scene_vertex, scene_fragment,
            helpers=(blend_raster_surfaces, default_raster_material_hook),
        )
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
        shadow_vertex_result, shadow_fragment_result, shadow_reflection = _compile(
            target, shadow_vertex, shadow_fragment,
        )
        shadow_records = {}
        for name, shader, suffix in (
            ("vertex", shadow_vertex_result, "vert"),
            ("fragment", shadow_fragment_result, "frag"),
        ):
            filename = (
                f"raster_shadow.{suffix}.spv" if target == "spirv"
                else f"raster_shadow.{suffix}.wgsl"
            )
            payload = (
                bytes(shader.binary) if target == "spirv"
                else shader.source.encode("utf-8")
            )
            files[filename] = payload
            shadow_records[name] = {
                "file": filename,
                "sha256": _digest(payload),
                "cache_key": shader.cache_key,
                "reflection": asdict(shader.reflection),
            }
        manifest.setdefault("programs", {})[target] = {
            "shadow": shadow_records,
        }
        manifest.setdefault("program_reflection", {})[target] = {
            "shadow": {
                "varyings": [asdict(item) for item in shadow_reflection.varyings],
            },
        }
        product_vertex, product_fragment, product_reflection = _compile(
            target, geometry_product_vertex, geometry_product_fragment,
        )
        product_records = {}
        for name, shader, suffix in (
            ("vertex", product_vertex, "vert"),
            ("fragment", product_fragment, "frag"),
        ):
            filename = (
                f"raster_geometry_products.{suffix}.spv"
                if target == "spirv"
                else f"raster_geometry_products.{suffix}.wgsl"
            )
            payload = (
                bytes(shader.binary) if target == "spirv"
                else shader.source.encode("utf-8")
            )
            files[filename] = payload
            product_records[name] = {
                "file": filename,
                "sha256": _digest(payload),
                "cache_key": shader.cache_key,
                "reflection": asdict(shader.reflection),
            }
        manifest["programs"][target]["geometry_products"] = product_records
        manifest["program_reflection"][target]["geometry_products"] = {
            "varyings": [asdict(item) for item in product_reflection.varyings],
        }
        volume_vertex_result, volume_fragment_result, volume_reflection = _compile(
            target, volume_vertex, volume_fragment,
        )
        volume_records = {}
        for name, shader, suffix in (
            ("vertex", volume_vertex_result, "vert"),
            ("fragment", volume_fragment_result, "frag"),
        ):
            filename = (
                f"raster_volume.{suffix}.spv"
                if target == "spirv"
                else f"raster_volume.{suffix}.wgsl"
            )
            payload = (
                bytes(shader.binary) if target == "spirv"
                else shader.source.encode("utf-8")
            )
            files[filename] = payload
            volume_records[name] = {
                "file": filename,
                "sha256": _digest(payload),
                "cache_key": shader.cache_key,
                "reflection": asdict(shader.reflection),
            }
        manifest["programs"][target]["volume"] = volume_records
        manifest["program_reflection"][target]["volume"] = {
            "varyings": [asdict(item) for item in volume_reflection.varyings],
        }
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
