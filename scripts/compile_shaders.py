"""Build and validate the manifest-owned Ordinary Light SPIR-V inventory."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import fnmatch
import json
from pathlib import Path
import shutil
import subprocess


ROOT = Path(__file__).resolve().parents[1]
SHADER_DIR = ROOT / "ordinarylight" / "shaders"
MANIFEST_PATH = SHADER_DIR / "manifest.json"


@dataclass(frozen=True)
class ShaderBuild:
    source: Path
    output: Path
    stage: str
    definitions: tuple[str, ...] = ()

    @property
    def name(self):
        return self.output.name


def load_manifest(path=MANIFEST_PATH):
    manifest = json.loads(Path(path).read_text())
    if manifest.get("schema") != 1:
        raise ValueError("unsupported shader manifest schema")
    return manifest


def find_compiler():
    candidates = [
        shutil.which("glslangValidator"),
        ROOT / ".tools/glslang/usr/bin/glslangValidator",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return str(candidate)
    raise RuntimeError(
        "glslangValidator was not found; install glslang-tools or extract it "
        "under .tools/glslang"
    )


def _build(source, output, stage, *definitions):
    return ShaderBuild(
        SHADER_DIR / source,
        SHADER_DIR / output,
        stage,
        tuple(definitions),
    )


def _compute_variant(stem, suffix, definitions):
    return _build(
        f"{stem}.comp", f"{stem}{suffix}.comp.spv", "comp", *definitions
    )


def build_plan(manifest=None):
    """Expand the compact manifest into a deterministic compiler plan."""
    manifest = manifest or load_manifest()
    families = manifest["families"]
    plan = []

    stages = {".comp": "comp", ".rgen": "rgen", ".rmiss": "rmiss"}
    for source in sorted(
        path for path in SHADER_DIR.iterdir() if path.suffix in stages
    ):
        plan.append(_build(
            source.name, f"{source.name}.spv", stages[source.suffix]
        ))

    for stem in families["ser"]:
        plan.append(_build(
            f"{stem}.rgen", f"{stem}_ser.rgen.spv", "rgen", "WAVE_SER=1"
        ))

    volume_combinations = (
        (True, False, False, False),
        (False, True, False, False),
        (True, True, False, False),
        (False, True, True, False),
        (True, True, True, False),
        (False, False, False, True),
        (True, False, False, True),
        (False, True, False, True),
        (True, True, False, True),
        (False, True, True, True),
        (True, True, True, True),
    )
    for stem in families["volume_matrix"]:
        for native in (False, True):
            for profile in (False, True):
                for overlap, scattering, multiple, skipping in volume_combinations:
                    suffix = (
                        ("_native" if native else "")
                        + ("_profile" if profile else "")
                        + ("_overlap" if overlap else "")
                        + ("_scatter" if scattering else "")
                        + ("_multi" if multiple else "")
                        + ("_skip" if skipping else "")
                    )
                    definitions = []
                    if overlap:
                        definitions.append("WAVE_OVERLAPPING_VOLUMES=1")
                    if scattering:
                        definitions.append("WAVE_VOLUME_SCATTERING=1")
                    if multiple:
                        definitions.append("WAVE_VOLUME_MULTIPLE_SCATTERING=1")
                    if skipping:
                        definitions.append("WAVE_VOLUME_EMPTY_SPACE_SKIPPING=1")
                    if native:
                        definitions.append("WAVE_NATIVE_TEXTURES=1")
                    if profile:
                        definitions.append("WAVE_WORK_COUNTERS=1")
                    plan.append(_compute_variant(stem, suffix, definitions))

    for stem in families["native"]:
        plan.append(_compute_variant(
            stem, "_native", ("WAVE_NATIVE_TEXTURES=1",)
        ))
    for stem in families["profile"]:
        plan.append(_compute_variant(
            stem, "_profile", ("WAVE_WORK_COUNTERS=1",)
        ))
        plan.append(_compute_variant(
            stem,
            "_native_profile",
            ("WAVE_WORK_COUNTERS=1", "WAVE_NATIVE_TEXTURES=1"),
        ))

    for stem in families["production"]:
        for native in (False, True):
            suffix = ("_native" if native else "") + "_production"
            definitions = ["WAVE_PRODUCTION_RESTIR=1"]
            if native:
                definitions.append("WAVE_NATIVE_TEXTURES=1")
            plan.append(_compute_variant(stem, suffix, definitions))

    for stem in families["opaque"]:
        for native in (False, True):
            for profile in (False, True):
                suffix = (
                    ("_native" if native else "")
                    + "_opaque"
                    + ("_profile" if profile else "")
                )
                definitions = ["WAVE_OPAQUE_SCENE=1"]
                if native:
                    definitions.append("WAVE_NATIVE_TEXTURES=1")
                if profile:
                    definitions.append("WAVE_WORK_COUNTERS=1")
                plan.append(_compute_variant(stem, suffix, definitions))

    for stem in families["opaque_untextured_production"]:
        for native in (False, True):
            suffix = (
                ("_native" if native else "")
                + "_opaque_untextured_production"
            )
            definitions = [
                "WAVE_OPAQUE_SCENE=1",
                "WAVE_UNTEXTURED_SCENE=1",
                "WAVE_PRODUCTION_RESTIR=1",
            ]
            if native:
                definitions.append("WAVE_NATIVE_TEXTURES=1")
            plan.append(_compute_variant(stem, suffix, definitions))

    for stem in families["megakernel"]:
        plan.append(_compute_variant(
            stem, "_untextured", ("WAVE_UNTEXTURED_SCENE=1",)
        ))
        for width in (8, 16, 32):
            plan.append(_compute_variant(
                stem,
                f"_untextured_swizzle{width}",
                ("WAVE_UNTEXTURED_SCENE=1", f"WAVE_GROUP_SWIZZLE_WIDTH={width}"),
            ))
        for part in ("primary", "secondary"):
            other = "SECONDARY" if part == "primary" else "PRIMARY"
            plan.append(_compute_variant(
                stem,
                f"_opaque_untextured_{part}",
                (
                    "WAVE_OPAQUE_SCENE=1",
                    f"WAVE_UNTEXTURED_{part.upper()}=1",
                    f"WAVE_UNTEXTURED_{other}=0",
                ),
            ))
        for native in (False, True):
            suffix = (
                ("_native" if native else "")
                + "_opaque_untextured_production"
            )
            definitions = [
                "WAVE_OPAQUE_SCENE=1",
                "WAVE_UNTEXTURED_SCENE=1",
                "WAVE_PRODUCTION_RESTIR=1",
            ]
            if native:
                definitions.append("WAVE_NATIVE_TEXTURES=1")
            for width in (8, 16, 32):
                plan.append(_compute_variant(
                    stem,
                    f"{suffix}_swizzle{width}",
                    definitions + [f"WAVE_GROUP_SWIZZLE_WIDTH={width}"],
                ))
        for native in (False, True):
            for profile in (False, True):
                prefix = "_native" if native else ""
                profile_suffix = "_profile" if profile else ""
                definitions = ["WAVE_OPAQUE_SCENE=1", "WAVE_UNTEXTURED_SCENE=1"]
                if native:
                    definitions.append("WAVE_NATIVE_TEXTURES=1")
                if profile:
                    definitions.append("WAVE_WORK_COUNTERS=1")
                plan.append(_compute_variant(
                    stem,
                    f"{prefix}_opaque_untextured{profile_suffix}",
                    definitions,
                ))
                plan.append(_compute_variant(
                    stem,
                    f"{prefix}_opaque_untextured_wg32{profile_suffix}",
                    definitions + ["WAVE_LOCAL_SIZE_X=8", "WAVE_LOCAL_SIZE_Y=4"],
                ))

    for stem in families["bgra"]:
        plan.append(_compute_variant(
            stem, "_bgra", ("WAVE_BGRA_OUTPUT=1",)
        ))

    by_output = {}
    for build in plan:
        previous = by_output.setdefault(build.name, build)
        if previous != build:
            raise ValueError(f"conflicting shader builds for {build.name}")
    return tuple(by_output.values())


def validate_plan(plan=None):
    plan = plan or build_plan()
    missing_sources = sorted({
        str(build.source.relative_to(ROOT))
        for build in plan if not build.source.is_file()
    })
    expected = {build.name for build in plan}
    actual = {path.name for path in SHADER_DIR.glob("*.spv")}
    return {
        "planned": len(expected),
        "missing_sources": missing_sources,
        "missing_outputs": sorted(expected - actual),
        "unmanaged_outputs": sorted(actual - expected),
    }


def compile_plan(plan, *, output_dir, compiler, target_environment):
    output_dir.mkdir(parents=True, exist_ok=True)
    for build in plan:
        output = output_dir / build.name
        command = [
            compiler,
            "-V",
            "--target-env",
            target_environment,
            "-S",
            build.stage,
            *(f"-D{definition}" for definition in build.definitions),
            str(build.source),
            "-o",
            str(output),
        ]
        subprocess.run(command, check=True)
        print(f"Wrote {output.relative_to(ROOT) if output.is_relative_to(ROOT) else output}")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--only", help="compile/list outputs matching this glob")
    parser.add_argument("--output-dir", type=Path, default=SHADER_DIR)
    args = parser.parse_args(argv)
    manifest = load_manifest()
    plan = build_plan(manifest)
    if args.only:
        plan = tuple(build for build in plan if fnmatch.fnmatch(build.name, args.only))
        if not plan:
            raise SystemExit(f"no shader outputs match {args.only!r}")
    if args.list:
        for build in plan:
            definitions = " ".join(f"-D{name}" for name in build.definitions)
            print(f"{build.name}: {build.source.name} {definitions}".rstrip())
        return
    if args.check:
        report = validate_plan(plan)
        print(json.dumps(report, indent=2))
        if any(report[name] for name in (
            "missing_sources", "missing_outputs", "unmanaged_outputs"
        )):
            raise SystemExit(1)
        return
    compile_plan(
        plan,
        output_dir=args.output_dir,
        compiler=find_compiler(),
        target_environment=manifest["target_environment"],
    )


if __name__ == "__main__":
    main()
