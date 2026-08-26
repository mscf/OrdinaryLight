"""Compile ordinarylight GLSL shaders to SPIR-V."""

from pathlib import Path
import shutil
import subprocess


ROOT = Path(__file__).resolve().parents[1]


def find_compiler():
    candidates = [
        shutil.which("glslangValidator"),
        ROOT / ".tools/glslang/usr/bin/glslangValidator",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return str(candidate)
    raise RuntimeError(
        "glslangValidator was not found; install glslang-tools or extract it under .tools/glslang"
    )


def main():
    compiler = find_compiler()
    output_dir = ROOT / "ordinarylight/shaders"
    output_dir.mkdir(parents=True, exist_ok=True)
    for source in sorted((ROOT / "ordinarylight/shaders").glob("*.rmiss")):
        output = output_dir / f"{source.name}.spv"
        subprocess.run(
            [
                compiler, "-V", "--target-env", "vulkan1.2", "-S", "rmiss",
                str(source), "-o", str(output),
            ],
            check=True,
        )
        print(f"Wrote {output.relative_to(ROOT)}")
    for source in sorted((ROOT / "ordinarylight/shaders").glob("*.rgen")):
        output = output_dir / f"{source.name}.spv"
        subprocess.run(
            [
                compiler, "-V", "--target-env", "vulkan1.2", "-S", "rgen",
                str(source), "-o", str(output),
            ],
            check=True,
        )
        print(f"Wrote {output.relative_to(ROOT)}")
        if source.stem in {"ser_probe", "wavefront_megakernel"}:
            ser_output = output_dir / f"{source.stem}_ser.rgen.spv"
            subprocess.run(
                [
                    compiler, "-V", "--target-env", "vulkan1.2",
                    "-S", "rgen", "-DWAVE_SER=1", str(source),
                    "-o", str(ser_output),
                ],
                check=True,
            )
            print(f"Wrote {ser_output.relative_to(ROOT)}")
    for source in sorted((ROOT / "ordinarylight/shaders").glob("*.comp")):
        output = output_dir / f"{source.name}.spv"
        subprocess.run(
            [compiler, "-V", "--target-env", "vulkan1.2", "-S", "comp", str(source), "-o", str(output)],
            check=True,
        )
        print(f"Wrote {output.relative_to(ROOT)}")
        if source.stem in {
            "wavefront_primary", "wavefront_hybrid", "wavefront_megakernel",
            "wavefront_persistent", "wavefront_persistent_coarse",
            "wavefront_persistent_continuation", "wavefront_shade",
        }:
            for native in (False, True):
                for profile in (False, True):
                    for overlap, scattering, multiple, skipping in (
                        (True, False, False, False),
                        (False, True, False, False), (True, True, False, False),
                        (False, True, True, False), (True, True, True, False),
                        (False, False, False, True), (True, False, False, True),
                        (False, True, False, True), (True, True, False, True),
                        (False, True, True, True), (True, True, True, True),
                    ):
                        suffix = ("_native" if native else "") \
                            + ("_profile" if profile else "") \
                            + ("_overlap" if overlap else "") \
                            + ("_scatter" if scattering else "") \
                            + ("_multi" if multiple else "") \
                            + ("_skip" if skipping else "")
                        overlap_output = output_dir / (
                            f"{source.stem}{suffix}.comp.spv"
                        )
                        definitions = []
                        if overlap:
                            definitions.append("-DWAVE_OVERLAPPING_VOLUMES=1")
                        if scattering:
                            definitions.append("-DWAVE_VOLUME_SCATTERING=1")
                        if multiple:
                            definitions.append(
                                "-DWAVE_VOLUME_MULTIPLE_SCATTERING=1"
                            )
                        if skipping:
                            definitions.append(
                                "-DWAVE_VOLUME_EMPTY_SPACE_SKIPPING=1"
                            )
                        if native:
                            definitions.append("-DWAVE_NATIVE_TEXTURES=1")
                        if profile:
                            definitions.append("-DWAVE_WORK_COUNTERS=1")
                        subprocess.run(
                            [
                                compiler, "-V", "--target-env", "vulkan1.2",
                                "-S", "comp", *definitions, str(source),
                                "-o", str(overlap_output),
                            ],
                            check=True,
                        )
                        print(f"Wrote {overlap_output.relative_to(ROOT)}")
        if source.stem == "wavefront_reconstruct":
            bgra_output = output_dir / "wavefront_reconstruct_bgra.comp.spv"
            subprocess.run(
                [
                    compiler, "-V", "--target-env", "vulkan1.2", "-S", "comp",
                    "-DWAVE_BGRA_OUTPUT=1", str(source),
                    "-o", str(bgra_output),
                ],
                check=True,
            )
            print(f"Wrote {bgra_output.relative_to(ROOT)}")
        if source.stem in {
            "wavefront_primary", "wavefront_hybrid",
            "wavefront_megakernel", "wavefront_persistent",
            "wavefront_persistent_coarse", "wavefront_shade",
            "wavefront_persistent_continuation",
        }:
            native_output = output_dir / f"{source.stem}_native.comp.spv"
            subprocess.run(
                [
                    compiler, "-V", "--target-env", "vulkan1.2", "-S", "comp",
                    "-DWAVE_NATIVE_TEXTURES=1", str(source),
                    "-o", str(native_output),
                ],
                check=True,
            )
            print(f"Wrote {native_output.relative_to(ROOT)}")
        if source.stem in {
            "wavefront_primary", "wavefront_hybrid", "wavefront_megakernel",
            "wavefront_persistent", "wavefront_persistent_coarse",
            "wavefront_persistent_continuation", "wavefront_shade",
        }:
            for native in (False, True):
                suffix = "_native_profile" if native else "_profile"
                profile_output = output_dir / f"{source.stem}{suffix}.comp.spv"
                definitions = ["-DWAVE_WORK_COUNTERS=1"]
                if native:
                    definitions.append("-DWAVE_NATIVE_TEXTURES=1")
                subprocess.run(
                    [
                        compiler, "-V", "--target-env", "vulkan1.2",
                        "-S", "comp", *definitions, str(source),
                        "-o", str(profile_output),
                    ],
                    check=True,
                )
                print(f"Wrote {profile_output.relative_to(ROOT)}")
        if source.stem in {
            "wavefront_hybrid", "wavefront_persistent_continuation",
        }:
            for native in (False, True):
                suffix = ("_native" if native else "") + "_production"
                specialized_output = output_dir / (
                    f"{source.stem}{suffix}.comp.spv"
                )
                definitions = ["-DWAVE_PRODUCTION_RESTIR=1"]
                if native:
                    definitions.append("-DWAVE_NATIVE_TEXTURES=1")
                subprocess.run(
                    [
                        compiler, "-V", "--target-env", "vulkan1.2",
                        "-S", "comp", *definitions, str(source),
                        "-o", str(specialized_output),
                    ],
                    check=True,
                )
                print(f"Wrote {specialized_output.relative_to(ROOT)}")
            for native in (False, True):
                for profile in (False, True):
                    suffix = ("_native" if native else "") + "_opaque" \
                        + ("_profile" if profile else "")
                    specialized_output = output_dir / (
                        f"{source.stem}{suffix}.comp.spv"
                    )
                    definitions = ["-DWAVE_OPAQUE_SCENE=1"]
                    if native:
                        definitions.append("-DWAVE_NATIVE_TEXTURES=1")
                    if profile:
                        definitions.append("-DWAVE_WORK_COUNTERS=1")
                    subprocess.run(
                        [
                            compiler, "-V", "--target-env", "vulkan1.2",
                            "-S", "comp", *definitions, str(source),
                            "-o", str(specialized_output),
                        ],
                        check=True,
                    )
                    print(f"Wrote {specialized_output.relative_to(ROOT)}")
            if source.stem == "wavefront_hybrid":
                for native in (False, True):
                    suffix = ("_native" if native else "") \
                        + "_opaque_untextured_production"
                    specialized_output = output_dir / (
                        f"{source.stem}{suffix}.comp.spv"
                    )
                    definitions = [
                        "-DWAVE_OPAQUE_SCENE=1",
                        "-DWAVE_UNTEXTURED_SCENE=1",
                        "-DWAVE_PRODUCTION_RESTIR=1",
                    ]
                    if native:
                        definitions.append("-DWAVE_NATIVE_TEXTURES=1")
                    subprocess.run(
                        [
                            compiler, "-V", "--target-env", "vulkan1.2",
                            "-S", "comp", *definitions, str(source),
                            "-o", str(specialized_output),
                        ],
                        check=True,
                    )
                    print(f"Wrote {specialized_output.relative_to(ROOT)}")
        if source.stem == "wavefront_megakernel":
            specialized_output = output_dir / (
                f"{source.stem}_untextured.comp.spv"
            )
            subprocess.run(
                [
                    compiler, "-V", "--target-env", "vulkan1.2",
                    "-S", "comp", "-DWAVE_UNTEXTURED_SCENE=1", str(source),
                    "-o", str(specialized_output),
                ],
                check=True,
            )
            print(f"Wrote {specialized_output.relative_to(ROOT)}")
            for swizzle_width in (8, 16, 32):
                swizzle_output = output_dir / (
                    f"{source.stem}_untextured_swizzle{swizzle_width}.comp.spv"
                )
                subprocess.run(
                    [
                        compiler, "-V", "--target-env", "vulkan1.2",
                        "-S", "comp", "-DWAVE_UNTEXTURED_SCENE=1",
                        f"-DWAVE_GROUP_SWIZZLE_WIDTH={swizzle_width}",
                        str(source), "-o", str(swizzle_output),
                    ],
                    check=True,
                )
                print(f"Wrote {swizzle_output.relative_to(ROOT)}")
            for native in (False, True):
                for profile in (False, True):
                    suffix = ("_native" if native else "") + "_opaque" \
                        + ("_profile" if profile else "")
                    specialized_output = output_dir / (
                        f"{source.stem}{suffix}.comp.spv"
                    )
                    definitions = ["-DWAVE_OPAQUE_SCENE=1"]
                    if native:
                        definitions.append("-DWAVE_NATIVE_TEXTURES=1")
                    if profile:
                        definitions.append("-DWAVE_WORK_COUNTERS=1")
                    subprocess.run(
                        [
                            compiler, "-V", "--target-env", "vulkan1.2",
                            "-S", "comp", *definitions, str(source),
                            "-o", str(specialized_output),
                        ],
                        check=True,
                    )
                    print(f"Wrote {specialized_output.relative_to(ROOT)}")
            for part in ("primary", "secondary"):
                specialized_output = output_dir / (
                    f"{source.stem}_opaque_untextured_{part}.comp.spv"
                )
                other = "SECONDARY" if part == "primary" else "PRIMARY"
                definitions = [
                    "-DWAVE_OPAQUE_SCENE=1",
                    f"-DWAVE_UNTEXTURED_{part.upper()}=1",
                    f"-DWAVE_UNTEXTURED_{other}=0",
                ]
                subprocess.run(
                    [
                        compiler, "-V", "--target-env", "vulkan1.2",
                        "-S", "comp", *definitions, str(source),
                        "-o", str(specialized_output),
                    ],
                    check=True,
                )
                print(f"Wrote {specialized_output.relative_to(ROOT)}")
            for native in (False, True):
                suffix = ("_native" if native else "") \
                    + "_opaque_untextured_production"
                specialized_output = output_dir / (
                    f"{source.stem}{suffix}.comp.spv"
                )
                definitions = [
                    "-DWAVE_OPAQUE_SCENE=1",
                    "-DWAVE_UNTEXTURED_SCENE=1",
                    "-DWAVE_PRODUCTION_RESTIR=1",
                ]
                if native:
                    definitions.append("-DWAVE_NATIVE_TEXTURES=1")
                subprocess.run(
                    [
                        compiler, "-V", "--target-env", "vulkan1.2",
                        "-S", "comp", *definitions, str(source),
                        "-o", str(specialized_output),
                    ],
                    check=True,
                )
                print(f"Wrote {specialized_output.relative_to(ROOT)}")
                for swizzle_width in (8, 16, 32):
                    swizzle_output = output_dir / (
                        f"{source.stem}{suffix}_swizzle{swizzle_width}"
                        ".comp.spv"
                    )
                    swizzle_definitions = definitions + [
                        f"-DWAVE_GROUP_SWIZZLE_WIDTH={swizzle_width}"
                    ]
                    subprocess.run(
                        [
                            compiler, "-V", "--target-env", "vulkan1.2",
                            "-S", "comp", *swizzle_definitions, str(source),
                            "-o", str(swizzle_output),
                        ],
                        check=True,
                    )
                    print(f"Wrote {swizzle_output.relative_to(ROOT)}")
            for native in (False, True):
                for profile in (False, True):
                    suffix = ("_native" if native else "") \
                        + "_opaque_untextured_wg32" \
                        + ("_profile" if profile else "")
                    specialized_output = output_dir / (
                        f"{source.stem}{suffix}.comp.spv"
                    )
                    definitions = [
                        "-DWAVE_OPAQUE_SCENE=1",
                        "-DWAVE_UNTEXTURED_SCENE=1",
                        "-DWAVE_LOCAL_SIZE_X=8",
                        "-DWAVE_LOCAL_SIZE_Y=4",
                    ]
                    if native:
                        definitions.append("-DWAVE_NATIVE_TEXTURES=1")
                    if profile:
                        definitions.append("-DWAVE_WORK_COUNTERS=1")
                    subprocess.run(
                        [
                            compiler, "-V", "--target-env", "vulkan1.2",
                            "-S", "comp", *definitions, str(source),
                            "-o", str(specialized_output),
                        ],
                        check=True,
                    )
                    print(f"Wrote {specialized_output.relative_to(ROOT)}")
            for native in (False, True):
                for profile in (False, True):
                    suffix = ("_native" if native else "") \
                        + "_opaque_untextured" \
                        + ("_profile" if profile else "")
                    specialized_output = output_dir / (
                        f"{source.stem}{suffix}.comp.spv"
                    )
                    definitions = [
                        "-DWAVE_OPAQUE_SCENE=1",
                        "-DWAVE_UNTEXTURED_SCENE=1",
                    ]
                    if native:
                        definitions.append("-DWAVE_NATIVE_TEXTURES=1")
                    if profile:
                        definitions.append("-DWAVE_WORK_COUNTERS=1")
                    subprocess.run(
                        [
                            compiler, "-V", "--target-env", "vulkan1.2",
                            "-S", "comp", *definitions, str(source),
                            "-o", str(specialized_output),
                        ],
                        check=True,
                    )
                    print(f"Wrote {specialized_output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
