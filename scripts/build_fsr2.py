"""Build the optional Linux FSR 2 Vulkan bridge with fixed FP32 permutations."""

from pathlib import Path
import subprocess, struct, json
from compile_shaders import find_compiler

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "native/fsr2"
API = SRC / "upstream"
OUT = ROOT / ".tools/fsr2"
passes = [
    "depth_clip",
    "reconstruct_previous_depth",
    "lock",
    "accumulate",
    "accumulate",
    "rcas",
    "compute_luminance_pyramid",
    "autogen_reactive",
    "tcr_autogen",
]


def reflect(data):
    w = struct.unpack("<%dI" % (len(data) // 4), data)
    names = {}
    bindings = {}
    types = {}
    variables = []
    i = 5
    while i < len(w):
        n = w[i] >> 16
        op = w[i] & 65535
        a = w[i + 1 : i + n]
        i += n
        if op == 5:
            names[a[0]] = (
                struct.pack("<%dI" % (len(a) - 1), *a[1:]).split(b"\0")[0].decode()
            )
        if op == 71 and a[1] == 33:
            bindings[a[0]] = a[2]
        if op in (25, 30, 32):
            types[a[0]] = (op, a[1:])
        if op == 59:
            variables.append(a[:3])
    groups = [[], [], []]
    for typ, ident, storage in variables:
        if ident not in bindings:
            continue
        pointee = types[typ][1][1]
        kind, info = types.get(pointee, (0, ()))
        if storage == 2:
            g = 2
        elif kind == 25:
            g = 0 if info[5] == 2 else 1
        else:
            continue
        name = names.get(ident, "")
        groups[g].append((name, bindings[ident]))
    return groups


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cpp = ['#include "upstream/vk/shaders/ffx_fsr2_shaders_vk.h"']
    for i, p in enumerate(passes):
        target = OUT / f"{i}.spv"
        flags = {
            "FFX_GPU": 1,
            "FFX_GLSL": 1,
            "FFX_HALF": 0,
            "FFX_FSR2_OPTION_HDR_COLOR_INPUT": 1,
            "FFX_FSR2_OPTION_LOW_RESOLUTION_MOTION_VECTORS": 1,
            "FFX_FSR2_OPTION_INVERTED_DEPTH": 1,
            "FFX_FSR2_OPTION_JITTERED_MOTION_VECTORS": 0,
            "FFX_FSR2_OPTION_REPROJECT_USE_LANCZOS_TYPE": 0,
            "FFX_FSR2_OPTION_APPLY_SHARPENING": int(i == 4),
        }
        subprocess.run(
            [
                find_compiler(),
                "-V",
                "--target-env",
                "vulkan1.1",
                "-S",
                "comp",
                "-Os",
                *[f"-D{k}={v}" for k, v in flags.items()],
                str(API / f"shaders/ffx_fsr2_{p}_pass.glsl"),
                "-o",
                str(target),
            ],
            check=True,
        )
        data = target.read_bytes()
        groups = reflect(data)
        cpp.append(
            f"alignas(4) static const uint8_t data{i}[]={{"
            + ",".join(map(str, data))
            + "};"
        )
        for j, g in enumerate(groups):
            cpp.append(
                f"static const char* names{i}_{j}[]={{"
                + (",".join(json.dumps(n) for n, b in g) or "nullptr")
                + "};"
            )
            cpp.append(
                f"static const uint32_t bindings{i}_{j}[]={{"
                + (",".join(str(b) for n, b in g) or "0")
                + "};"
            )
        cpp.append(
            f"static Fsr2ShaderBlobVK blob{i}={{data{i},sizeof(data{i}),"
            + ",".join(str(len(g)) for g in groups)
            + ","
            + ",".join(f"names{i}_{j},bindings{i}_{j}" for j in range(3))
            + "};"
        )
    cpp.append(
        'extern "C" Fsr2ShaderBlobVK fsr2GetPermutationBlobByIndexVK(FfxFsr2Pass pass,uint32_t flags) { switch(pass) {'
        + "".join(f"case {i}: return blob{i};" for i in range(9))
        + "default: return {};}}"
    )
    (OUT / "shaders.cpp").write_text("\n".join(cpp))
    subprocess.run(
        [
            "g++",
            "-std=c++17",
            "-O2",
            "-fPIC",
            "-shared",
            "-DNDEBUG",
            "-include",
            str(SRC / "compat.h"),
            "-I" + str(SRC),
            str(API / "ffx_fsr2.cpp"),
            str(API / "ffx_assert.cpp"),
            str(API / "vk/ffx_fsr2_vk.cpp"),
            str(OUT / "shaders.cpp"),
            str(SRC / "bridge.cpp"),
            "-lvulkan",
            "-o",
            str(OUT / "libordinarylight_fsr2.so"),
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
