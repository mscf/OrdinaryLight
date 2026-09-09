# Optional Linux FSR 2 Vulkan bridge

The unmodified `upstream/` subtree is AMD's `src/ffx-fsr2-api` from
https://github.com/GPUOpen-Effects/FidelityFX-FSR2 at revision
`1680d1edd5c034f88ebbbb793d8b88f8842cf804`. MIT notices are retained in the
source files. `compat.h` supplies Linux equivalents for the MSVC declarations
used by the upstream host code.

From the checkout, with g++, Vulkan development headers/libraries and
`glslangValidator` (or the existing `.tools/glslang` extraction):

```sh
.venv/bin/python scripts/build_fsr2.py
```

This compiles FP32 GLSL shaders and the upstream host/Vulkan backend into
`.tools/fsr2/libordinarylight_fsr2.so`. The bridge uses fixed HDR, low-resolution
unjittered motion, inverted finite depth permutations. Sharpening is disabled.
SPIR-V resource bindings/names are reflected directly by the build script,
replacing upstream's Windows-only precompiled permutation generator. The
upstream algorithm itself is not substituted.

Set `ORDINARYLIGHT_FSR2_LIBRARY` to the absolute library path when using an
installed wheel or another checkout. The optional native library is not
bundled into the pure-Python wheel. Other upscalers do not require it.
