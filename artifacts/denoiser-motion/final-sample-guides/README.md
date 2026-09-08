# Final-sample geometry guides

ReLAX consumes geometry guides after the complete sample batch. Previously
all four samples reconstructed and wrote motion, depth, normals and identity,
overwriting earlier samples. The prepare kernel now returns after radiance
accumulation for non-final samples. Every sample's diffuse/specular radiance
still contributes, and hit distances remain the last sample's values. Single-
sample behavior is unchanged. The portable WGSL and packaged SPIR-V were
regenerated from the same Ordinary Shade source.

The optimization does not skip tracing, change sample counts, or lower
resolution. It does not skip the per-sample radiance classification work.
Invalid-final-surface identity texels are not promised bit-identical: temporal
reprojection rejects those texels by depth/validity before using identity.

## Validation

Full 48-frame HDR sequences at 320×240 were byte-identical for glass motion,
background motion, and camera motion (16 moving + 32 stationary frames each).
32 focused denoising/motion/fixture tests passed. No native validation layer is
available on this machine.

Sequential isolated 4K direct-presentation runs (12 frames per phase after
8 warm-up frames; medians omit first two frames):

| Cost | Before | After |
| --- | ---: | ---: |
| Moving guide preparation, mean GPU ms | 51.17 | 40.92 |
| Stationary guide preparation, mean GPU ms | 47.91 | 37.03 |
| Moving total GPU, median ms | 342.41 | 336.31 |
| Stationary total GPU, median ms | 336.87 | 333.93 |
| Moving submission, median ms | 353.50 | 343.97 |
| Stationary submission, median ms | 332.47 | 333.10 |

Guide cost falls 20–23%; end-to-end improvement is small (roughly 1–2% GPU).
Absolute timings vary from earlier 4K runs; use paired measurements rather
than comparing different desktop sessions. The first non-isolated run was
inconclusive and is not used as the speedup claim. No 30/60 FPS claim follows.

From repository root:

```sh
PYTHONPATH=. .venv/bin/python artifacts/denoiser-motion/final-sample-guides/check_parity.py
OLD_GUIDES=1 DIRECT=1 PROFILE=0 OUT=/tmp/guides-before PYTHONPATH=. .venv/bin/python artifacts/denoiser-motion/final-sample-guides/profile.py
DIRECT=1 PROFILE=0 OUT=/tmp/guides-after PYTHONPATH=. .venv/bin/python artifacts/denoiser-motion/final-sample-guides/profile.py
```

The preserved baseline shader allows comparisons without changing working
files. Native Vulkan/display access is required. Further gains must target
primary/secondary tracing and per-sample radiance/resolve bandwidth.
