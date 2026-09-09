# OrdinaryShade EASU port vs AMD reference

`ordinarylight/shaders/easu.py` ports AMD's FP32 EASU arithmetic into typed
OrdinaryShade functions. The MIT notice and upstream revision are retained
in the module and generated Vulkan shader. The port retains the reciprocal
bit approximations, direction/length estimation, anisotropic 12-tap filter,
accumulation order, and central 2×2 clamping. The renderer loads each RGB tap
directly instead of emulating three separate channel gathers. Tone mapping
and sRGB encoding still occur before filtering. RCAS is disabled.

The viewer keeps both choices: `fsr1` (AMD reference) and `fsr1-shade`
(OrdinaryShade). Defaults are unchanged. The pure port has no AMD header
references and generates GLSL and WGSL; the combined native reconstruction
shader retains the reference branch for A/B testing. No sibling repository
was modified.

## Numerical comparisons

`numerical.py` runs 4,096 cases through AMD FP32 EASU, OrdinaryShade-generated
GLSL/SPIR-V, and OrdinaryShade-generated WGSL using wgpu-native. Cases include
flat black/white, directional steps, ramps, checkerboards, random RGB, and
HDR-derived display-range colors. It checks finite results and central 2×2
bounds. Both generated targets match the reference float output exactly:
maximum absolute error and RMSE are zero in this run. This is a test result,
not a guarantee of bitwise identity on every compiler/GPU.

## Native renderer comparisons

`capture.py` reads actual reconstructed RGBA GPU output for glass, target and
camera motion: 12 moving + 12 settled frames per sequence.

- 640×480 output, 50% scale: every output channel matches exactly across all
  three sequences; internal HDR also matches exactly.
- 639×479 output, two-thirds scale: glass matches exactly. Target and camera
  each have two changed channel values across their 24 frames, each differing
  by one 8-bit level. Internal HDR remains identical. This is consistent with
  small coordinate/arithmetic rounding differences.

`comparison.png` shows final 50% frames side by side. `metrics.json`,
`odd-metrics.json`, and `numerical-metrics.json` record the measurements.

## Provisional GPU cost

4K output, 50% internal scale, one reservoir, moving camera; eight warmup and
sixteen measured frames per run:

| Order | Implementation | Median reconstruction ms | Median GPU frame ms |
| --- | --- | ---: | ---: |
| 1 | AMD | 1.2114 | 16.616 |
| 2 | OrdinaryShade | 1.1257 | 17.052 |
| 3 | OrdinaryShade | 1.1295 | 17.252 |
| 4 | AMD | 1.2142 | 17.057 |

GPU contention is uncontrolled. The filter pass is slightly faster in these
samples; whole-frame times do not establish a speedup. No default was changed.

68 focused tests + 8 subtests pass. Generated-source checks pass and both RGBA
and BGRA reconstruction binaries compile. The pure kernel compiles to GLSL
and WGSL without vendor includes.

## Reproduce

```sh
PYTHONPATH=. .venv/bin/python artifacts/denoiser-motion/easu-shade/numerical.py
PYTHONPATH=. .venv/bin/python artifacts/denoiser-motion/easu-shade/capture.py
SCALE=.6666666666666666 WIDTH=639 HEIGHT=479 OUT=/tmp/easu-odd PYTHONPATH=. .venv/bin/python artifacts/denoiser-motion/easu-shade/capture.py
PYTHONPATH=. .venv/bin/python artifacts/denoiser-motion/easu-shade/profile.py
```

Raw arrays stay in `/tmp/easu-shade` and `/tmp/easu-odd`; floating-point
comparison shaders are in `/tmp/easu-numerical`.

The actual Qt viewer activated `fsr1-shade` through its control and restart,
then completed F11/escape transitions including 1920×1080 internal → 4K
output and a return to 75% windowed rendering without presentation failure.
