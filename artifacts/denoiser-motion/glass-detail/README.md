# High-contrast glass-motion diagnostic

A standalone fixture places a Fresnel glass sphere before alternating bright,
dark and orange emissive bars. Emission makes the target legible independently
of indirect-light noise. Camera, glass and target translation are tested
separately: 16 moving frames followed by 32 fixed frames. The existing mirror
showcase is unchanged. Mirror guides are disabled and there is no mirror in
this test: it isolates primary-transmission denoising.

The 320x240 test uses the viewer GI configuration (eight bounces, four ReSTIR
streams, evaluated lobes). Independent references average eight 64-sample
batches at the middle and final moving poses; stopped frames share the final
reference pose. Metrics use log luminance in a conservative central patch
x=125:195,y=85:155, inside the sphere throughout the trajectory.

| Motion | Moving-end RMSE | After 32 stopped frames | Reference split RMSE |
| --- | --- | --- | --- |
| Camera | 0.0606 | 0.0169 | 0.0125 |
| Glass | 0.1529 | 0.1305 | 0.0129 |
| Target | 0.1751 | 0.0233 | 0.0148 |

Camera and target errors largely settle. The moving-glass result retains a
large discrepancy after stopping, visible as displaced/smeared refracted bars.
Horizontal gradient magnitude ratios are included in metrics.json as a rough
detail diagnostic; noise and misalignment also affect these ratios, so they
are not a standalone sharpness score. Reference sampling uncertainty remains.

## Controls localize the glass discrepancy

- Starting a fresh denoised renderer at the final glass pose and running 48
  stationary frames yields RMSE 0.0170, versus 0.1305 after the motion sequence.
- Moving the glass through the same sequence without denoising, then collecting
  eight reference batches with matching seeds, matches the independent final
  reference exactly (RMSE 0 in the patch).

This rules out an underlying transformed-scene radiance mismatch for this
control and points to the denoised motion/history path. It does not yet identify
which guide, cached constant, rejection rule or history resource causes it.
No production denoiser settings or algorithms were changed for these captures.
Next inspect glass-motion guide positions, normals, motion, identity and history
lengths against the fresh stationary render, then test targeted invalidation.

## Reproduce

```bash
PYTHONPATH=. .venv/bin/python -m tools.denoiser_motion.glass_detail
PYTHONPATH=. .venv/bin/python artifacts/denoiser-motion/glass-detail/controls.py
```

Requires a working native Vulkan desktop. Raw sequences and reference batches
are written under /tmp/glass-detail; `--output` changes the main diagnostic's
output directory. The control script uses the default directory and runs after
the main diagnostic. The retained contact sheet uses one display transform for
all panels. This is a diagnostic test, not a CI quality threshold.
