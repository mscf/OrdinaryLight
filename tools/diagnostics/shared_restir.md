# Shared-primary ReSTIR experiment

Enable **Shared primary ReSTIR (experimental)** in the raster workbench's GI
view, choose **Path samples per pixel**, and apply/restart. The existing ReSTIR
reservoir control then selects direct-light samples per primary hit, independently
of the number of complete paths. The default remains the legacy path.

From the repository root:

```bash
python components/OrdinaryLight/tools/raster_feature_viewer.py \
  --target wavefront-gi --showcase optical-screen-rough-reflection \
  --shared-primary-restir --path-spp 1
```

Each shared primary hit performs intersection, material evaluation, and indirect
continuation once. Its direct-light reservoirs have separate selection, visibility
tests, and history slots; only their direct contributions are averaged. Multiple
path samples have separate reservoir storage. Storage scales with path SPP times
reservoir count; shared mode is not a memory-saving option.

The prototype requires a GLSL compiler and fixed SPP with the wavefront strategy.
Spatial reuse is rejected. At 1 SPP, it rejects geometry-incompatible history and
the viewer allows 20 represented candidates instead of the legacy budget of four
(which leaves no space beyond four fresh candidates). At higher SPP, reservoir
temporal reuse is disabled: current per-pixel guides describe only the last
jittered hit and cannot safely validate every path's history. Temporal denoising
remains available. Existing half-precision reservoir limitations are unchanged.
First-use shader/pipeline compilation and Apply/restart latency are not addressed
by this experiment.

## Measured comparison

Measured on the RTX 5090 Laptop GPU, 2026-09-09. Fixed 960 × 540 output,
eight bounces, four fresh candidates per reservoir, denoising off, temporal
reservoir reuse explicitly disabled, 32 warm-up frames and 64 measured frames.
Both modes compile the current GLSL, including for built-in-only materials.
Times are median GPU timestamps, not viewer FPS or total restart time.

| Scene | Mode | Reservoirs | Complete paths/pixel | GPU ms |
|---|---|---:|---:|---:|
| Emitter room | Legacy | 1 | 1 | 5.02 |
| Emitter room | Legacy | 2 | 2 | 9.47 |
| Emitter room | Legacy | 4 | 4 | 18.81 |
| Emitter room | Shared | 1 | 1 | 4.93 |
| Emitter room | Shared | 2 | 1 | 5.26 |
| Emitter room | Shared | 4 | 1 | 7.17 |
| Emitter room | Shared | 2 | 2 | 11.33 |
| Rough reflections | Legacy | 1 | 1 | 7.28 |
| Rough reflections | Legacy | 2 | 2 | 14.23 |
| Rough reflections | Legacy | 4 | 4 | 27.74 |
| Rough reflections | Shared | 1 | 1 | 7.16 |
| Rough reflections | Shared | 2 | 1 | 7.32 |
| Rough reflections | Shared | 4 | 1 | 7.54 |
| Rough reflections | Shared | 2 | 2 | 14.36 |

Four shared reservoirs were 2.62× faster in the emitter room and 3.68× faster
in Rough reflections than four legacy paths. These are not equal-quality
comparisons: legacy also takes four camera/indirect-path samples.

One-reservoir first frames matched exactly. Increasing shared reservoirs kept
the path tile count at four; two path SPP doubled it to eight. HDR outputs were
finite and mean radiance remained within 0.05% across the 1/2/4 reservoir cases.
These checks do not establish unbiasedness for arbitrary scenes.

Whole-image stationary temporal variance in the emitter room fell only from
0.0963 to 0.0923 with four shared reservoirs, versus 0.0240 for four legacy paths.
In Rough reflections it did not improve (3.5421 to 3.6493); two path SPP reduced
it to 1.8040. This metric includes indirect noise and camera-jitter aliasing,
which more direct-light reservoirs cannot remove. More path SPP is the useful
quality control for those components. The eight-SPP reference in the JSON report
is independently seeded but remains noisy, not ground truth.

## Reproduce

From `components/OrdinaryLight`:

```bash
python -m tools.diagnostics.shared_restir --width 960 --height 540 \
  --output /tmp/shared-restir-emitter
python -m tools.diagnostics.shared_restir --width 960 --height 540 \
  --showcase optical-screen-rough-reflection --output /tmp/shared-restir-rough
python -m tools.diagnostics.shared_restir --width 512 --height 384 \
  --showcase optical-screen-rough-reflection --frames 32 --warmup 8 \
  --history-limit 20 --motion 0.15 --output /tmp/shared-restir-motion
```

The test window closes automatically. Reports contain timings, variance, mean
radiance, and first-frame parity; mean HDR images are saved as NumPy arrays.
Assertions check finite output, normalization, single-reservoir parity, and path
work scaling. Motion-run variance includes changing scene content and should not
be interpreted as pure noise.
