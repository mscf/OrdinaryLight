# Composition validation freeze — 2026-09-08

The fused compute operation is integrated. Feature development is paused for
validation; this is not a declaration that every specialized native path has
been extracted. See the current boundary in `docs/native_gi_composition.md`.

## Correctness

- Full default suite: **709 passed, 163 skipped, 80 subtests passed**.
- Full suite with `ORDINARYLIGHT_TEST_VULKAN_GRAPH=1` and
  `ORDINARYLIGHT_TEST_VULKAN_TRANSPORT=1`: **836 passed, 36 skipped,
  80 subtests passed** in 82.45 seconds, after the parity correction below.
  Raw output is in `gpu_test_results.txt`.
- Direct versus composed primary dispatch: **bit-for-bit identical HDR** for
  eight moving-camera frames with bilinear, FSR1 OrdinaryShade and FSR2. Each
  comparison also exercises a custom application graph and reset to defaults.
- Native lifecycle: FSR1 and FSR2 passed active extent changes, command rebuild,
  resize, resource retirement and close. FSR1 reused commands; FSR2 rebuilt them.
- Final native smoke matrix: glass/opaque scenes, native textures and profiling
  passed tracing, scene rebind and close. Compact indirect reuse passed FSR1/FSR2
  extent/resize lifecycle checks. All five indirect output modes preserved exact
  default/custom graph HDR parity. Qt `glass-detail-motion` viewer startup and
  timed close passed. Raw output: `native_freeze_results.txt`.
- Vulkan validation layers are not installed on this machine, so these runs do
  not constitute a validation-layer-clean result.

The first parity run caught an accidental double dispatch: a pass both dispatched
inside its recorder and supplied workgroups for the graph to dispatch again.
The extra dispatch was removed, the single-dispatch invariant is covered in
`tests/test_primary_operation.py`, and all three HDR comparisons then matched
exactly. Full-suite checks also exposed stale tests referring to pre-extraction
source/layout details; the signal-capture check now exercises adapter behavior.

## Performance methodology

`primary_performance.py` compares the pre-migration direct binding/dispatch
sequence in `primary_baseline.py` with the composed primary operation. Both use
the same current shader binaries and all other current pipeline stages. This
isolates the latest primary migration; it is **not** a comparison against the
entire renderer before the composition work.

Each case uses direct/composed/composed/direct ordering, 12 warmup frames and
30 measured frames per run, static and moving cameras, the glass-detail fixture,
FSR1 OrdinaryShade, and 50% GI render scale. Values below pool the two runs per
path and report medians. Queue-idle waits serialize frames for measurement;
wall latency is not the viewer's pipelined FPS. GPU clocks and other desktop
activity are not isolated.

At 1280×720 output (640×360 GI):

| Case | Direct GPU ms | Composed GPU ms | Direct primary CPU ms | Composed primary CPU ms |
|---|---:|---:|---:|---:|
| Cached, static | 4.909 | 4.990 | 0 | 0 |
| Cached, moving | 4.809 | 4.777 | 0 | 0 |
| Rebuilt, static | 4.626 | 4.717 | 0.140 | 3.990 |
| Rebuilt, moving | 4.732 | 4.820 | 0.143 | 3.964 |

Rebuilt frames contain eight primary calls. Their added CPU cost is about
3.8 ms/frame (0.48 ms/call); total command recording increased from about
28.5–29.2 ms to 32.5–32.8 ms. Cached frames avoid Python operation construction.
The rebuilt GPU delta is approximately 0.09 ms; cached run-to-run variation
precludes a strong small-delta claim. Raw frames are retained in
`primary_performance_cached.json` and `primary_performance_rebuilt.json`.

The next performance target is reusing prepared primary resource bindings and
compiled graph structure on command rebuilds, while retaining fresh single-use
recordings and correct invalidation on scene/resource replacement. FSR2 is
particularly relevant because it does not use the same command replay path.
No optimization is folded into this validation freeze.

At 3840×2160 output (1920×1080 GI), with normal command caching:

| Case | Direct GPU ms | Composed GPU ms | Direct serialized wall ms | Composed serialized wall ms |
|---|---:|---:|---:|---:|
| Static | 54.927 | 55.306 | 56.788 | 59.484 |
| Moving camera | 54.687 | 55.030 | 56.122 | 56.580 |

Raw frames: `primary_performance_4k.json`. The composed GPU delta is about
0.35–0.38 ms for this fixture. Total GPU time remains above the 33.3 ms/frame
budget for 30 FPS, and well above 16.7 ms for 60 FPS. The 4K renderer therefore
still needs substantial performance work; this migration has not met that goal.
These are FSR1 results, not an FSR2 performance assessment.

Reproduce the timing runs with:

```bash
PYTHONPATH=. .venv/bin/python artifacts/denoiser-motion/gi-composition/primary_performance.py --output /tmp/primary-cached.json
PYTHONPATH=. .venv/bin/python artifacts/denoiser-motion/gi-composition/primary_performance.py --force-recording --output /tmp/primary-rebuilt.json
PYTHONPATH=. .venv/bin/python artifacts/denoiser-motion/gi-composition/primary_performance.py --width 3840 --height 2160 --output /tmp/primary-4k.json
```
