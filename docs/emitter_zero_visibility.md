# Exact-zero native emitter visibility rejection

`sampleAreaLightTechnique` now returns before visibility when all three evaluated
contribution components are exactly zero. The implementation is typed
OrdinaryShade in `shaders/lighting_programs.py`; `transport_v1/lighting.glsl` is
regenerated. Candidate selection, all three random draws, material evaluation,
PDFs, and nonzero contributions are unchanged. There is no distance or brightness
threshold. This shared helper covers native primary and secondary direct-light
sampling. It does not change ReSTIR selection or ordinary triangle emitters.

## Validation (2026-09-14)

- All 459 packaged shader variants compiled successfully.
- 34 upstream tests and 18 subtests pass; 42 vxl8r headless regression tests
  pass (6 native-presentation tests deselected).
- Source execution tests verify random draw order, zero and signed-zero rejection,
  and retention of tiny nonzero, NaN, and infinite contributions. Generation and
  shader authorship checks pass; wavefront/ABI tests pass.
- Headless native GPU comparison: a close camera over a voxel floor and one
  emitter, four camera positions, uniform and power selection, four bounces.
  All 32 identity/raw-HDR/denoised-HDR/face-output arrays match bit for bit.
  A reported sample drops from 4,690 to 3,193 shadow rays with exactly 9,481 path
  rays in both versions. Counters are delayed diagnostics, not per-call timing.
- A separate moving-emitter comparison preserves every primary identity. Three
  poses match exactly; the fractional-cell transition has stochastic lighting
  differences that also occur between repeated runs of identical code. Do not
  use that transition as an exact image-parity oracle.

## Animated native 4K performance

Headless RTX 5090 Laptop GPU, 3840x2160, 512x512 uneven floor, emissive controlled
object and four moving fireflies, GPU animation/layout, cached chunks, tight
bounds, 4 bounces, 1 spp, uniform emitters, roulette disabled, dark-environment
rejection disabled. Each throughput measurement includes 240 frames and final
GPU drain. Before/after runs alternate, with shader compilation finished first.

| Run | Before frame ms | After frame ms | Before primary GPU ms | After primary GPU ms |
| --- | ---: | ---: | ---: | ---: |
| 1 | 31.269 | 30.347 | 10.172 | 9.652 |
| 2 | 31.000 | 30.739 | 10.637 | 9.670 |

Mean frame time improves about 1.9%, from 31.134 to 30.543 ms (32.12 to 32.74 FPS).
Individual gains range from 0.8% to 2.9%; this is a small saving, not a major
throughput improvement. Primary timing improves about 7.1%. GPU stage timings
come from a separate serialized diagnostic interval and are not additive with
throughput. These are headless render measurements, not desktop presentation FPS.
No visual-quality setting or viewer default changes are needed to use the fix.

Logs and fixture scripts: `artifacts/emitter-zero-visibility/`. The fixtures need
vxl8r on PYTHONPATH and an available Vulkan device. They compare separate builds;
saved shader source from before the change is required to recreate a before run.
