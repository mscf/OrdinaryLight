The primary-stage follow-up retained BRDF preparation reuse and two previously
unused specializations in the scene-compiled GI path. They apply automatically
in `raster_feature_viewer`; sampling budgets, bounce counts, and texture defaults
are unchanged. All shader algorithms are typed OrdinaryShade, with generated
GLSL and SPIR-V rebuilt from those sources.

Measurements used the RTX 5090 Laptop GPU, driver 595.91.07, Rough reflections,
2052 × 1764, fixed camera angle −0.45, fresh presenters, 32 warmup frames and 64
measured frames unless stated otherwise. These are medians of GPU timestamps,
not CPU wall-clock times. Both controls and optimized runs were repeated. Full
per-case results, including unfavorable runs, are in
[primary_optimization_results.json](primary_optimization_results.json).

| Preset, repeated run | Before: primary / total GPU ms | After: primary / total GPU ms |
|---|---:|---:|
| Fast GI, 1 SPP / 4 bounces | 4.022 / 15.208 | 3.578 / 14.548 |
| Normal GI, 2 SPP / 8 bounces | 7.107 / 26.614 | 6.657 / 24.555 |

Normal-GI controls repeated after the optimized run measured 7.24–7.42 ms primary
and 25.12–25.62 ms total. The primary improvement persisted; the total-time
variation makes a precise whole-frame percentage less reliable.

These pairs show about 11% and 6% less primary time respectively. Total-frame
changes are not attributable solely to primary: other stages varied between
runs too. Longer 64-warmup/256-measurement controls before adding BRDF reuse
ranged from 3.87–4.17 ms primary and 14.50–15.18 ms total. Treat the measurements
as workload-specific gains, not universal FPS predictions.

The retained changes are:

- Prepare the material coefficients, tangent frame, view cosine, and view-side
  GGX geometry terms once per primary surface, then reuse them across point,
  area, environment, fresh-candidate, and temporal-candidate evaluations.
  The material-only prototype did not help; including the orientation and
  view-dependent terms reduced primary time to 3.53–3.55 ms in its first two
  runs. An immediate cache-disabled comparison measured 3.83–3.92 ms; the final
  cache-enabled repeat measured 3.58 ms. The cache is cleared before continuation.
  Generalized/unified/stratified estimator paths remain uncached: proposal
  evaluation can use a different surface orientation and view.
- Enable the existing opaque primary specialization only for volume-free scenes
  with builtin materials proven nontransmissive, no material modifier, and no
  object effects. Glass, custom programs, modifiers, effects, and volumes retain
  their general path. Eligibility participates in pipeline invalidation.
- Connect the existing production ReSTIR specialization to scene-compiled primary
  pipelines. Disabled optional estimators are removed at compilation. Enabling
  any of those estimators, profiling, or disabling ReSTIR specialization restores
  their general implementation. The selected estimator is unchanged.
- Route scene-compiled pipelines through the same executable-statistics capture
  used for packaged pipelines. Previously the diagnostic omitted the actual GI
  primary and secondary shaders. Pipeline lifecycle logs retain shader hashes.

The other candidates were investigated as follows:

| Experiment | Primary ms, two runs | Decision |
|---|---:|---|
| Unused UV/tangent/material texture guards | 4.31, 4.13 | Reverted: exact HDR parity but no speedup against 3.91–4.02 ms control |
| Material coefficients only | 4.11, 4.05 | Replaced by the more effective full BRDF preparation |
| Skip disabled clearcoat evaluation | 4.07, 3.97 | Reverted: no convincing gain |
| Remove reservoir visibility rays | 3.75, 3.82 | Diagnostic only; reverted because shadows changed, with only ~0.2 ms saved |
| 8×4 rather than 8×8 workgroups, before BRDF reuse | 3.82, 3.69 | Remains diagnostic; too marginal to change production scheduling |
| Native textures with the new specializations | 3.27 | Default unchanged: total 14.14 ms fell within the 13.75–14.58 ms bracketing-control spread |

The texture guards were per-material branches, including skipping tangent loads
without a normal map. They did not remove UVs needed by custom material programs
or the existing normal correction. The shadow-ray ablation removed only selected
reservoir visibility tests, not all light visibility in the renderer. Its HDR
RMSE was about 0.017; it is not an equivalent-quality mode.

Driver statistics also explain why register count alone was not a reliable
optimization target. Primary executable size fell from 690,432 to 309,760 bytes
(about 55%), while reported register count rose from 128 to 164. Reported shared
memory fell from 6,144 to zero bytes. Actual occupancy was not measured. The
raw driver `Local Memory Size` values contained implausibly large values even
for trivial shaders, so they were retained in the JSON but not interpreted as
spill-byte counts. Splitting the shader solely to lower registers is not
justified by these measurements; it would introduce extra intermediate storage
and dispatch work that needs its own comparison.

A depth-one ablation measured 1.50 ms for the remaining hit/material/guide work.
That is not isolated ray-intersection time: it removes lighting, BSDF continuation,
and subsequent bounces. A rasterized-primary alternative would still need
material evaluation and a compatible G-buffer. No rasterized-primary replacement
was implemented in this pass; this measurement does not establish its net gain.
Bulk scene and rendering resources were already device-local, so no additional
memory-placement change was indicated by this investigation.

Validation rebuilt all 458 manifest outputs. The focused specialization,
configuration, preparation, shader-authorship, manifest, and viewer tests passed
105 tests plus 14 subtests. Material, primary, and shared-ReSTIR checks passed
another 26 tests plus two subtests, with one existing test skipped.

At the fixed Rough pose, the final fast-GI HDR had RMSE 8.3e-7 and maximum
absolute difference 0.0004883 against the original control. Normal GI had RMSE
3.8e-8 and maximum difference 0.0001221. Cache-on versus cache-off checks at
320 × 180 also covered refraction, nested dielectrics, clearcoat, anisotropy,
custom material programs, a surface modifier, and multiple-scattering volumes.
The repeated glass, nested-glass, clearcoat, anisotropy, and custom-program images
matched exactly. Modifier and volume differences were no larger than their
repeated-control variation. Earlier specialization-only runs and their larger
occasional differences are preserved in the JSON; individual captures are not
assumed universally deterministic.

To reproduce the comparison from the repository root:

```bash
python components/OrdinaryLight/tools/diagnostics/primary_costs.py \
  --cases baseline repeat --primary-variant surface-only --output /tmp/primary-before
python components/OrdinaryLight/tools/diagnostics/primary_costs.py \
  --cases baseline repeat --reference /tmp/primary-before/repeat.npy --output /tmp/primary-after
```

Add `--gi-mode full` to both commands for normal GI. Use
`--no-primary-brdf-cache` to isolate reuse from the other retained changes,
`--primary-variant opaque` to isolate opaque specialization, and
`--pipeline-statistics --warmup 4 --frames 4` for a separate diagnostic statistics
run. `--primary-workgroup-rows 4` changes only the benchmark's shader and matching
dispatch coverage. The viewer retains 8×8 groups. HDR arrays and prototype source
snapshots from this investigation are under `/tmp/primary-list-*` and
`/tmp/primary-material-cache-source`; the checked-in JSON preserves the numerical
results without committing large images.
