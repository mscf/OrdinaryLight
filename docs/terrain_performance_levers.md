# Maintained-renderer terrain performance levers

All changes are opt-in. This work does not resume selected-diffuse/voxel-sharing
experiments or change the maintained renderer's defaults.

## Implemented

- `RendererConfig.wavefront_environment_early_reject=True`: the native custom
  material/geometry primary sampler evaluates environment radiance before its
  visibility ray. Exactly black samples return zero immediately, preserving
  random draws; nonzero samples reuse the evaluated radiance. This is authored
  in OrdinaryShade and exposed by vxl8r `--reject-dark-environment`. It does not
  implement a brightness/distance threshold or replace secondary traversal.
  Environment texture lookup moves earlier, so other lighting workloads must
  be measured before enabling it generally.
- `RendererConfig.denoiser_fused_resolve=True`: native sampled-indirect rendering
  resolves raw HDR in the preparation kernel instead of dispatching a separate
  HDR resolve. OrdinaryShade entry point: `denoising/prepare_hdr.py`. The public
  `VulkanRelaxPrepare(..., hdr_output=...)` exposes the same implementation.
  Native scheduling falls back to separate resolve when denoising is disabled,
  legacy signals are selected, or indirect reservoir seeding is enabled.
- vxl8r `--fused-resolve` selects that upstream implementation through public
  configuration. No private renderer overrides or handwritten shaders are used.
- vxl8r `--history-policy local`, `--spatial-iterations 1..5`, and
  `--tile-capacity {131072,262144,524288}` expose existing public controls.
  `V` / `--direct-gi` retain the existing face-averaging comparison.
- The profiler supports the emitter/uneven-floor scene, one-at-a-time candidate
  settings, transport work counters, and separate image capture. Smaller path
  batches test transport scheduling; no new ray traversal algorithm was adopted.

The combined pass preserves per-sample float16 accumulation and writes the same
raw HDR and guides before denoising. Binding 14 is a distinct matching rgba16f
image. Sample zero overwrites addressed pixels; later samples accumulate.
Custom previous-position/identity records remain supported. Standalone users
must select sampled_indirect=True and must not omit any separate reservoir
seeding they require. Resources follow the existing preparation lifetime and
submission contract. Native activation is a constructor configuration option;
changing it requires renderer recreation. No new frame allocations or readbacks
are introduced. Signal preparation remains separately available.

## Animated 4K measurements

512×512 solid stepped floor, moving emissive controlled object, black environment,
perspective close-up, 3840×2160 at 1:1, four bounces, one sample, identity exports.
Each test alternates baseline/candidate render order for 40 measured frames after
12 warmup frames. Baseline: face averaging, three filtering passes, reset history
on content changes, 524288 paths, separate resolve. Times are GPU total medians;
CPU, desktop presentation and instrumented image capture are not included.

| Candidate | Matched baseline | Candidate | Saving |
| --- | ---: | ---: | ---: |
| Reject black environment samples | 22.473 ms | 21.579 ms | 0.894 ms |
| Combined resolve/preparation | 22.320 ms | 22.111 ms | 0.210 ms |
| Combined resolve, repeat | 22.408 ms | 22.297 ms | 0.111 ms |
| Face averaging off | 22.169 ms | 20.043 ms | 2.126 ms |
| Two filtering passes | 21.912 ms | 21.555 ms | 0.357 ms |
| Local history | 22.357 ms | 23.101 ms | -0.744 ms |
| 262144-path batches | 21.740 ms | 23.887 ms | -2.148 ms |

A separate counter run confirmed the early environment rejection removes
8,294,400 primary shadow rays per frame (16,588,800 down to 8,294,400); path
rays remain 19,731,100. Total shadow rays fell from about 17.62 million to
9.32 million. Counter instrumentation is excluded from timing conclusions.

The fusion saving is small (roughly 0.5–0.9%), not a major speedup. Local history
had 52 valid / 1 reset frames versus 0 valid / 53 reset for the baseline, so its
additional temporal cost is expected; reuse alone does not reduce transport.
Do not sum isolated savings. The old flat-floor 17.6 ms measurement is a different
scene and lighting workload. Emitter face enumeration is stochastic during timing;
these timings alone do not establish image parity.

## Quality and validation

- Early environment rejection: exact animated black-environment HDR/guide/hit
  parity with local history and explicit resets. Generated lighting source
  validates; all 52 default primary SPIR-V artifacts recompile byte-identically.
  A reflective scene with nonzero environment lighting also matched HDR and
  guides exactly through history/reset changes.
  The opt-in shader and option type checks pass (13 dedicated CPU tests,
  including combined-resolve policies).
- Animated emitter/terrain 161×97 fixed-enumeration check: exact raw/upstream HDR,
  all five exported signal/guide images and sampled primary hits, including local
  history and explicit resets every three frames.
- Native triangle/refraction tests: separate versus fused, two samples, valid
  history, resize and real indirect reuse fallback pass (2 GPU tests).
- CPU policy, compilation and existing resolve/preparation tests: 15 passed,
  6 GPU-gated tests skipped. Viewer regression tests: 10 passed, 7 opt-in skipped.
- Separate 960×540 captures use deterministic emitter enumeration. Raw lighting
  matched exactly for all three output/history comparisons; direct output also
  retained exact upstream denoised lighting.
- Averaging off shows substantially more per-pixel noise. Two filtering passes
  look close to three after face averaging in the saved frames, but differ
  numerically. Local history smooths the result and changes illumination; the
  screenshots cannot rule out lag or stale shadows during longer motion.

[Interactive image comparison](../artifacts/terrain-levers/compare.html) includes
three animation positions for each quality option. Captures are not timing runs.
Logs and images: `artifacts/terrain-levers/`.

## Reproduction

From Ordinary, set `PYTHONPATH=../vxl8r/src`, then run:

```sh
.venv/bin/python components/OrdinaryLight/scripts/profile_primary_prefixes.py \
  --stage full-control --emitter-object --uneven-floor \
  --primary-hit-format identity --width 3840 --height 2160 --warmup 12 --frames 40 \
  --fused-resolve
```

Replace the last option with `--reject-dark-environment`, `--candidate-direct`, `--candidate-iterations 2`,
`--candidate-history local`, or `--candidate-tile-capacity 262144`.
`--transport-counters` enables diagnostic counters and must not be treated as
uninstrumented performance. `--stable-emitter-fixture --capture-dir DIRECTORY`
enables reproducible comparison captures, with extra readbacks outside timing.
