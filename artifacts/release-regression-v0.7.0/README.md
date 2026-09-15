# Upstream release regression — 2026-09-15

Initial audit status: **release held**. Superseded by the user-approved baseline
acceptance and passing reruns in [RELEASE-FOLLOWUP.md](RELEASE-FOLLOWUP.md).
The original results below are preserved as historical evidence.
OrdinaryShade 0.1.0a8 and OrdinaryLight 0.7.0 have draft version metadata and
release notes. The requested condition was to publish only after regression
passes. Two formal quality gates remain red; no baseline or tolerance was changed.
The eventual OrdinaryLight release still needs its CI compiler SHA advanced to
the corresponding committed OrdinaryShade release.

## Results

- OrdinaryShade: **121 passed**; generated WGSL validation passed with Naga 30.0.0.
- OrdinaryLight full GPU-enabled pytest, including legacy quality/performance gates: **1125 passed, 2 failed, 13 skipped, 84 subtests passed** (1140 tests, 1134.88 seconds).
- All 13 skipped cases passed separately: 12 compiler checks were hidden by a lowercase sibling-checkout assumption; the browser contract required Node. The corrected compiler suite passed all 37 tests; portable contract suite passed all 15.
- Final hardware-independent OrdinaryLight suite: **907 passed, 233 GPU-opt-in skips, 80 subtests passed**.
- Corrected presentation performance matrix: **8/8 passed**, unchanged 50 FPS and minimum-pixel thresholds. The original matrix's performance subprocesses returned 127 because the launcher chose a nonexistent component-local venv.
- OrdinaryLattice: 278 passed. OrdinaryScience: 83 passed, 141 optional skips. Workspace release tooling: 28 passed.
- Shader authorship and every generated text source passed. All 459 core SPIR-V artifacts rebuilt identically; raster/denoiser artifact generators passed separately. All 472 packaged SPIR-V files validated.
- Both candidate source/wheel distributions passed metadata and wheel-content validation. A clean consumer installation of the final wheels passed. Distribution hashes are recorded separately; archives remain under /tmp/upstream-release-regression.

The corrected compiler-discovery checks, native callback probe declarations,
and launcher update were made after the full run had collected/imported tests;
supplemental runs validate those changes. The corrected performance matrix is
separate from the original failing aggregate gate, not a rewritten result.

## Remaining failures, compared with the published release

Control uses clean OrdinaryLight v0.6.0 (c9ed8f0) and OrdinaryShade v0.1.0a7
(ea5e0d7), on the same GPU, sequentially after the candidate. The full noise
configuration and the small-emitter matrix settings (8 bounces, roulette start
4, 320×180, 8 frames, 16 reference samples) are matched.

| Check | Candidate | Published release | Existing limit |
| --- | ---: | ---: | ---: |
| Area-light low-frequency noise metric | 2.109238 | 2.109238 | 2.071776 |
| Moving-volume median GPU time | 4.116592 ms | 4.185840 ms | 2.240717 ms |
| Small-emitter roulette/control low-frequency noise ratio | 1.243968 | 1.243968 | 1.15 |

All measured noise-quality metrics are exactly identical between candidate and
release across all six noise scenarios. Small-emitter termination quality
metrics are also exactly identical. These are inherited gate failures, not
observed new quality regressions. The timing observation is a sequential check,
not an alternating performance study; the historical timing baseline does not
record hardware/clock state. Resolving or explicitly accepting the inherited
gate failures is required before publishing under the user's condition.

## Fixes made during regression

1. Replace no-op GLSL string fault injections with six typed OrdinaryShade sampler callbacks. Generated return spelling had changed, silently leaving valid samplers in the old tests.
2. Update the obsolete whole-secondary-record store assertion to verify the retained field writes and preservation of the PDF word.
3. Include shared typed analytic-light definitions in standalone transport lighting. The standalone example previously failed shader linking. Add a hardware-independent compile regression; original GPU numerical expectations pass.
4. Discover the installed OrdinaryShade package and available glslangValidator rather than silently skipping checks based on local checkout/tool paths. Supply current native occlusion/triangle callbacks to compiler probes.
5. Forward the running Python interpreter through the 4K shell launcher.

No new handwritten shader algorithms or application-specific renderer patches
were introduced. No vxl8r files were changed by this release task.

## Presentation gate measurements

These are the upstream presentation gate scenes, not the vxl8r large-room
benchmark. The request was 3840×2160; the compositor produced **3773×2368** in
all runs, 7.7% more pixels than the gate's 4K target. Do not label these exact
3840×2160 measurements. Each run reports 90 measured frames after 30 warmup
frames. GPU medians and total presentation throughput are different metrics.

| Scene | Throughput FPS | Median GPU ms | Measured framebuffer |
| --- | ---: | ---: | --- |
| area_lights | 56.49 | 16.88 | [3773, 2368] |
| diffuse | 69.71 | 13.35 | [3773, 2368] |
| glossy_glass | 55.64 | 16.73 | [3773, 2368] |
| textured | 68.27 | 14.27 | [3773, 2368] |
| small_emitter | 70.57 | 13.49 | [3773, 2368] |
| occlusion | 69.85 | 13.52 | [3773, 2368] |
| nested_glass | 54.10 | 16.98 | [3773, 2368] |
| dense | 68.16 | 13.49 | [3773, 2368] |

## Reproduction

Use the workspace Python with both editable upstream packages, Vulkan/GLFW,
glslangValidator, Naga, Node, and the built optional FSR2 bridge. Logs record
the source-tree test results; the control worktrees and raw outputs remain in
`/tmp/upstream-release-regression`.

```sh
cd components/OrdinaryLight
ORDINARYLIGHT_TEST_VULKAN_GRAPH=1 \
ORDINARYLIGHT_TEST_VULKAN_TRANSPORT=1 \
ORDINARYLIGHT_TEST_VULKAN_RUNTIME=1 \
ORDINARYLIGHT_RUN_GPU_GATES=1 \
ORDINARYLIGHT_RUN_PERFORMANCE_GATES=1 \
../../.venv/bin/python -m pytest -vv -ra --tb=short tests

../../.venv/bin/python -m tests.gates.validation_matrix \
  --stages performance --performance-platform x11 \
  --performance-logical-width 3840 --performance-logical-height 2160 \
  --output /tmp/ordinarylight-performance-recheck
```
