# Visibility-only query experiment

A typed OrdinaryShade boolean occlusion helper was implemented and tested in place
of full-hit queries used by environment, area and analytic light visibility.
It retained ray masks and intervals, distance/normal validation, invalid-candidate
rejection and equal-IOR optical-boundary pass-through. It returned upon an accepted
procedural blocker without retaining the selected surface payload. Full primary
and secondary surface queries were unchanged.

The experiment was **discarded**: its measured benefit did not justify a production
change. The original generated sources were restored and generator checks pass.
vxl8r's previously approved 524,288-path default remains in place.

## Validation

13 GPU tests passed: two direct boolean/full-hit query comparisons plus the 11
existing native-intersection tests. The direct comparisons covered 240 ray cases
with optical boundaries disabled and the same 240 with them enabled, including
masks, near/far clipping, misses, inside-box origins, rejected normals and
nonfinite distances followed by later blockers. Existing tests covered native
lighting, emitter and optical-boundary behavior. This was not exhaustive coverage
of all geometry and material combinations.

An interleaved animated vxl8r comparison used actual 3840×2160 GI and output,
524,288-path capacity, four bounces, one sample per pixel, denoising, face averaging,
GPU animation and GPU layout. Each variant used 48 warmed samples with alternating
execution order. Both generated-source variants were compiled before timing;
current source files were restored after baseline setup. Captured final raw and
denoised HDR matched exactly. Timings use separate serialized GPU timestamps and
are not displayed FPS.

| GPU interval | Full-hit visibility | Boolean visibility |
| --- | ---: | ---: |
| Whole graph median | 37.683 ms | 37.506 ms |
| Whole graph p95 | 39.990 ms | 40.394 ms |
| Native GI median | 34.832 ms | 34.591 ms |
| Primary median | 13.445 ms | 13.386 ms |
| First secondary median | 9.672 ms | 9.625 ms |

The 0.47% whole-graph median reduction accompanied a 1.01% p95 increase. This does
not establish a useful speedup or a frame-pacing improvement. Shader compilation
may already eliminate much unused payload bookkeeping; that is a plausible
explanation, not a verified compiler finding. The earlier environment-sampling
ablation changed sampling and subsequent paths, so its approximately 7 ms delta
was never a prediction for this query change.

The result argues against spending more effort on selected-hit bookkeeping for
this workload. Further investigation should target actual ray traversal and
lighting evaluation, while preserving sampling quality and avoiding intrusive
shared-atomic counters in timing runs.

Local reproducibility artifacts are retained under `artifacts/visibility-query/`:
`experiment.patch`, `gpu-test.py.txt`, and `benchmark.jsonl`. These are experiment
records, not active shader sources or a retained API.
