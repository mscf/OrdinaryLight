# Batched face worklist reservation

Status: diagnostic only; not promoted. Production direct anchors and 32 stripes
remain unchanged.

Profiling current production at animated 4K gives approximately 1.598 ms for
linking, 0.523 ms reduction, 0.164 ms clearing, 0.160 ms finish and 0.160 ms resolve.
Individual pass instrumentation disables averaging command caching and can
alter GPU overlap. Cached full-operation comparisons below are separate runs.

The typed OrdinaryShade prototype replaces the three per-insertion worklist
counter increments (direct anchors, hashed anchors and stripes) with subgroup
reservation. Active insertion winners ballot their lanes; one elected lane adds
the lane count, broadcasts the base, and each lane uses an exclusive ballot
count as its offset. Hash insertion, pixel linking and reduction remain unchanged.
This uses subgroup ballot capabilities and would require device-support handling
before production promotion.

All **19 existing GPU averaging tests pass**, including direct/fallback allocation
boundaries, high identities, output scaling, HDR, both hit formats and cached
reuse. Final raw/denoised HDR and compact hits match exactly in the 4K comparisons;
face output remains within the existing tolerance.

RTX 5090 Laptop GPU, native 3840×2160 perspective, GPU animation/layout, four
bounces, one sample, 524288 paths, compact hits and paired spatial denoising.
Twelve warmups, forty measured frames, alternating renderer order at matching
animation times. Headless GPU measurements exclude CPU and presentation.

| View | Production averaging | Candidate averaging | Production total | Candidate total |
| --- | ---: | ---: | ---: | ---: |
| Close-up | 2.660 ms | 2.586 ms | 18.640 ms | 18.617 ms |
| Overview | 2.659 ms | 2.665 ms | 12.692 ms | 12.686 ms |

The close-up saves only 0.023 ms total; the overview is effectively unchanged.
These results do not justify adding a subgroup-dependent variant to production.
They do not establish that the shared counter is free, only that this batching
strategy provides no meaningful end-to-end gain in these workloads. Other
linking work and memory access remain candidates for further investigation.

## Replay

```sh
VXL8R_TEST_VULKAN=1 PYTHONPATH=../vxl8r/src .venv/bin/python components/OrdinaryLight/scripts/validate_face_averaging_experiment.py --variant worklist
PYTHONPATH=../vxl8r/src .venv/bin/python components/OrdinaryLight/scripts/profile_primary_prefixes.py --stage full-face-worklist
PYTHONPATH=../vxl8r/src .venv/bin/python components/OrdinaryLight/scripts/profile_primary_prefixes.py --stage full-face-worklist --overview
```

Logs: `artifacts/face-worklist/`. Python compilation and whitespace checks pass.
