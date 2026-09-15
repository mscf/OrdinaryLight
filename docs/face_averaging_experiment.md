# Face averaging profiling and coherent anchor lookup

Status: the 32-stripe cap is promoted in vxl8r. Coherent anchor sharing remains
diagnostic only. No upstream production renderer changes were required.

## Per-pass profile

`profile_primary_prefixes.py --split-face` exposes the six sparse face-averaging
passes to the diagnostic graph timer. It bypasses the averaging command cache
for both renderers, and timestamps each pass at bottom-of-pipe. These boundaries
can change GPU overlap; CPU recording costs do not represent the cached viewer.
No extra waits or readbacks are inserted between rendering passes. The existing
diagnostic timer waits for each complete frame before collecting measurements.

Two identical full renderers, animated native 3840×2160 perspective close-up,
RTX 5090 Laptop GPU, 512×512 voxel scene, four bounces, 524288 paths, compact hits,
production paired spatial denoising: twelve warmups, forty measured frames,
alternating order at matching animation times.

| Pass | Median GPU time across the two controls |
| --- | ---: |
| Clear tables | 0.303 ms |
| Link pixels to faces | 1.750–1.772 ms |
| Prepare indirect dispatch | 0.003 ms |
| Reduce stripe lists | 0.527–0.535 ms |
| Finish face means | 0.271–0.273 ms |
| Resolve output | 0.161 ms |

Summed per-frame pass durations have medians of 3.013 and 3.054 ms. This sum
excludes gaps between passes and is not identical to a timestamp interval around
the entire operation. Final raw/denoised HDR and compact hits match exactly;
face output differs by at most 8.95e-8.

## Coherent anchor experiment

`face_anchor_experiment.py` loads a temporary typed OrdinaryShade variant of the
vxl8r sparse averaging module. If all active subgroup lanes have the same face
key, one elected lane inserts/looks up the anchor and broadcasts the result.
Mixed-face subgroups retain individual lookups. Stripe insertion and pixel
linking remain unchanged. Capabilities are declared on the compute entry point.
The prototype assumes subgroup ballot support on the test GPU; a production
implementation would require a supported-device fallback.

The 321×181 smoke test checks raw, denoised and face-output HDR on each animated
frame (three warmups, eight measured frames). It passes, with exact final HDR
and compact hit outputs. The 4K performance runs retain normal cached averaging
commands and do not use `--split-face`.

| Run | Existing face output | Candidate face output | Existing total | Candidate total |
| --- | ---: | ---: | ---: | ---: |
| Initial | 3.015 ms | 2.864 ms | 18.908 ms | 18.948 ms |
| Repeat | 3.031 ms | 2.841 ms | 18.990 ms | 18.785 ms |

The local face-averaging improvement repeats (0.151–0.190 ms, 5–6%). Total GPU
time ranges from a 0.040 ms regression to a 0.205 ms improvement; a stable total
speedup is not established. Final raw/denoised HDR and compact identity/validity
match exactly in both runs. Face-output error is at most 1.20e-7.

Retain this as a candidate for wider testing rather than changing viewer
defaults. Follow-up cases should include mixed and coherent faces, mostly-miss
views, full and compact hits, and sparse/direct face-index domains. The linking
pass still performs per-pixel stripe lookups and list insertion, providing a
larger remaining target than the shared anchor alone.

## Follow-up validation and stripe count

The coherent-anchor candidate passes all 17 existing vxl8r GPU averaging tests.
These cover mixed/coherent faces, full/compact identities, sparse/direct indexing,
HDR formats, all-miss updates, output scaling, and cached resource reuse. Dense
averaging cases remain unchanged controls; only sparse cases use the candidate.

A separate `full-face-stripes` experiment caps stripes at 32 instead of the
production maximum of 128. This changes the existing push constant only, after
target construction and before its first operation. Allocation sizes, shaders,
and transport remain unchanged. Fewer stripe keys reduce lookup/finish work,
while longer linked lists increase reduction work. It does not combine the
coherent-anchor candidate with the new stripe count.

| Animated 4K close-up | Existing averaging | 32 stripes | Existing total | 32-stripe total |
| --- | ---: | ---: | ---: | ---: |
| Initial | 3.035 ms | 2.669 ms | 19.146 ms | 18.677 ms |
| Repeat | 3.020 ms | 2.705 ms | 18.991 ms | 18.777 ms |

Averaging saves 0.315–0.367 ms. Total GPU savings range from 0.215–0.469 ms
(1.1–2.4%); unmodified stage timing variation contributes to this range.
Raw/denoised HDR and compact hits match exactly. Face-output differences remain
within the diagnostic tolerance. The 32-stripe variant also passes all 17
existing GPU averaging tests. These are diagnostics, not a viewer default change.

The animated 4K perspective overview shows a smaller improvement: averaging
3.111 → 3.062 ms, total 13.155 → 13.081 ms. Final raw, denoised and face-output
HDR and compact hits match exactly. This supports the expected dependence on
pixels per face, rather than a uniform speedup across views. Orthographic and
other stripe counts have not been measured in this follow-up.

Validation replay from the Ordinary root:

```sh
VXL8R_TEST_VULKAN=1 PYTHONPATH=../vxl8r/src .venv/bin/python components/OrdinaryLight/scripts/validate_face_averaging_experiment.py --variant anchor
VXL8R_TEST_VULKAN=1 PYTHONPATH=../vxl8r/src .venv/bin/python components/OrdinaryLight/scripts/validate_face_averaging_experiment.py --variant stripes32
PYTHONPATH=../vxl8r/src .venv/bin/python components/OrdinaryLight/scripts/profile_primary_prefixes.py --stage full-face-stripes --face-stripes 32
```

## Combined candidate and production decision

The combined anchor-sharing/32-stripe variant passes all 17 averaging tests.
At animated perspective native 4K, comparing it against the old 128-stripe
implementation gives averaging 3.028 → 2.613 ms, total 19.067 → 18.491 ms.
However, a direct comparison against **32 stripes alone** gives averaging
2.696 → 2.640 ms and total 18.734 → 18.672 ms. The incremental saving from
subgroup sharing is only about 0.06 ms. Raw/denoised HDR and compact hits match
exactly, and face output remains within tolerance.

Promote the simpler 32-stripe cap in `vxl8r_render/backends/sparse_average.py`.
It adds no subgroup capability requirement or shader variant. Retain the combined
version as diagnostic only. The unpatched production averaging suite passes
**17 tests** after the change; the viewer needs only a restart, with no new flag.
This changes partitioning and floating-point reduction order, not the contributing
samples or lighting estimator. The profiler reconstructs the old stripe count
explicitly for comparisons, independently of the current production default.

Post-change animated orthographic 4K comparison against reconstructed 128 stripes:
averaging 2.286 → 1.987 ms; total GPU 18.835 → 18.734 ms. This also passes final
HDR and compact-hit parity checks. The smaller total improvement reinforces that
per-stage savings should not be reported as an equivalent presentation gain.

```sh
PYTHONPATH=../vxl8r/src .venv/bin/python components/OrdinaryLight/scripts/profile_primary_prefixes.py --stage full-face-combined --face-baseline-stripes 32
VXL8R_TEST_VULKAN=1 PYTHONPATH=../vxl8r/src .venv/bin/python components/OrdinaryLight/scripts/validate_face_averaging_experiment.py --variant combined
```

## Earlier profiling replay

```sh
PYTHONPATH=../vxl8r/src .venv/bin/python components/OrdinaryLight/scripts/profile_primary_prefixes.py --stage full-control --primary-hit-format identity --split-face
PYTHONPATH=../vxl8r/src .venv/bin/python components/OrdinaryLight/scripts/profile_primary_prefixes.py --stage full-face-anchor --width 321 --height 181 --warmup 3 --frames 8 --check-each-frame
PYTHONPATH=../vxl8r/src .venv/bin/python components/OrdinaryLight/scripts/profile_primary_prefixes.py --stage full-face-anchor
```

Logs: `artifacts/face-averaging/`. These headless GPU measurements exclude CPU
and presentation time. No GUI performance or broad geometry/material validation
is claimed.
