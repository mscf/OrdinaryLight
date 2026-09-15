# Primary continuation diagnostics

Status: diagnostic only. Retain production queue emission; the apparent first-run
improvement did not repeat.

## Queue record stores

`full-primary-queue` replaces individual continuation fields with one complete
48-byte `WaveRay` assignment in a temporary typed OrdinaryShade module. It keeps
the capacity check, overflow count, path deactivation, cone values and padding.
Subgroup queue reservation is already enabled in this workload; this experiment
does not replace a per-ray atomic reservation with subgroup batching.

Animated perspective close-up, native 3840×2160, RTX 5090 Laptop GPU,
512×512 voxel scene, four bounces, one sample, 524288 path capacity, compact
primary identity outputs and production paired spatial filtering. Each comparison
alternates renderer order at identical animation times, with 12 warmups and 40
measured frames. These are headless GPU timings, excluding presentation.

| Run | Existing primary | Candidate primary | Existing total | Candidate total |
| --- | ---: | ---: | ---: | ---: |
| Initial | 4.932 ms | 4.809 ms | 19.237 ms | 19.034 ms |
| Repeat | 4.851 ms | 4.883 ms | 19.009 ms | 19.033 ms |

The initial 1.05% total improvement reversed to a 0.13% regression. This is not
evidence of a repeatable rendering gain. Final raw and denoised HDR match exactly
in both runs; face-averaged output differs by at most 1.20e-7. Compact hit identity
and validity match exactly. No separate camera-ray or full raw queue-record
comparison was performed, and optical/material variant coverage is not claimed.

## Continuation workload bound

A separate `lighting` prefix comparison measures 4.858 ms for full primary and
3.931 ms for a shader stopped after direct lighting, before opaque continuation
sampling and capture. Both sides disable face averaging to prevent synthetic or
truncated outputs from changing its cost. Earlier optical/custom scattering
branches remain. Final compact primary identity/validity match.

The 0.927 ms difference is a changed cumulative shader workload, not an isolated
BSDF-sampling measurement or an additive stage timing. The truncated shader's
radiance and total GPU time are not valid rendering results.

Source inspection finds repeated half-vector, Fresnel and GGX calculations across
`samplePbr`, `ordinarylight_pbr_pdf`, and `ordinarylight_pbr_evaluate`. A future
shared-evaluation experiment must preserve the sampling distribution, random
draw sequence, PDF clamps and diffuse/specular signals. Source duplication alone
does not prove duplicate GPU work: compiler optimization may already share it.

## Replay

```sh
PYTHONPATH=../vxl8r/src .venv/bin/python components/OrdinaryLight/scripts/profile_primary_prefixes.py --stage full-primary-queue --warmup 12 --frames 40
PYTHONPATH=../vxl8r/src .venv/bin/python components/OrdinaryLight/scripts/profile_primary_prefixes.py --stage lighting --primary-hit-format identity --warmup 12 --frames 40
```

Logs are in `artifacts/primary-continuation/`. No production shaders or viewer
defaults were changed by these experiments.
