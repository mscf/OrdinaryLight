# Eight- and sixteen-pixel grouping sweep

Status: diagnostic only. Retain the production four-pixel stripe grouping.

The temporary typed OrdinaryShade variant changes only the integer divisor in
stripe selection from four to eight or sixteen. Every valid pixel still
contributes once, with the same direct-anchor/hash fallback and 32 stripes.
Both candidates pass **19 GPU averaging tests**. Temporary fixtures adjust only
the expected number of stripe-zero pixels; expected image means stay unchanged.

## Animated native 4K

RTX 5090 Laptop GPU, 3840×2160, four bounces, one sample, 524288 paths,
GPU animation/layout, compact hits and paired spatial denoising. Twelve warmups,
forty measured frames with alternating renderer order and matched animation
times. Cached averaging, headless GPU measurements excluding CPU/presentation.
Each row is a separate comparison against four-pixel production grouping.

| Candidate / view | Production averaging | Candidate averaging | Production total | Candidate total |
| --- | ---: | ---: | ---: | ---: |
| 8 / perspective close-up | 2.331 ms | 2.213 ms | 18.393 ms | 18.289 ms |
| 16 / perspective close-up | 2.335 ms | 2.270 ms | 18.459 ms | 18.404 ms |
| 8 / perspective overview | 1.719 ms | 1.573 ms | 11.787 ms | 11.665 ms |
| 8 / orthographic close-up | 2.045 ms | 2.011 ms | 18.605 ms | 18.737 ms |

Eight performs better than sixteen in the close-up, but savings are small:
0.10–0.12 ms total in perspective views, while the orthographic total regresses
by 0.13 ms. Averaging itself improves slightly in that orthographic comparison,
so the total regression must not be attributed directly to grouping. Other
stages and between-stage gaps contribute to the measured total. These single
comparisons do not establish a stable general frame-time win; leave the default
unchanged rather than promote based solely on the smallest operation median.

Final raw and denoised HDR and compact identities match exactly in every run.
Face output remains within the established numerical tolerance. Larger grouping
changes occupied-stripe counts and reduction order, not the sample population.

## Replay

```sh
VXL8R_TEST_VULKAN=1 PYTHONPATH=../vxl8r/src .venv/bin/python components/OrdinaryLight/scripts/validate_face_averaging_experiment.py --variant group --group-size 8
VXL8R_TEST_VULKAN=1 PYTHONPATH=../vxl8r/src .venv/bin/python components/OrdinaryLight/scripts/validate_face_averaging_experiment.py --variant group --group-size 16
PYTHONPATH=../vxl8r/src .venv/bin/python components/OrdinaryLight/scripts/profile_primary_prefixes.py --stage full-face-group --face-group-size 8
PYTHONPATH=../vxl8r/src .venv/bin/python components/OrdinaryLight/scripts/profile_primary_prefixes.py --stage full-face-group --face-group-size 16
PYTHONPATH=../vxl8r/src .venv/bin/python components/OrdinaryLight/scripts/profile_primary_prefixes.py --stage full-face-group --face-group-size 8 --overview
PYTHONPATH=../vxl8r/src .venv/bin/python components/OrdinaryLight/scripts/profile_primary_prefixes.py --stage full-face-group --face-group-size 8 --projection orthographic
```

Logs: `artifacts/face-group-size/`. Python compilation and whitespace checks pass.
