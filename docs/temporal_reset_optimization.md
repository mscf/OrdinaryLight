# Temporal reset fast path

The OrdinaryShade temporal kernel now returns immediately after reading current
radiance when the explicit history-valid bit is false. It writes the unchanged
radiance and history length one, without fetching geometry, motion or history.
The previous implementation produced the same result after unnecessary guide
reads. The transmission cap cannot change a history length of one. Valid-history
filtering, guide generation, diagnostics, layouts, APIs and history policy remain
unchanged. Packaged SPIR-V/WGSL and their manifest were regenerated and validated
using the available wgpu validator (naga is not installed).

## Native 4K comparison

NVIDIA GeForce RTX 5090 Laptop GPU. vxl8r animated 512×512 scene, four bounces,
1 spp, GPU animation/layout, tight dynamic bounds, 524,288 paths per batch,
occupied-stripe averaging and the preceding sampled-resolve optimization.
GI and output both 3840×2160. Two renderers alternate order for 48 warmed GPU
samples each. These are serialized diagnostics, not displayed FPS.

| Median GPU time | Baseline | Fast path |
| --- | ---: | ---: |
| Close-up temporal marker | 3.041 ms | 2.818 ms |
| Close-up total | 23.216 ms | 22.881 ms |
| Overview temporal marker | 2.308 ms | 2.300 ms |
| Overview total | 16.324 ms | 16.339 ms |

This is a modest, workload-dependent improvement: approximately 0.22 ms in the
close-up temporal marker and effectively no overview improvement. Do not
extrapolate it to a universal FPS gain. Close-up total p95 was 24.640 → 24.193 ms;
overview p95 varied 17.138 → 17.416 ms. Native temporal markers describe the
previous use of a frame slot.

Raw and denoised HDR were identical in both captures. Averaged close-up output
changed by at most 1.4901e-7 due to floating-point averaging order; overview output
was identical. After eight stationary overview frames, raw, denoised and averaged
HDR also matched exactly. Subsequent inspection found that orthographic cameras
are excluded by the native denoiser-history validity gate, so this stationary
check did not exercise valid temporal history. See the paired-temporal experiment.

Six GPU tests passed across test_relax_temporal.py and test_relax_prepare.py,
covering resets, history reuse, failed submission and guide preparation.

## Discarded workgroup experiments

Signal preparation at 64 versus 128 threads measured 2.472 versus 2.475 ms;
64 versus 256 threads measured 2.503 versus 2.500 ms. No useful gain, so its
64-thread workgroup remains unchanged. All variants were compiled from the
same typed OrdinaryShade source; no handwritten shader variants were introduced.

Results, baseline temporal binary and reproducible comparison are saved under
`artifacts/temporal-reset/`. From vxl8r:

```bash
PYTHONPATH=src ../Ordinary/.venv/bin/python ../Ordinary/components/OrdinaryLight/artifacts/temporal-reset/compare.py
PYTHONPATH=src ../Ordinary/.venv/bin/python ../Ordinary/components/OrdinaryLight/artifacts/temporal-reset/compare.py --overview
```

Run sequentially without other GPU workloads. No native window was opened;
presentation, compositor and display scanout are excluded.
