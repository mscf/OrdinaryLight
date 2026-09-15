# Sampled-indirect HDR resolve optimization

The path-to-HDR shader previously loaded and wrote back the complete 128-byte
secondary record for each pixel on the last sample, even when sampled-indirect
preparation needed that record unchanged. It also classified lighting before
rejecting pixels that do not represent an indirect reservoir entry.

The OrdinaryShade implementation now rejects non-representative pixels after
writing HDR, before loading the secondary record, when sampled-indirect mode is
active. It stores secondary records only in legacy lobe-classification mode,
where those fields actually change. HDR accumulation, reservoir representatives,
seed/reservoir formats, legacy capture and sampled-indirect scratch are preserved.
There are no new resources, API/settings changes, shader ABI changes, or changes
to temporal history. The existing resource declarations remain conservative.

Source: `scripts/generate_core_shaders.py::wavefront_path_to_hdr`.
The packaged GLSL and SPIR-V were regenerated from this typed OrdinaryShade source.
The test image reader was also converted to OrdinaryShade.

## Animated native-4K validation

NVIDIA GeForce RTX 5090 Laptop GPU; vxl8r 512×512 scene, four bounces, 1 spp,
524,288-path capacity, GPU animation/layout, tight dynamic bounds, denoising,
occupied-stripe face averaging and overlay. Both GI and output: 3840×2160.

Baseline and candidate renderers alternate order on identical rotation frames;
48 warmed samples per variant. GPU timings are serialized diagnostic timestamps,
not displayed FPS. Both variants include the preceding vxl8r averaging improvement.

| Median GPU time | Baseline | Optimized |
| --- | ---: | ---: |
| Close-up HDR resolve | 2.958 ms | 0.289 ms |
| Close-up signal preparation | 2.863 ms | 2.523 ms |
| Close-up total | 26.421 ms | 23.405 ms |
| Overview HDR resolve | 3.121 ms | 0.488 ms |
| Overview signal preparation | 2.212 ms | 2.031 ms |
| Overview total | 19.059 ms | 16.287 ms |

Total GPU time fell 11.4% in the close-up and 14.5% in the overview. Preparation
code itself is unchanged; its smaller measured cost may reflect reduced memory
traffic/cache disruption from resolve, rather than an independent optimization.
Close-up total p95 was 27.498 → 24.700 ms; overview was 19.332 → 16.583 ms.

Raw and denoised HDR were bit-identical in both captures. Averaged overview HDR
also matched exactly. Averaged close-up HDR differed by at most 2.0862e-7,
consistent with nondeterministic floating-point summation in vxl8r averaging.

36 tests passed, 9 skipped across `test_path_resolve.py`, `test_relax_prepare.py`,
and `test_ordinaryshade_core.py`. GPU tests cover capture disabled, legacy and
sampled capture; one- and two-entry reservoir grids; valid/invalid secondary hits;
multi-sample accumulation; exact secondary scratch preservation; denoiser guides;
and generation checks. All changed shader code is OrdinaryShade.

Artifacts and baseline binary: `artifacts/sampled-resolve/`. Run from vxl8r with
its source on PYTHONPATH and the shared development environment:

```bash
PYTHONPATH=src ../Ordinary/.venv/bin/python ../Ordinary/components/OrdinaryLight/artifacts/sampled-resolve/compare.py
PYTHONPATH=src ../Ordinary/.venv/bin/python ../Ordinary/components/OrdinaryLight/artifacts/sampled-resolve/compare.py --overview
```

Run sequentially, without other GPU applications. No native window was launched;
swapchain/compositor/VSync/scanout costs are excluded.

A separate unprofiled 180-frame animated close-up run measured 24.869 ms/frame
(40.21 headless FPS) and 4.24 ms/frame rendering-thread CPU. The earlier baseline
was 27.961 ms/frame (35.76 FPS). These throughput runs were separate rather than
interleaved; use the alternating GPU comparison above to attribute the improvement.
CPU and GPU overlap, so the CPU time is not additive to the frame interval.
