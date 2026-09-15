# Secondary initialization ownership

Fused primary generation already clears each captured 128-byte secondary record
before intersection. Native denoiser rendering also issued a transfer fill of
those same records immediately before dispatch. The normal viewer captures all
primary pixels, making the transfer fill redundant.

`VulkanWavefrontExecutor._needs_secondary_transfer_clear` now omits that fill
only when denoiser signals are active, the execution strategy is `wavefront`,
primary generation is fused, and the capture stride is zero (all pixels) or one.
The shader clear remains. Split generation, other strategies, and sparse capture
strides retain their transfer initialization. Without denoiser signals the old
behavior already did not issue the fill.

The secondary-buffer barrier now uses shader read/write source access when the
transfer clear is absent. The buffer layout, allocated capacity, shader behavior,
material evaluation, transport, guides and history are unchanged. No additional
CPU work occurs between stages; this decision is part of reusable recording.

This is different from the earlier rejected experiment removing **shader**
clears while retaining the transfer clear. That experiment became slower.

## Refreshed baseline

RTX 5090 Laptop GPU, native 3840×2160, perspective close-up, 512×512 scene,
animation, GPU layout, tight bounds, compact primary-hit exports, four bounces,
one sample per pixel and 524,288 paths. The refreshed profile recorded:

| Work | GPU median |
| --- | ---: |
| Primary including setup | 5.723 ms |
| First secondary intersection/shading | 5.061 ms |
| Later secondary stages | 0.767 / 0.496 ms |
| Denoiser preparation | 2.469 ms |
| Temporal filtering | 2.823 ms |
| HDR resolve | 0.286 ms |
| Face averaging | 3.021 ms |
| Total GPU | 21.533 ms |

Its separate 120-frame headless throughput run measured 42.57 FPS, 23.49 ms per
frame, and 3.46 ms calling-thread CPU. CPU/GPU values overlap and are not additive.
These runs exclude swapchain/compositor presentation; they are not viewer FPS.

## Controlled comparison

The comparison alternates old/new renderer order for 40 samples after 12 warmup
frames, at identical animation times. Its primary setup timestamp separates
resets/transfer operations from shader execution. The old renderer retains the
previous always-fill behavior; both renderers use the current compact hit ABI.

| Close-up | Transfer plus shader clear | Shader clear only |
| --- | ---: | ---: |
| Primary setup | 0.773 ms | 0.024 ms |
| Primary graph/dispatch | 4.970 ms | 4.893 ms |
| Total GPU | 21.635 ms | 20.818 ms |

The total reduction was 0.817 ms, about 3.8%. This removes approximately 1 GiB
of repeated transfer writes per 4K one-sample frame, not an allocated buffer.
Raw and denoised HDR were identical; face averaging differed by at most 1.19e-7,
and compact identity/validity records were identical. It does not specifically
speed up the first secondary intersection shader, which remains an important
next target.

The wide overview comparison reduced total GPU time from 14.928 to 13.729 ms
(1.199 ms, about 8.0%). Setup fell from 0.736 to 0.021 ms; primary graph/dispatch
measured 4.428 versus 3.890 ms. All three HDR outputs and compact hit records
matched exactly. The extra primary-time change is an observed effect of removing
the transfer work; these timings do not establish a particular cache mechanism.

## Final production profile

A second full close-up profile with the change enabled measured 44.30 FPS /
22.58 ms per frame in the 120-frame headless throughput run, versus 42.57 FPS /
23.49 ms before. These throughput captures were sequential, not interleaved.
Calling-thread CPU averaged 3.34 ms. The separate GPU diagnostic measured
20.890 ms total, primary including setup 4.953 ms, first secondary 5.083 ms,
denoiser preparation 2.478 ms, temporal filtering 2.831 ms and face output
3.029 ms. Thus the improvement is concentrated in primary setup; the first
secondary stage remains the largest individual native stage.

## Validation and replay

Fourteen initialization tests passed: eight ownership-policy cases and six GPU
cases covering tiny repeatedly reused tiles, hit/miss transitions, resize,
one/two samples, one/four bounces, and all-pixel versus sparse capture. The GPU
cases compare the final HDR of each tested frame exactly against transfer
clearing. Eight existing native graph/history/readback and hit-format tests
also passed.

Reflective, refractive and emissive fixtures passed exact raw, denoised and
face-averaged HDR checks and exact identity/validity comparisons. Emissive parity
uses the diagnostic deterministic emitter sampler documented in
`compact_primary_hit_experiment.md`, not a production sampling change.

From the Ordinary checkout:

```sh
PYTHONPATH=../vxl8r/src .venv/bin/python components/OrdinaryLight/scripts/profile_primary_prefixes.py --stage full-secondary-clear
PYTHONPATH=../vxl8r/src .venv/bin/python components/OrdinaryLight/scripts/profile_primary_prefixes.py --stage full-secondary-clear --overview
PYTHONPATH=../vxl8r/src .venv/bin/python ../vxl8r/scripts/profile-4k-animation.py --native-4k --native-gpu-timings --projection perspective --tight-dynamic-bounds --frames 120
```

The vxl8r profiler now defaults to `--primary-hit-format identity`, matching the
viewer; `--primary-hit-format full` retains diagnostic exports. It logs the
selection. Comparison shader/clear selection and blocking readbacks are confined
to the diagnostic harness. Production rendering uses the tested ownership rule
automatically. Logs are archived in `artifacts/secondary-initialization/`.

## Remaining traffic

Secondary records are not equivalent to application hit exports. Their fields
feed diffuse/specular decomposition, first-secondary-hit distances, reprojection,
transmission flags, and optional reservoir reuse; raw capture also exposes them.
The 128-byte record must not be globally reduced merely because one consumer
uses fewer fields. A further experiment should measure the first secondary stage
and denoiser preparation separately, preserving the needs of each active consumer.
