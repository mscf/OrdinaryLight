# Paired diffuse/specular spatial filtering

Status: promoted with user approval. VulkanRelaxSpatial selects the packaged
paired OrdinaryShade kernel when the device supports seven storage images per
stage/set; otherwise it retains the separate-lobe implementation.

## Change

Each existing spatial iteration dispatches diffuse and specular filters
separately. Their geometry-dependent work is identical: neighbor coordinates,
material/depth rejection, normals, and normal/depth weights. The paired typed
OrdinaryShade prototype shares that work but retains independent radiance,
luminance/color weights, accumulation, and output alpha for each lobe. The
first-iteration specular firefly clamp remains specular-only.

Three iterations become three paired filtering dispatches instead of six
single-lobe dispatches, followed by the unchanged HDR composition pass. The
same four scratch images, filtered outputs, iteration count, image precision,
and graph lifetime rules are retained. Explicit hazards include both output
images. There are no per-frame allocations, CPU waits, or readbacks between
filtering stages; diagnostic readbacks happen only for measurement/parity.

The maintained paired shader is `relax_atrous_paired` in the denoising kernels
module, generated to SPIR-V and WGSL by `generate_denoiser_artifacts.py`. The
reusable spatial component preserves filter-only/compose-only operations, active
extents, standalone usage, and the separate-lobe fallback. The diagnostic now
forces that fallback only for its baseline; the candidate uses production code.

## Native 4K results

RTX 5090 Laptop GPU; perspective close-up, animated 512×512 voxel scene,
GPU animation/layout, tight bounds, four bounces, one sample per pixel,
524288 paths, compact primary-hit exports. Alternating baseline/candidate
order at matching animation times; 40 measured frames after 12 warmup frames.
Headless serialized GPU diagnostics exclude presentation.

Per-pass instrumentation measured each separate diffuse/specular iteration
pair at roughly 0.77–0.79 ms versus 0.55–0.57 ms for the paired dispatch.
Composition measured 0.158 versus 0.167 ms. Total GPU was 19.737 → 18.917 ms.
Because timestamps serialize boundaries and the candidate has fewer passes,
a confirmation run omitted the per-spatial-pass timestamps:

| GPU work | Separate lobes | Paired lobes |
| --- | ---: | ---: |
| Spatial filtering + composition | 2.481 ms | 1.840 ms |
| Total frame | 19.664 ms | 19.075 ms |

This confirmation saves 0.642 ms (25.9%) in spatial work and 0.589 ms (3.0%)
in total GPU time. Other stage variation contributes to the total difference;
do not add stage medians or assume the same gain for every scene.

Both 4K comparisons produced identical raw/denoised HDR and compact hit
identity/validity. Face-averaged output differed by at most 1.49e-7.
The animated default uses reset-on-content history; valid history is verified
separately rather than inferred from perspective projection.

## Replay

From the Ordinary checkout:

```sh
PYTHONPATH=../vxl8r/src .venv/bin/python components/OrdinaryLight/scripts/profile_primary_prefixes.py --stage full-paired-spatial --split-spatial
PYTHONPATH=../vxl8r/src .venv/bin/python components/OrdinaryLight/scripts/profile_primary_prefixes.py --stage full-paired-spatial --split-denoiser
```

`--split-spatial` logs every filter/compose pass; the remaining `relax_temporal`
marker then measures only trailing work, not the spatial total. With just
`--split-denoiser`, `temporal_median_ms` is spatial+composition time, and
`temporal_filter_median_ms` is temporal filtering. Logs preserve this distinction.

Diagnostic sources: `scripts/paired_spatial_experiment.py` and
`scripts/profile_primary_prefixes.py`. Logs: `artifacts/paired-spatial/`.

## Additional validation

Five headless cases passed at odd 321×181 extents: one spatial iteration,
five iterations with local history and periodic resets, and reflective,
refractive, and emissive fixtures at three iterations. Each compared raw,
denoised, and face-output HDR after every one of 11 animated frames, plus final
compact identity/validity. Final HDR outputs matched exactly in these cases.
The local-history case explicitly observed valid and reset policies. Emissive
parity used the deterministic diagnostic emitter sampler to avoid atomic list
ordering changing stochastic samples. These fixtures do not add optical
materials to vxl8r's public voxel format.

Production validation is recorded below. This work is local; no commit, push,
or release is performed as part of promotion.

## Production verification

Paired SPIR-V and WGSL artifacts were generated and validated with the installed
wgpu validator (naga-cli is unavailable); generation equality checks passed.
The shader is registered in both ownership and authorship inventories.

All 32 source/manifest/signal tests passed. All 23 headless GPU regression tests
passed, including five new paired-versus-fallback cases covering every iteration
count, nonuniform independent lobes, firefly inputs, material boundaries,
background preservation, alpha metadata, reusable recording, active-extent
shrink/restore, and separate filter/compose calls. Both filtered lobes and final
HDR matched the fallback exactly across the new cases. Existing lifetime,
preparation, temporal, and native-pipeline tests also passed.

A fresh production 4K alternating comparison (40 frames after 12 warmups) measured
spatial filtering/composition 2.486 → 1.841 ms and total GPU 19.868 → 19.233 ms,
a 0.634 ms / 3.2% total reduction. Raw and denoised HDR and hit records matched
exactly; face output differed by at most 1.19e-7. Presentation is excluded.

Local vxl8r uses the change on restart with its existing settings. No renderer
adapter patch or new command-line flag is required.

The production per-frame replay matrix also passed all five cases (one iteration,
five iterations with local history/resets, reflective, refractive, and deterministic
emissive) at odd 321×181 extents. Logs are `production-*.jsonl` in the artifact
folder. Each case compares all three HDR outputs after every measured/warmup
frame, with final compact-hit parity.
