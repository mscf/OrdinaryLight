# Preparation / temporal fusion investigation

Status: rejected diagnostic prototype. Production OrdinaryLight shaders,
resources, and vxl8r defaults are unchanged.

## Dependency constraint and tested scope

Full preparation/temporal fusion is not a simple adjacent-pass merge. Temporal
history clamping reads a 3×3 neighborhood of prepared signals and sometimes
neighbor geometry/identity. Native rendering prepares signals per path tile,
then reuses the bounded path/secondary buffers for subsequent tiles. Threads
cannot read prepared neighbors across workgroups/tiles before those writes
finish. A workgroup barrier alone cannot establish that dependency. Retaining
all full-resolution path records, recomputing halo data, or changing staging
would require a larger design; this experiment does not test or rule out those
alternatives.

The tested subset is invalid-history frames, whose temporal output is exactly
the current signal with history length one. Preparation writes those outputs
directly, guarded by the existing GPU temporal-policy uniform. Valid-history
frames use the normal temporal calculation unchanged. Both temporal dispatches
remain, but return immediately on reset frames. Raw diagnostic signals remain
materialized, so this prototype avoids their temporal reread (16 bytes/pixel,
126.6 MiB at 3840×2160), not every intermediate write or buffer allocation.

All shader behavior comes from AST transformations of typed OrdinaryShade in
`reset_prepare_fusion_experiment.py`. Four extra output images and the existing
policy uniform are bound through persistent diagnostic kernels. Explicit
resource uses cover preparation's new writes and uniform read. Existing graph
barriers order subsequent temporal/spatial work. No between-stage CPU readback,
allocation, or wait is introduced. Compiler/runtime interception is confined to
the diagnostic harness; it is not a supported integration API.

## Measurements

RTX 5090 Laptop GPU, animated 512×512 voxel scene, native 3840×2160,
GPU animation/layout, tight bounds, four bounces, one sample, 524288 paths,
compact hit exports. Alternating baseline/candidate order, 40 measured frames
after 12 warmups. These are serialized headless GPU diagnostics, not presented
FPS. The close-up perspective split capture measured:

| GPU work | Baseline | Reset fusion |
| --- | ---: | ---: |
| Preparation | 2.489 ms | 2.738 ms |
| Temporal filtering only | 0.344 ms | 0.251 ms |
| Spatial filtering/composition | 2.493 ms | 2.523 ms |
| Total | 19.826 ms | 20.029 ms |

The temporal saving is outweighed by more expensive preparation. Earlier
unsplit runs also showed no benefit: orthographic total 19.753 → 19.812 ms;
perspective total 20.010 → 20.199 ms. Do not promote this candidate.

The old `relax_temporal` marker covers both temporal and spatial work. With
`--split-denoiser`, the new `temporal_filter` marker ends temporal filtering and
the existing `relax_temporal` remainder measures spatial filtering/composition.
In JSON, `temporal_filter_median_ms` is the former, and `temporal_median_ms` is
the latter when `split_denoiser` is true. Stage medians are not additive.

## History and parity

Instrumentation found zero valid-history frames in the default animated
perspective benchmark: all 53 observed policies were resets. vxl8r's default
`reset_on_content` policy invalidates history when object transforms change.
Perspective projection alone does not establish history reuse. Earlier claims
that a perspective smoke run exercised reset-to-valid transitions were therefore
incorrect; policy counters now verify this explicitly.

A separate 640×360 run using the public `history_policy='local'`, with explicit
resets every seven frames, observed 13 valid-history and three reset frames in
each renderer. Final raw HDR, denoised HDR, face output, and hit identity/validity
matched exactly. In all 4K runs raw and denoised HDR and hit identity/validity
also matched exactly; face-output differences were at most 2.09e-7. These are
final-frame comparisons, not exhaustive per-frame or material coverage. Since
the performance candidate is rejected, no production feature matrix is claimed.

## Replay

```sh
PYTHONPATH=../vxl8r/src .venv/bin/python components/OrdinaryLight/scripts/profile_primary_prefixes.py --stage full-reset-fusion --split-denoiser
PYTHONPATH=../vxl8r/src .venv/bin/python components/OrdinaryLight/scripts/profile_primary_prefixes.py --stage full-reset-fusion --projection orthographic
PYTHONPATH=../vxl8r/src .venv/bin/python components/OrdinaryLight/scripts/profile_primary_prefixes.py --stage full-reset-fusion --width 640 --height 360 --warmup 3 --frames 12 --history-policy local --reset-every 7 --split-denoiser
```

Logs: `artifacts/reset-prepare-fusion/`. The next focused denoiser target is the
roughly 2.5 ms spatial filtering/composition work, rather than the 0.34 ms reset
temporal pass. Changing the viewer's history policy is a separate image-quality
and history-rejection decision, not part of this experiment.
