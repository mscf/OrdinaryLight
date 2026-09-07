# Folded-surface sequence validation

The factor-1 geometric allowance was tested on two nearby, nonconvex height
fields z=0.4*sin(2*x)+0.2*sin(2*y)+offset. The foreground patch ends at x=0;
the rear patch spans the view, with offset 0.08. Both share a mesh/material.
Each patch uses either a 16x16 or 64x64 grid. Resolutions are 160x80 and
320x160. Each sequence contains 12 frames: initial hold, camera translation,
reversal, return and settling. Geometry is static.

The GPU temporal shader runs adaptive depth with constant radiance to isolate
geometric rejection. A CPU geometric gate evaluates:
previous_depth*1e-5 + normal_difference*max(pixel_footprint, tangent_separation).
It is applied to observed GPU acceptance, not fed back into temporal history.
No claim about resulting radiance quality, ghost trails or GPU timing follows.

## Counts across eleven frame transitions

| Width | Grid | Accepted valid matches | Additional valid rejections | Wrong matches before gate | Wrong matches after gate |
| --- | ---: | ---: | ---: | ---: | ---: |
| 160 | 16 | 93,182 | 0 | 99 | 0 |
| 160 | 64 | 95,719 | 0 | 103 | 0 |
| 320 | 16 | 379,930 | 0 | 129 | 0 |
| 320 | 64 | 385,180 | 0 | 137 | 0 |

All 468 observed false accepts are rejected, with no additional rejection among
954,011 accepted valid matches. These are correlated pixel/frame observations,
not independent statistical trials. Patch labels identify cross-patch
mismatches only; same-patch matches are treated as valid. Height-field geometry
does not exercise every kind of self-occlusion, overhang, moving/deforming
geometry, or optical transform. Mesh tessellation also changes the approximation
of the analytic surface, so this is robustness coverage rather than identical
geometry at both grid sizes.

## Outcome

This is stronger evidence for the candidate than the earlier sparse
close-curved test, and supports implementing an experimental shader replay
gate. Before native integration, verify GPU position precision, actual history
feedback, changing radiance, and moving/reflected geometry. The production
renderer and viewer remain unchanged.

All four sequences (48 GPU frames) completed. The existing 38 regression tests
passed; lint and diff checks passed. Per-frame counts are retained here.

Reproduce:
```bash
.venv/bin/python -m tools.denoiser_motion.folded_sequence --output /tmp/folded-sequence
```
