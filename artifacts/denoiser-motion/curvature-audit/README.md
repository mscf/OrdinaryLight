# Curvature allowance and faceted surfaces

The preceding close-curved experiment used half the normal-change allowance:
depth*1e-5 + 0.5*normal_difference*max(pixel_footprint, tangent_separation).

All additional valid rejections in that experiment occur at changed normals.
At unit scale, width 160, the 256 rejected accepted-valid matches have normal
changes; none has an effectively identical normal. The diagnostic geometry
is tessellated, so the smooth-curvature approximation is questionable. A
smooth arc distributes its normal change along the displacement, motivating
an average factor near one-half. Across a facet boundary, much of the
displacement may occur after the normal change. The endpoint normal change
does not provide a universal curvature bound for arbitrary surfaces.

We compared coefficients 0.5 and 1.0 on the same saved captures. This is a
CPU geometric gate applied to GPU temporal acceptance, not a shader change.

| Scale | Width | Extra valid rejections, 0.5 | Extra valid rejections, 1.0 | Wrong matches accepted before / after 1.0 |
| --- | ---: | ---: | ---: | ---: |
| 0.1 | 64 | 53 | 0 | 0 / 0 |
| 0.1 | 160 | 255 | 0 | 2 / 0 |
| 0.1 | 320 | 519 | 0 | 0 / 0 |
| 1 | 64 | 52 | 0 | 0 / 0 |
| 1 | 160 | 256 | 0 | 2 / 0 |
| 1 | 320 | 519 | 0 | 0 / 0 |
| 10 | 64 | 53 | 0 | 0 / 0 |
| 10 | 160 | 255 | 0 | 2 / 0 |
| 10 | 320 | 519 | 0 | 0 / 0 |

The factor-1 candidate retains all history already accepted by adaptive depth
while removing the observed false matches. It does not recover matches
previously rejected by the temporal shader. Only two accepted wrong matches
per scale constrain false acceptance; the scaled copies are not independent
scene diversity. This supports the faceting explanation, not a general
guarantee or production recommendation.

Next: validate this candidate on nonconvex/folded surfaces, different
tessellations, and multi-frame motion. Only then consider a shader experiment
with the required geometric inputs. No native renderer or viewer change.

Nine GPU replay cases completed. The existing 38 regression tests and lint
checks passed. Full report counts are retained beside this file.

Reproduce:
```bash
.venv/bin/python -m tools.denoiser_motion.analyze_curvature /tmp/close-curved-1 --output /tmp/curvature-unit.json
```
Repeat for the small and large saved captures.
