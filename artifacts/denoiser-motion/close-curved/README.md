# Closely spaced curved surfaces and scene scale

Two tessellated paraboloid patches share a mesh and material. The rear patch
covers the view; the front patch ends at x=0. Their z separation is 0.08 before
scaling. Geometry and camera translation are scaled together by 0.1, 1, and 10.
Each scale is rendered at 64x32, 160x80, and 320x160. Motion is checked against
independent camera projection. Known patch triangle ranges label cross-patch
matches; this does not identify every possible same-patch self-occlusion.

GPU replay supplies the existing temporal acceptance. Two extra geometric
gates are measured on the CPU:
- Fixed absolute point-to-plane distance <= 0.01.
- Candidate tolerance = previous depth * 1e-5 + 0.5 * normal difference *
  max(projected one-pixel footprint, tangential sample separation).

The second expression is a heuristic, not an established curvature bound.
It uses accurate captured world positions and normals. No production shader
implements it. Radiance is constant to isolate geometric rejection.

## Adaptive-depth results

At width 160 there are seven cross-patch samples at each scale, of which the
adaptive temporal shader wrongly accepts two. The geometric candidate rejects
both at every scale, but rejects roughly 6.3% of otherwise accepted valid
samples as well.

| Scale | Accepted valid samples | Extra valid rejections: fixed | Extra valid rejections: candidate | Remaining false accepts: fixed | Remaining false accepts: candidate |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0.1 | 4,044 | 0 | 255 | 2 | 0 |
| 1 | 4,049 | 0 | 256 | 0 | 0 |
| 10 | 4,046 | 720 | 255 | 0 | 0 |

At width 64 there are no cross-patch samples, so that resolution supplies no
false-acceptance evidence. The candidate rejects 52–53 of 546 accepted valid
samples. At width 320, all nine cross-patch samples are already rejected;
the candidate rejects 519 additional valid samples at each scale.

Scale copies have slightly different GPU acceptance counts, so they are
comparable camera/geometry configurations rather than bit-identical inputs.
The fixed tolerance is clearly scale dependent. The candidate is much more
consistent across scales but too aggressive: rejecting two wrong matches
does not establish a favorable tradeoff against hundreds of valid rejections.
Sample counts are small and this is not a real-scene failure-rate estimate.

## Outcome

Neither fixed world tolerance nor this simple curvature heuristic is ready for
production. Next inspect the rejected valid matches by triangle boundaries and
normal variation, then compare a locally fitted surface model or a better
geometric error bound. Do not compensate by blindly increasing the constant.
Native renderer and viewer behavior remain unchanged.

All 27 GPU replay cases completed. The existing 38 capture/guide/diagnostic
tests passed; lint and diff checks passed.

Reproduce with a fresh output directory for each scale:
```bash
.venv/bin/python -m tools.denoiser_motion.rendered_occlusion --close --scale 1 --output /tmp/close-curved-1
.venv/bin/python -m tools.denoiser_motion.rendered_occlusion --close --scale 0.1 --output /tmp/close-curved-small
.venv/bin/python -m tools.denoiser_motion.rendered_occlusion --close --scale 10 --output /tmp/close-curved-large
```
