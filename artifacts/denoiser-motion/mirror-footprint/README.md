> Superseded motion inputs: these historical replay results predate the
> canonical capture motion correction. See the
> [motion audit](../motion-audit/README.md) before interpreting them.

# Adaptive depth-footprint experiment

This is an offline shader experiment; native renderer shaders and defaults are
unchanged. It uses the corrected captures described in
[the preceding ablation](../mirror-ablation/README.md).

The experimental shader estimates depth variation over one pixel from the
previous depth image. For each axis it takes the smaller available depth
difference to adjacent pixels with matching identity/material and compatible
normals. The sum augments the original 0.5% tolerance, capped at 2%.
Flat neighborhoods retain the original tolerance. This is a screen-space
finite-difference approximation, not an analytic projected surface derivative.

| Motion | Depth policy | Finite-hit acceptance | Finite-hit log-RGB RMSE |
| --- | --- | ---: | ---: |
| Object | Original 0.5% | 30.3% | 0.06195 |
| Object | Blanket 2% | 67.6% | 0.04964 |
| Object | Adaptive footprint | 66.8% | 0.05001 |
| Camera | Original 0.5% | 39.3% | 0.06382 |
| Camera | Blanket 2% | 78.2% | 0.04743 |
| Camera | Adaptive footprint | 75.0% | 0.04855 |

Acceptance excludes the initial frame and background misses. Error includes all
frames, on finite reflected hits only. Adaptive error improves 19.3% and 23.9%
over original reflected guides. Object revealed-region error is 0.02422 for both
adaptive and blanket policies, versus 0.02433 originally; this is not evidence of
a significant disocclusion improvement. Primary-guide error remains lower
(0.03795 object, 0.03531 camera), with visibly stronger smoothing.

Both saved ten-frame captures were replayed on the GPU through all nine variants.
Unmodified primary/reflected baseline outputs matched the source captures exactly.
The existing 35 diagnostic/signal tests and lint checks passed.

## Limits and next validation

Measured depth differences include stochastic ray jitter and curvature; they
can overestimate local slope. Matching identity and normals does not guarantee
the same continuous surface, especially near self-occlusion. This fixture uses
unique materials as instance identities. A 2% cap also makes this experiment
incapable of demonstrating that the cap is unnecessary.

Before native integration, test flat/sloped surfaces and same-object depth
discontinuities at multiple resolutions with explicit false-acceptance masks.
Compare an analytic plane-distance or projected-footprint test against this
finite-difference approximation. No production threshold change is justified yet.

## Reproduce

Use the replayable corrected object and camera captures from the preceding
ablation, then:

```bash
.venv/bin/python -m tools.denoiser_motion.ablate_mirror /tmp/mirror-object-corrected --output /tmp/mirror-object-footprint
.venv/bin/python -m tools.denoiser_motion.ablate_mirror /tmp/mirror-camera-signals --output /tmp/mirror-camera-footprint
```

Full arrays remain temporary; JSON measurements and representative images are
retained here.

## Motion comparisons and follow-up

[Object motion animation](object-motion.apng) and
[camera motion animation](camera-motion.apng) show all ten captured frames,
synchronized across columns. Playback is deliberately slowed to 300 ms per
frame, with a longer final hold; the loop reset is a discontinuity. Images use
the same tone mapping as the still comparisons and nearest-neighbor enlargement.
These short sequences are useful for comparison, not representative real-time
playback or a long convergence test.

**Follow-up found false history acceptance:** see the
[synthetic occlusion stress results](../footprint-stress/README.md).
The adaptive policy is not ready for native integration.

Rebuild an animation with:
```bash
.venv/bin/python -m tools.denoiser_motion.animate_mirror /tmp/mirror-object-footprint/arrays.npz --output /tmp/object-motion.apng
```
