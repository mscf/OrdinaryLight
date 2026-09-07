# Geometric gate: rendered-radiance comparison

[Animated comparison](comparison.apng) · [moving frame](moving.png) ·
[settled frame](settled.png)

Columns are reference (64 spp), raw capture, adaptive-depth baseline, and
adaptive-depth plus geometric gate. Both filters receive identical noisy
radiance and geometry. The sequence translates the camera, reverses, returns,
and holds for ten frames. Playback is slowed to 200 ms/frame with a final hold.
The image uses common Reinhard tone mapping and gamma 2.2; nearest-neighbor
enlargement makes pixels visible.

The folded, two-patch scene now has an area emitter and three-bounce rendered
radiance. Both replay paths use three spatial iterations and independent
temporal histories. No constant-radiance substitution is used. This remains
static geometry with a 45-degree camera, not a mirror or moving-object test.

| Log-RGB RMSE | Baseline | Geometric |
| --- | ---: | ---: |
| Whole sequence | 0.020663 | 0.020767 |
| Moving frames | 0.017979 | 0.018264 |
| Settling frames | 0.013244 | 0.013257 |

The geometric variant is approximately 0.5% worse overall and 1.6% worse
during motion in this run. The inspected moving frame looks very similar.
This does not establish a visual-quality improvement. References have only
64 samples; no independent reference split or uncertainty estimate is available.
Do not interpret these small differences as a robust general regression either.

## CPU/GPU discrepancy resolved

The earlier oracle classified matches using full-precision motion, whereas
the shader receives half-precision motion. Near rounding boundaries that can
select a different previous pixel. Repeating the four constant-radiance
sequences with CPU masks using uploaded motion and normal precision gives
matching CPU predictions and GPU counts:

| Width / grid | Predicted and actual accepted valid | Predicted and actual wrong accepts |
| --- | ---: | ---: |
| 160 / 16 | 93,182 | 0 |
| 160 / 64 | 95,719 | 0 |
| 320 / 16 | 379,932 | 0 |
| 320 / 64 | 385,182 | 0 |

The previous apparent one false accept and two lost valid matches were artifacts
of comparing against a different pixel-selection/precision convention. This
agreement is limited to the tested sequences and does not prove universal
floating-point parity.

## Status

Experimental replay only; native renderer and raster_feature_viewer unchanged.
Next use a fixture where wrong history produces clearly distinguishable
radiance, then inspect disocclusion regions alongside noise and temporal
stability. Current evidence does not justify promoting the gate.

Validation: eight 12-frame precision-audit runs and a paired 20-frame visual
run completed. Existing 38 regression tests and lint/diff checks passed.
Full radiance arrays remain in the temporary capture directory.

Reproduce:
```bash
.venv/bin/python -m tools.denoiser_motion.visual_plane --output /tmp/visual-plane
```
