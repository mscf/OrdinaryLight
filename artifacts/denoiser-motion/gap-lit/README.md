# Gap-lit disocclusion comparison

[Animation](comparison.apng) · [moving frame](moving.png) · [settled frame](settled.png)

Columns: 256-sample reference, noisy input, adaptive-depth baseline, geometric
gate. The camera translates, reverses, then holds. Both filters receive the same
radiance. An area emitter lies between nearby sloped patches, intended to
illuminate the rear patch while the foreground hides the source.

Visual inspection shows weaker contrast than intended. This is a completed
gap-lit experiment, not evidence from a successful strong-contrast stress scene.

| Log-RGB RMSE | Baseline | Geometric |
| --- | ---: | ---: |
| Whole sequence | 0.020278 | 0.020318 |
| Moving | 0.018371 | 0.018465 |
| Settling | 0.013415 | 0.013433 |
| Rear-edge mask | 0.066047 | 0.068672 |

The gate is approximately 0.2% worse overall and 4% worse in the mask.
No visual benefit is established. The reference split difference is 0.008564
log-RGB RMSE (two independent 128-sample halves); this is not a confidence
interval on the paired filter difference.

The 442-pixel mask selects current rear-patch hits whose half-precision
reprojection lands on a previous non-rear pixel. Non-rear includes foreground,
background and emitter, so it is a rear-edge correspondence mask, not a pure
foreground-disocclusion ground truth. The JSON retains the older
'revealed_pixels' and 'revealed_log_rgb_rmse' names for these fields.

Recommendation: pause promotion of this geometric gate. Acceptance diagnostics
demonstrated improved cross-patch rejection, but neither rendered-radiance
comparison has established a visual win. Native renderer/viewer remain unchanged.
The canonical motion correction is independently useful and already committed.

The paired 20-frame GPU run completed with 256-sample references. Existing
38 regression tests and lint/diff checks passed.

Reproduce:
```bash
.venv/bin/python -m tools.denoiser_motion.visual_plane --contrast --output /tmp/visual-plane-contrast
```
