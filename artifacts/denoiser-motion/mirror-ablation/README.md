# Reflected-guide rejection diagnosis, 2026-09-07

**Depth validation is a major bottleneck in this low-resolution prototype.**
Increasing its relative tolerance from 0.5% to 2%, while retaining normal,
material and identity checks, substantially improves valid-surface history
acceptance. It does not yet beat the smoother primary-guide baseline on
image error.

## Corrected experiment

The earlier object-motion prototype assigned mesh transforms directly,
bypassing scene revision notification. GPU geometry could remain stale while
CPU motion bookkeeping used the new transform. Those object-motion results
are superseded, including the earlier 61-pixel revealed-region result.

The prototype now calls `Scene.update_instance_transforms`. A regression test
checks that movement increments the scene transform revision and repeating
the stopped pose does not. Both camera and corrected object sequences were
recaptured with replayable signals and expected previous depths.

A separate replay fix preserves temporal textures when spatial iterations
are zero. Ten-frame temporal-only runs now retain usable history.

## Measured ablations

Acceptance excludes the initial frame and includes **only finite reflected
surface hits inside the visible mirror**. Background misses are excluded.
Only about 14–16% of mirror pixels in these fixtures have finite reflected
hits, so whole-mirror history averages were misleading.

| Sequence / guides | Finite-surface acceptance | Finite-surface log-RGB RMSE |
| --- | ---: | ---: |
| Object / primary | 84.9% | 0.037948 |
| Object / reflected, 0.5% depth tolerance | 30.3% | 0.061949 |
| Object / reflected, zero reactive sigma | 31.6% | 0.061648 |
| Object / reflected, 2% depth tolerance | 67.6% | 0.049636 |
| Camera / primary | 93.8% | 0.035307 |
| Camera / reflected, 0.5% depth tolerance | 39.3% | 0.063817 |
| Camera / reflected, zero reactive sigma | 40.8% | 0.063482 |
| Camera / reflected, 2% depth tolerance | 78.2% | 0.047428 |

Wider depth tolerance reduces reflected-surface error about 20% for the
object and 26% for the camera. After stopping, reflected acceptance rises
from 31% to 80% and from 50% to 86%, respectively.

The temporal-only runs already show a substantial error gap: object
0.049961 primary versus 0.068175 reflected; camera 0.045202 versus 0.069727.
Spatial filtering therefore is not the sole source of this regression.

Broader normal weighting (power 8 instead of 32) has little effect.
Broader luminance tolerance (weight 2 instead of 4) improves reflected error
to 0.056577 and 0.058582, but does not address history rejection.

The zero-reactive mode is not a perfectly isolated photometric switch:
the current shader also uses positive reactive sigma to enable its
surface-local neighborhood statistics. Its small benefit nevertheless does
not explain the much larger effect of changing only depth tolerance.

In the corrected object run, revealed-region error is 0.017130 for primary
guides, 0.024331 for reflected guides, and 0.024221 with wider depth tolerance.
The wider tolerance does not establish improved disocclusion quality.

## Interpretation

The offscreen renderer jitters primary rays per sample. Neighboring samples
of a sloped or curved reflected surface can have appreciably different depths
even with negligible geometric motion. The fixed relative threshold at
160×120 can reject these samples. Depth/motion consistency was checked in
the stationary-camera construction; the measured depth-tolerance ablation
identifies a concrete sensitivity, not proof of a generally safe threshold.

The next design should account for projected pixel footprint and surface
slope, retain identity/material/normal rejection, and be checked across
resolutions and newly revealed surfaces. Do not replace the native depth
threshold with 2% based on these fixtures.

## Artifacts and validation

[Object comparison](object.png), [camera comparison](camera.png). Moving frame
above, stopped frame below. Columns are reference, primary guides, reflected
guides at 0.5%, and reflected guides at 2%.

Full metrics are in `object.json` and `camera.json`; the corrected object
capture report is also retained. Baseline replays are image-exact against
their source captures. Both sets of eight GPU variants completed, including
temporal-only runs. Thirty-five focused tests pass, with Ruff and whitespace
checks passing. No production renderer behavior changes were made.

Run a new replayable capture and ablation:

```bash
python -m tools.denoiser_motion.planar_mirror --motion object --output /tmp/mirror-capture-new
python -m tools.denoiser_motion.ablate_mirror /tmp/mirror-capture-new --output /tmp/mirror-ablation-new
```
