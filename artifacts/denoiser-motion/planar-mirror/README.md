# Static planar-mirror guide prototype, 2026-09-07

**Follow-up:** The object-motion results below are superseded by the
[corrected rejection diagnosis](../mirror-ablation/README.md). Direct transform
assignment bypassed scene revision notification in that initial prototype.
The follow-up fixes the motion setup and measures finite-hit acceptance.

**The prototype is not ready for live integration.** Reflected-surface guides
preserve more edge contrast, but the current filter produces substantially
more noise and higher image error than mirror-surface guides.

## Implemented scope

A fixed, white, perfect mirror at z=0. Scene geometry lies in front of the
plane. A reflected camera captures the scene without the mirror; its raster
is flipped horizontally to restore reflection handedness. World positions
and normals are reflected across z=0, and motion X is negated after the raster
flip. Previous-depth validation uses reflected previous positions, including
the known translation of the moving object.

The actual scene's primary hit mask determines which screen pixels see the
mirror, including foreground occlusion. Both denoiser variants receive
**identical reflected-camera radiance samples** inside that mask. The baseline
uses the mirror's primary guides; the experiment uses the reflected scene's
surface guides. This isolates guide behavior without claiming native PSR
transport integration.

This is an offline reflected-camera construction, not a replacement for
the live renderer's first-bounce tracing. It does not support moving/curved
mirrors, multiple mirror chains, transmission, or general scene geometry
between the reflected camera and the mirror plane. It adds no production
renderer option and is not exposed in the feature viewer.

## Results

Ten frames at 160×120, with initial hold, four movement steps and four
additional stopped frames. Identical filter settings in both variants:
history limit 8, normal threshold 0.95, depth threshold 0.005, clamp sigma 1,
reactive sigma 2.5, three spatial iterations at the existing color weight 4.
The material ID acts as identity in replay; each fixture object has a distinct
material. References average two independent 64-sample reflected-camera
renders per frame. Comparisons use only visible mirror pixels.

| Metric | Camera / primary | Camera / reflected | Object / primary | Object / reflected |
| --- | ---: | ---: | ---: | ---: |
| Log-RGB RMSE | 0.014766 | 0.026349 | 0.012967 | 0.025964 |
| Signed reference-edge gain | 0.8672 | 0.9502 | 0.9064 | 0.9408 |

Lower RMSE is better. Edge gain closer to one means better retained reference
contrast, measured from log-luminance gradients above 0.02. Better edge gain
does not compensate for the substantial noise increase in this test.

The moving reflected object exposes 61 sampled background pixels across the
sequence. Their log-RGB error is 0.025879 with primary guides and 0.065190
with reflected guides. This small mask is not enough to generalize about
disocclusions, but it does not establish a benefit.

Split-reference log-RGB RMSE is about 0.00574–0.00576; references remain noisy.
A stationary-camera consistency check found zero discrepancy between mapped
expected depth and captured view depth, with stationary motion below
0.000034 pixels. The reflected camera sees many background misses, which lack
finite surface depth; low average history retention must not be attributed
solely to rejection of valid reflected hits.

## Visuals and validation

Panels: reference, primary guides, reflected guides, raw reflection samples.

- [Camera moving](camera-moving.png), [camera stopped](camera-stopped.png).
- [Object moving](object-moving.png), [object stopped](object-stopped.png).
- Full metrics: [camera](camera.json), [object](object.json).

Eight focused tests pass, including reflection involution, projection
handedness, depth equivalence through clip projection, motion-component
mapping, masking, and input preservation. Both GPU sequences completed;
the object run was repeated with explicit revealed-pixel measurement.
Ruff and whitespace checks pass.

Run with a new output directory:

```bash
python -m tools.denoiser_motion.planar_mirror --motion camera --output /tmp/mirror-camera-new
python -m tools.denoiser_motion.planar_mirror --motion object --output /tmp/mirror-object-new
```

Raw arrays, masks, histories and complete frame galleries are written to the
requested directory. The compact report and selected images are retained here.

## Next decision

Do not add native guide buffers yet. First measure temporal acceptance only
on finite reflected hits and isolate spatial filtering's noise/detail tradeoff
on those surfaces. A reflection-aware history policy or different spatial
guidance may be needed. These results establish a testable geometric mapping,
not a demonstrated denoising improvement.
