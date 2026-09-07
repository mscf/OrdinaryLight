# Rendered occlusion validation

The adaptive depth gate also accepts wrong history in an actual rendered
sloped-surface fixture. This supports keeping it experimental.

Two parallel patches separated by 0.08 world units share one mesh and material.
One patch ends halfway across the view. A camera translation exposes the rear
patch. Tests use flat patches and patches with z=x+offset at 64x32, 160x80,
and 320x160. Hit positions and normals come from Vulkan captures.

Correspondence is explicitly reconstructed from current world hit positions
and the previous camera matrix, including the sampled pixel offset. Patch
labels from the known plane equations identify wrong-surface correspondence.
This is a depth-gate experiment, not end-to-end renderer parity: radiance is
constant to prevent photometric rejection from masking geometric failures.

## Sloped-patch results

| Width | Wrong-surface pixels | Original false accepts | Blanket false accepts | Adaptive false accepts | Original valid acceptance | Adaptive valid acceptance |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 1 | 0 | 0 | 0 | 42.0% | 94.2% |
| 160 | 4 | 0 | 4 | 3 | 81.1% | 99.4% |
| 320 | 18 | 0 | 18 | 3 | 98.5% | 99.99% |

The invalid-pixel counts are small. These demonstrate existence of a failure,
not a reliable estimate of its frequency. Flat-patch adaptive validation
rejects all invalid correspondences; blanket 2% accepts all of them.

An independent CPU point-to-plane test uses the previous world normal and
the difference between current and previous hit positions. Requiring absolute
plane distance <=0.01 rejects every wrong-surface match in these fixtures
without rejecting any additional valid matches accepted by each depth policy.
This result is limited to static planar patches with accurate positions.
It does not establish a suitable general threshold for curved, deforming,
or reflected surfaces. The GPU temporal shader does not yet implement this
geometric test.

## Motion discrepancy

An initial run using captured motion instead of reconstructed correspondence
rejected all sloped valid history. Inspection showed a direction discrepancy
between captured motion and projection through the recorded camera matrix.
Those initial runs are superseded by the results here. The capture/replay
motion convention must be audited separately; do not extrapolate these depth
results to live rendering or assume prior camera-motion diagnostics are
unaffected. No native motion code has been changed.

## Reproduce

```bash
.venv/bin/python -m tools.denoiser_motion.rendered_occlusion --output /tmp/rendered-flat
.venv/bin/python -m tools.denoiser_motion.rendered_occlusion --slope 1 --output /tmp/rendered-sloped
```

The final two runs completed all 18 GPU replay cases. Full guide arrays remain
in temporary output directories; JSON results are retained here.
Next priority: audit the motion convention, then test a geometric gate in
shader replay on curved surfaces and the mirror sequence.

## Audit resolution

The discrepancy was confirmed as canonical capture reusing forward general
output motion. Capture now converts to backward pixel-center displacement,
including the sampled-pixel offset. The sloped diagnostic was rerun using
captured motion with an independent projection assertion; see the
[motion audit](../motion-audit/README.md). The original geometry-derived
results above remain useful because the corrected capture reproduces their
correspondence.
