# Curved-surface point-to-plane validation

A rendered unit sphere (48 rings, 96 segments) and a rear plane share one mesh
and material. The camera translates 0.3 units. Vulkan supplies the hit positions,
normals and corrected canonical motion. Captured motion is checked against
independent previous-camera projection at all three resolutions.

Known triangle ranges label sphere versus plane. Same-patch matches are treated
as valid for this diagnostic; this is not a general self-occlusion oracle.
Constant radiance isolates geometry gates from photometric rejection. Plane
distance is measured on the CPU using captured world positions and previous
normals; the GPU shader has not been changed to implement it.

## Results

Adding a 0.01-world-unit point-to-plane gate to adaptive depth:

| Resolution | Valid matches | Existing accepted fraction | With plane gate | Additional valid rejections |
| --- | ---: | ---: | ---: | ---: |
| 64x32 | 1,069 | 98.78% | 97.85% | 10 |
| 160x80 | 6,875 | 99.72% | 99.67% | 4 |
| 320x160 | 27,499 | 99.93% | 99.93% | 1 |

The valid population includes both sphere and rear plane. All original,
blanket and adaptive policies already reject all 15, 35 and 101 cross-patch
matches, respectively. Consequently this fixture demonstrates curvature cost,
not improved false-history rejection. It does not contradict the earlier
parallel-patch false acceptance, which used much smaller depth separation.

A threshold sweep from 0.001 to 0.05 is retained in results.json. At 0.02 the
adaptive gate loses only one extra valid match at the lowest resolution and
none at the others. That is not a justification for adopting 0.02: absolute
world-unit tolerances are scene-scale dependent and this fixture has no
remaining false acceptance to constrain the choice.

## Conclusion and next work

The planar result did not generalize without a cost: points on a curved surface
are not on the previous sample's tangent plane. Keep this offline. Next compare
a tolerance derived from projected footprint and local normal variation on
closely spaced curved surfaces, including scaled copies of the same scene.
Require both valid-history retention and wrong-surface rejection before
implementing the gate in production.

Nine GPU replay cases completed, 38 existing regression tests passed, and lint
and diff checks passed. Temporary guide captures include full positions and
plane-distance arrays.

Reproduce:
```bash
.venv/bin/python -m tools.denoiser_motion.rendered_occlusion --curved --output /tmp/curved-occlusion
```
