> Motion audit clarification: this study uses live native preparation and
> denoising via `present_wavefront`. It does not consume canonical captured
> motion and is unaffected by that capture fix. Its noise/detail tradeoff
> conclusions remain applicable.

# Broader-filter validation, 2026-09-07

**Retain the default color weight of 4.0.** Weight 2.0 reduces noise and
recovery error in these fixtures, but measurably softens textured detail.
It remains useful as an explicit user preference.

## Results

Lower normalized log-luminance RMSE is better.

| Sequence | Phase | Default (4) | Broader (2) |
| --- | --- | ---: | ---: |
| Moving object | Whole sequence | 0.128713 | 0.106621 |
| Moving object | Moving | 0.106563 | 0.091654 |
| Moving object | Stopped | 0.069796 | 0.064774 |
| Camera | Whole sequence | 0.141220 | 0.116338 |
| Camera | Moving | 0.146676 | 0.124062 |
| Camera | Stopped | 0.072046 | 0.064420 |
| Textured floor | Whole sequence | 0.235929 | 0.208808 |
| Textured floor | Moving | 0.259650 | 0.234040 |
| Textured floor | Stopped | 0.149125 | 0.146376 |

Whole-sequence error falls approximately 17%, 18%, and 11%, respectively.
Object-disocclusion log-luminance error falls from 0.021098 to 0.016848.
Error in the revealed region after stopping falls from 0.006446 to 0.005651.
These masks include pixels vacated by the moving sphere, rather than only
its current silhouette. They provide no evidence of worse trailing in this
sequence, but do not prove arbitrary scene motion is safe.

### Fine-detail cost

For the checker-textured floor, signed reference-gradient gain falls:

| Phase | Default | Broader |
| --- | ---: | ---: |
| Whole sequence | 0.8016 | 0.7192 |
| Moving | 0.7529 | 0.6706 |
| Stopped | 0.8192 | 0.7456 |

Gain measures alignment with reference edges:
sum(candidate gradient · reference gradient) / sum(reference gradient²),
on floor pixels with reference log-luminance gradient above 0.02.
A value of one corresponds to matched reference-gradient amplitude.
The broader setting retains roughly 10% less contrast overall and 9% less
after stopping, relative to the default. Gradient RMSE improves because
noise also decreases; it alone would obscure this softening tradeoff.

[Textured comparison](detail.png), [object comparison](object.png),
[camera comparison](camera.png). Top row is the final moving frame; bottom
row follows seven additional frames at the stopped pose. Reference, default,
and broader panels share the same display transform.

## Protocol

- Native Vulkan captures, 320×180, eight bounces, three A-trous iterations.
- Evaluated lobes disabled; three-frame motion-history floor; two ReSTIR
  reservoirs and four candidates. Only color weight differs.
- Fifteen frames: three at the initial pose, five movement steps, and seven
  further frames at the final pose. Object animation advances from time zero
  to 1.5; camera sequences traverse a 0.25-radian arc.
- Shared independent 512-sample live references per distinct pose (eight
  64-sample batches, seeds starting at 1000). The same reference is reused
  across held frames, preventing reference noise from masquerading as
  recovery flicker. Candidate seeds remain independent.
- Split-reference log-luminance RMSE is approximately 0.00337, 0.00339 and
  0.00388 for object, camera and detail sequences.
- Masks use stable packed-instance keys specific to these fixtures. Floor
  normals can point downward, so the texture mask uses identity, not sign.
- Three live scenarios were executed successfully; the diagnostic passes
  Ruff and whitespace checks. No production renderer changes were required
  after checkpoint 69fd5322.

This validates the tested resolution and short movement/recovery sequences,
not arbitrary animation, frame rates, resolutions or material complexity.
No timing claim is made. Full metric and configuration JSON files are
retained beside this report; raw arrays remain in the temporary capture
directories.

Reproduce using a new output directory per kind:

```bash
python -m tools.denoiser_motion.validate_broader --kind object --output /tmp/broader-object-new
python -m tools.denoiser_motion.validate_broader --kind camera --output /tmp/broader-camera-new
python -m tools.denoiser_motion.validate_broader --kind detail --output /tmp/broader-detail-new
```

The next filter improvement should reduce noise while protecting material
detail, rather than uniformly broadening luminance tolerance. Keep the
existing viewer checkbox available for interactive preference testing.
