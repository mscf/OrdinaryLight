# Direct glass versus reflected glass

The 320x240 capture uses the viewer GI configuration (eight bounces, four
ReSTIR streams, evaluated lobes), with mirror guides off/on. The camera moves
horizontally for 16 frames then remains fixed for 32. A separate undenoised,
ReSTIR-off reference averages eight independent 64-sample batches at the final
pose. Both endpoints therefore share the reference camera. The reference still
has sampling noise and is not ground truth. Fixed exposure/tone mapping is
shared by all displayed panels. Enlarged crops use nearest-neighbor scaling.

## Findings

- The noisy direct glass and smooth reflected glass occur with guides **off**
  as well as on. The selected direct-glass patch is identical across toggles.
- At the last frame, 95.8% of the direct patch has zero prepared guide depth;
  mean denoiser history is about one frame. The reflected patch has nonzero
  depth throughout and roughly 31–32 history frames.
- Primary secondary-signal capture excludes transmitted primary events
  (`ordinarylight_primary_capture_secondary` / the transmission <= 0.001
  fallback). Invalid captured primary positions generate zero guide depth.
  `compose_relax` preserves raw HDR when guide depth is zero. Thus most direct
  glass samples bypass denoising. The mirror's first reflection has a valid
  capture, enabling denoising of radiance that subsequently traverses glass.
- Direct log-luminance temporal standard deviation after stopping is 0.113;
  reflected is 0.00061 off / 0.00140 on. These differently sized, differently
  viewed patches are diagnostic examples, not a general quality comparison.
- Reflected settled reference RMSE changes only from 0.0141 to 0.0136 with
  guides on. Reference split disagreement is 0.0108, so this is not convincing
  evidence of an accuracy gain. The reference contains a faint bright feature
  that is subdued in both denoised images; smoothing alone is not proof of
  correct detail preservation. No clear persistent moving ghost is established
  by these endpoint images; a more detailed refracted target is needed to test it.

This corrects the initial explanation that direct glass merely has imperfect
surface guides: most of the sampled direct patch has **no valid denoising guide**.
The next implementation investigation should address primary-transmission
signal capture and composition, with explicit glass/refraction validation.
The current mirror-guide prototype still does not track refracted geometry.

## Reproduce

Run from the repository root on a working Vulkan desktop:

```bash
PYTHONPATH=. .venv/bin/python artifacts/denoiser-motion/glass-reflection/capture.py
.venv/bin/python artifacts/denoiser-motion/glass-reflection/analyze.py
```

Raw sequences, reference batches and prepared buffers are saved under
`/tmp/glass-comparison`. The compact images and metrics are retained here.
The direct patch is x=100:160,y=205:235; reflected x=145:166,y=140:162.
The scene and renderer are unchanged by these diagnostics.
