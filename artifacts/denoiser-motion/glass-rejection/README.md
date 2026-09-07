# Targeted glass history policies (diagnostic only)

Native temporal shader variants are compiled into a temporary directory and
loaded only by capture.py. Production shaders and runtime defaults are unchanged.
The fixture's glass instance has identity 0; both variants explicitly select
that identity. This is not a general material classifier and must not be used
as a production selection rule.

- **cap4**: limit history length to four when glass guide motion exceeds one
  pixel. Once motion subsides, history can grow normally. This limits reuse;
  it does not introduce a new binary rejection condition.
- **reactive1**: use reactive sigma 1 for the glass, including while stationary,
  instead of the normal policy's reactive setting. It retains the existing
  minimum radiance tolerance and geometry-compatible neighborhood calculation.

All three motion sequences use the same fixture, frame counts and independent
references as the glass-detail diagnostic. Camera/target histories have their
existing global policies; cap4 is an additional per-pixel limit, not a replacement.

| Motion | Baseline moving / settled RMSE | cap4 | reactive1 |
| --- | --- | --- | --- |
| Camera | .0606 / .0169 | .0606 / .0169 | .0500 / .0226 |
| Glass | .1529 / .0376 | .1216 / .0199 | .0569 / .0246 |
| Target | .1751 / .0233 | .1751 / .0233 | .0396 / .0158 |

For glass motion, cap4 reduces moving error by 20% and settled error by 47%.
Its stationary temporal standard deviation drops from .00536 to .00255.
Camera and target metrics are unchanged. In camera motion the existing global
policy already limits history; for target motion the glass surface remains
stationary, so a glass-guide motion cap cannot follow the refracted target.

Reactive rejection helps target motion but increases camera-case stationary
noise from .00182 to .00657, with visible speckling and worse settled error.
It is not a suitable blanket default based on these results. cap4 is the better
candidate for broader validation across speed, IOR and geometry before adding
a runtime option with an explicit transmissive-material mask.

The comparison patch x=125:195,y=85:155 is fixed, and can include a small amount
of background at its corners for shifted glass poses. It is shared across
policies and references, but is not a perfect glass-only segmentation. Earlier
fixture documentation calling it wholly inside the sphere was too strong.
Reference split RMSE is .0125/.0129/.0148 for camera/glass/target respectively.
These are one-fixture diagnostic results, not universal quality guarantees.

## Reproduce

First generate matching references:

```bash
PYTHONPATH=. .venv/bin/python -m tools.denoiser_motion.glass_detail --output /tmp/glass-detail-fixed
PYTHONPATH=. .venv/bin/python artifacts/denoiser-motion/glass-rejection/capture.py
```

The script compiles validated temporary SPIR-V and runs baseline/cap4/reactive1
through the same native tracing and denoising pipeline. It writes raw arrays
and temporary shader sources to /tmp/glass-rejection. Retained source diffs,
metrics, moving-frame panels and settled panels are alongside this document.
No experimental setting is yet exposed in the viewer.
