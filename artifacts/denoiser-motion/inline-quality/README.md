# Inline continuation quality gate and viewer integration

New 512-sample references use the corrected staged emission MIS code,
ReSTIR/denoising/temporal accumulation disabled, eight independent 64-sample
batches at moving-middle and final poses. Three 48-frame sequences compare
standard and inline execution (glass, target and camera motion).

Metrics use log1p luminance. Both a central 70×70 patch and the full frame are
reported; the full-frame metrics include regions outside the central patch
where prior isolated outliers were found. Noise is mean pixel temporal
standard deviation over the final 16 stationary frames. Bias is signed mean
error, not an estimate over many independent full-sequence trials.

| Motion | Standard settled full-frame RMSE | Inline | Standard temporal noise | Inline |
| --- | ---: | ---: | ---: | ---: |
| Glass | .03608095 | .03608092 | .00878778 | .00878775 |
| Target | .03845330 | .03845330 | .00917956 | .00917968 |
| Camera | .03574245 | .03574245 | .00845006 | .00845006 |

Moving-middle/end errors and signed mean errors likewise match to the shown
precision in the JSON reports. Patch reference split-half RMSE is .0125–.0148,
much larger than the aggregate differences between modes. This is limited
quality acceptance for these fixtures, not proof of bitwise equivalence or
quality across arbitrary scenes. The known isolated path differences remain
possible; see inline-handoff for their causal investigation.

The integrated (non-monkey-patched) implementation produces byte-identical
HDR sequences to the validated diagnostic prototype for all three fixtures.
301 actual viewer frames completed without presentation failure, including
F11 fullscreen entry and Escape restoration. The harness asserted both the
active custom-inline configuration and actual dispatched hybrid strategy.
Timing samples in viewer.log are not controlled performance measurements;
other GPU work may have been running.

## Integration

`wavefront_custom_inline=True` is opt-in and requires explicit hybrid strategy
with three inline bounces. Dynamic custom primary compilation activates the
inline body; shader cache signatures include the option. Custom hybrid
pipelines are retained even for opaque scenes instead of being replaced by a
stock opaque specialization. Default custom execution remains wavefront.

The viewer exposes Inline continuations (experimental) only for
`glass-detail-camera`, `glass-detail-motion`, and `glass-detail-target`.
Enable it and Apply and restart renderer. Other scene/target selections do
not activate it. It is not a general stock-material/volume/hybrid quality claim.

Reference generation and integrated capture scripts require native Vulkan:

```sh
PYTHONPATH=. .venv/bin/python artifacts/denoiser-motion/inline-quality/reference.py
OUT=/tmp/inline-integrated PYTHONPATH=. .venv/bin/python artifacts/denoiser-motion/inline-quality/capture.py
```

Reference arrays remain under /tmp/inline-quality-reference; integrated
sequences under /tmp/inline-production. No new speedup claim was made during
this quality gate. Earlier ~17% measurements remain provisional until paired
benchmarks can run without competing GPU work.
