# Four-frame glass-motion cap: broader fixture sweep

Eight paired native captures vary one dimension at a time: IOR 1.1/1.33/1.52/1.8,
Z thickness scale 0.5/1/1.5, and translation over 8/16/32 moving frames. All
translate from x=-0.3 to +0.3 and then hold for 32 frames. Speed is defined per
rendered frame, not wall-clock time. Thickness scaling produces oblate/prolate
solid glass; this is not a hollow shell or thin-walled material test.

Each final-pose reference averages eight independent 64-sample batches. The
same reference is reused for speed variants at matching geometry/IOR. Baseline
and cap use the same ray sample sequence. Both use viewer GI settings and
ReSTIR; the references are undenoised and ReSTIR-off. Metrics are log luminance
on the same fixed patch as the preceding glass-detail test. The patch may include
some background near its corners. Tests are at 320x240.

Positive values below mean improvement relative to baseline:

| Case | Moving-end RMSE reduction | Settled RMSE reduction | Stopped temporal-noise reduction |
| --- | --- | --- | --- |
| low-ior | 20.2% | 59.3% | 57.5% |
| water-ior | 25.0% | 49.9% | 54.8% |
| glass-ior | 20.5% | 46.9% | 52.5% |
| high-ior | 23.1% | 48.7% | 52.8% |
| thin | 23.4% | 43.5% | 52.9% |
| thick | 23.2% | 52.2% | 53.5% |
| fast | 13.8% | 24.6% | 29.5% |
| slow | 24.5% | 56.1% | 60.8% |

The cap improves all three reported metrics in every tested case. This supports
further integration work, not a universal quality claim: it is one target,
one motion axis, one resolution and one material family. Reference split RMSE
ranges from .0080 to .0174; the absolute settled errors should be interpreted
alongside that uncertainty. Moving-end samples are sparse endpoints, not an
aggregate metric over the full motion sequence.

## Integration decision

Keep this as an experimental candidate. The temporary shader identifies glass
by this fixture's instance identity 0, and caps history only when guide motion
exceeds one pixel. It must not be promoted with that hard-coded identity.
A runtime option needs explicit transmissive-material metadata carried into
the denoising guides, followed by a parity test against this diagnostic. A
subsequent mixed opaque/glass scene should verify that opaque history is
unchanged. The cap does not address background motion behind stationary glass;
that remains a separate limitation established by the earlier comparison.

No production shaders or viewer settings changed in this sweep.

## Reproduce

```bash
PYTHONPATH=. .venv/bin/python artifacts/denoiser-motion/glass-cap-sweep/capture.py
```

Run from the repository root with a working Vulkan desktop. The script compiles
a validated temporary shader and routes only the diagnostic temporal pipeline
to it. Raw sequences, reference batches and temporary shader files are saved
under /tmp/glass-cap-sweep. Metrics and the common-exposure settled contact sheet
are retained here. It does not require the preceding experiment's temporary data.
