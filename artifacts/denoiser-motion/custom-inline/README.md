# Custom-material inline continuation probe

The current custom-material pipeline forces standard wavefront execution,
even if hybrid was requested. The diagnostic shim compiles the same custom
primary source with WAVE_HYBRID=1 and dispatches that pipeline with hybrid
continuation scheduling (three inline bounces). It does not edit production
execution policy. The scene remains the glass-detail custom-material fixture.

This exposed a dormant compiler injection bug: custom termination emitted
`break` inside the single-bounce function. That function returns a boolean;
termination must return false. The compiler fix and a hybrid GLSL compilation
regression test are retained. Production standard-wavefront behavior is
unchanged because the inline body is preprocessed out there.

Diagnostics now record requested, resolved, and actually dispatched strategy,
so a custom-material fallback cannot masquerade as a hybrid measurement.

Sequential native 4K comparison, same four streams/eight bounces:

| Mode | Moving wall median | Stationary wall median | Moving GPU median | Stationary GPU median |
| --- | ---: | ---: | ---: | ---: |
| Standard | 217.31 ms | 210.79 ms | 209.00 ms | 209.30 ms |
| Inline prototype | 182.49 ms | 173.42 ms | 173.47 ms | 174.26 ms |

About 17% lower GPU time, still far from 33.3/16.7 ms budgets. Absolute rates
vary with desktop/GPU conditions; this is a short paired experiment.

Three 48-frame 320×240 sequences compared glass, target, and camera motion.
HDR RMSE was approximately .00156, .00156, .00170 respectively; maximum
absolute differences were .242, .515, .264. These are raw linear-HDR
differences, not comparisons against a converged quality reference.

Settled depth, motion, normal/roughness, identity and diffuse signal images
matched exactly in all three cases. Specular hit distances also matched
(including one shared nonfinite FP16 distance in the glass case), but specular
RGB differed by up to roughly .07. No new nonfinite values were introduced in
that comparison. The specular discrepancy remains unexplained; therefore the
prototype is not enabled in the viewer and no quality-equivalence claim is
made. Next work is to isolate inline versus staged specular transport.

Reproduce from repository root with native Vulkan/display access:

```sh
OUT=/tmp/standard PYTHONPATH=. .venv/bin/python artifacts/denoiser-motion/custom-inline/check.py
INLINE=1 OUT=/tmp/inline PYTHONPATH=. .venv/bin/python artifacts/denoiser-motion/custom-inline/check.py
DIRECT=1 PROFILE=0 OUT=/tmp/standard-4k PYTHONPATH=. .venv/bin/python artifacts/denoiser-motion/custom-inline/profile.py
INLINE=1 DIRECT=1 PROFILE=0 OUT=/tmp/inline-4k PYTHONPATH=. .venv/bin/python artifacts/denoiser-motion/custom-inline/profile.py
```

The diagnostic imports inline_probe from its own directory. Raw frame and
guide arrays from this session remain in /tmp/inline-standard and
/tmp/inline-prototype. The shader dump is /tmp/inline-primary.glsl.

Follow-up: the [incoming emission MIS investigation](../incoming-emission-mis/README.md)
identified and corrected the main transport discrepancy in the typed staged
shader. The first continuation now matches exactly; isolated multi-bounce
outliers still require investigation before enabling custom inline execution.
