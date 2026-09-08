# Isolating the remaining background outlier

Continuation of incoming-emission-mis. No production shader/execution change
was made here. GPU performance was deliberately not benchmarked while other
work was running.

Raw-HDR target-motion sequence, denoising disabled, 48 frames at 320×240:

| Maximum bounces | Maximum absolute difference |
| --- | ---: |
| 2 | 0 |
| 3 | .0009765625 |
| 4 | .0009765625 |
| 8 | .3668213 |

The largest eight-bounce outlier is frame 44, pixel (x=179,y=175). Isolating
streams while preserving original sample index/count for ray generation
identified stream 2. Streams 0 and 1 matched at that pixel; the last stream's
handoff also matched except for a one-ULP throughput component.

For stream 2, full-precision state at bounce-three handoff showed identical
RNG (bitwise) and throughput, but direction differed by about 6e-7 and origin
by about 5e-7. The later ray outcome is consistent with an emissive-bar hit
versus an environment miss. Narrow `precise` qualifiers on hit/normal/weight
locals did not align the ray. A broader qualifier experiment failed shader
compilation and was discarded; no precision policy was added to production.

## Causal replay

The diagnostic replay substitutes only the standard path's exact handoff
origin/direction for this pixel in the inline run. Raw HDR at frame 44:

| Run | R | G | B |
| --- | ---: | ---: | ---: |
| Standard | 1.2880859 | 1.4257812 | 1.5 |
| Inline | .0150528 | .02037048 | .03115845 |
| Inline with standard handoff ray | 1.2880859 | 1.4257812 | 1.5 |

This reproduces the standard result exactly and establishes that the tiny
handoff geometry difference causes this outlier. It does not establish that
every possible scene/path difference is harmless. Remaining quality acceptance
should compare noise/bias against references over sequences, rather than
requiring bitwise equality through sensitive multi-refraction paths.

The debug shim stores origin/direction/throughput/RNG into unused-for-raw-
output secondary signal fields, then copies that buffer after GPU completion.
Denoising is disabled for these tests. `SAMPLE` chooses one stream but keeps
its original ray-generation sample count/index; only the HDR resolve is set
to overwrite at unit weight for easy inspection. `REPLAY_RAY` is hard-coded
to the diagnosed pixel/ray and is never a renderer fix.

Reproduce from repository root with native Vulkan/display access:

```sh
SAMPLE=2 BOUNCES=8 OUT=/tmp/handoff-standard PYTHONPATH=. .venv/bin/python artifacts/denoiser-motion/inline-handoff/check.py
SAMPLE=2 INLINE=1 BOUNCES=8 OUT=/tmp/handoff-inline PYTHONPATH=. .venv/bin/python artifacts/denoiser-motion/inline-handoff/check.py
REPLAY_RAY=1 SAMPLE=2 INLINE=1 BOUNCES=8 OUT=/tmp/handoff-replay PYTHONPATH=. .venv/bin/python artifacts/denoiser-motion/inline-handoff/check.py
```

`handoff.npy` contains final-frame full-precision secondary state; index
[175,179,1] is direction/RNG, [175,179,2,:3] origin, [175,179,3,:3] throughput.
`target-result.npy[44,175,179]` is the raw pixel above. `PRECISE=1` enables the
rejected narrow precision probe. Default viewer execution remains unchanged.
