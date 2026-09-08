# First-continuation tracing costs

Native 4K glass-detail, four streams, eight bounces, final-sample guide
optimization enabled. Compared the existing fused secondary intersection/
shading path with separate dispatches. No sample or quality setting changes.

| Configuration | Moving wall median | Stationary wall median | Moving GPU median | Stationary GPU median |
| --- | ---: | ---: | ---: | ---: |
| Fused | 232.9 ms | 211.5 ms | 237.2 ms | 208.3 ms |
| Split | 242.5 ms | 234.9 ms | 238.0 ms | 234.6 ms |

Split execution did not improve end-to-end time and is not promoted. Absolute
GPU rates differ noticeably from previous sessions; do not compare these
numbers with earlier runs as an implementation speedup. Desktop scheduling,
clock and thermal conditions are not controlled. The fused moving secondary
stage averaged 77.8 ms; split intersection + shading averaged 95.5 ms.

A separate 640×480 work-counter run used profiling shader variants solely to
measure ray populations, not to compare execution times. Last stationary
sample reported:

| Bounce | Path rays |
| --- | ---: |
| 0 | 1228800 |
| 1 | 1228800 |
| 2 | 224632 |
| 3 | 213858 |
| 4 | 38239 |
| 5 | 34452 |
| 6 | 2034 |
| 7 | 1685 |

Total: 2972500 rays. First two bounces: 82.7%; bounces 4–7: 2.6%.
Thus reducing maximum depth has little leverage over ray count in this
fixture, while potentially damaging glass/reflection paths. This is a ray
population observation, not a claim that each ray costs the same. Zero
shadow/ReSTIR counters are retained as reported, not interpreted as proof
that those operations are absent from every custom shader path.

The next architectural experiment is to reduce first-continuation path-state
queue traffic, e.g. a short inline continuation. Existing hybrid execution
needs transmission-guide correctness validation before any viewer default
change; raw speed alone is insufficient.

From repository root with native Vulkan/display:

```sh
DIRECT=1 PROFILE=0 OUT=/tmp/fused PYTHONPATH=. .venv/bin/python artifacts/denoiser-motion/tracing-costs/profile.py
SPLIT=1 DIRECT=1 PROFILE=0 OUT=/tmp/split PYTHONPATH=. .venv/bin/python artifacts/denoiser-motion/tracing-costs/profile.py
DIRECT=1 PROFILE=0 OUT=/tmp/counts PYTHONPATH=. .venv/bin/python artifacts/denoiser-motion/tracing-costs/count_rays.py
```

No production configuration was changed in this investigation.
