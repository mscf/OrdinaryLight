# GI composition validation

2026-09-08, local Vulkan GPU.

- Six 24-frame 640×480 sequences at 50% scale: glass, target and camera motion,
  each with `fsr1` and `fsr1-shade`.
- All display arrays were byte-identical to the pre-extraction EASU captures
  in `/tmp/easu-shade`. New arrays were saved to `/tmp/gi-graph-capture`.
- `smoke.py`: 320×240 output, 80×60 internal rendering, eight camera-motion
  frames, default composition versus an inserted application command.
- Bilinear, OrdinaryShade EASU and FSR 2 produced identical internal HDR between
  the default and custom compositions. Each custom callback ran eight times.
- Removing the builder and presenting another frame succeeded in all three modes.

Reproduce the insertion/reset check from the repository root:

```sh
PYTHONPATH=. .venv/bin/python artifacts/denoiser-motion/gi-composition/smoke.py
```

These checks validate behavior and recording integration, not independent stage
allocation or arbitrary reordering of temporal operations. No performance claim
is made; custom builders currently disable command caching.

## Independent spatial stage

`spatial_compare.py` copies the viewer's temporal lobe outputs and guides into
independent runtime-owned images, then executes `VulkanRelaxSpatial` through a
`VulkanGraph`. It compares the resulting HDR to the native viewer's filtered HDR.
At 161×121, six camera-motion frames for each iteration count from 1 to 5 all
matched exactly (`spatial-metrics.json`). The viewer remains the temporal producer
in this comparison; separate GPU unit tests initialize all inputs without a viewer.

```sh
PYTHONPATH=. .venv/bin/python artifacts/denoiser-motion/gi-composition/spatial_compare.py
ORDINARYLIGHT_TEST_VULKAN_GRAPH=1 .venv/bin/python -m pytest -q tests/test_relax_spatial.py
```

## Independent temporal history

`temporal_compare.py` maintains its own history ring, copies native current
signals/guides, and executes independent temporal and spatial operations in one
Vulkan graph. The comparison uses one spatial iteration so native raw signals
remain available after rendering. Native and independent temporal radiance,
history lengths and final HDR were identical across eight camera-motion and eight
glass-motion frames at 161×121 (`temporal-metrics.json`).

```sh
PYTHONPATH=. .venv/bin/python artifacts/denoiser-motion/gi-composition/temporal_compare.py
ORDINARYLIGHT_TEST_VULKAN_GRAPH=1 .venv/bin/python -m pytest -q tests/test_relax_temporal.py
```

## Native migration

The viewer now uses the temporal/spatial graph components with borrowed native
images, scratch and uniforms. It no longer creates the executor-owned temporal
or spatial pipelines. `migration-metrics.json` records exact display parity for
six 24-frame EASU sequences against `/tmp/easu-shade`; migrated captures are in
`/tmp/gi-denoiser-migration`.

`viewer_lifecycle.py` verifies cache hits, active 100/50/25% extents without
recreating denoiser bindings, resize in both directions, and close. EASU and FSR 2
passed (`viewer-lifecycle.json`). FSR 2 continues to bypass command caching.

```sh
PYTHONPATH=. .venv/bin/python artifacts/denoiser-motion/gi-composition/viewer_lifecycle.py
```

The external recording contract has CPU tests in
`tests/test_vulkan_graph_recording.py`. The combined Vulkan graph/denoiser GPU
regression run passed 15 tests. These are correctness/lifecycle checks; no new
performance claim is made.

## Reconstruction migration

`VulkanReconstruction` now owns the native viewer's reconstruction pipeline and
bindings. The executor no longer creates reconstruction descriptors/pipelines,
and FSR 2 no longer rewrites those private descriptors. Native presentation owns
swapchain transitions around cached array-selected output commands.

The final shader skips disabled history writes and inactive material/effect
reads. All six 24-frame display sequences still matched the original captures
exactly (`reconstruction-metrics.json`; raw files in
`/tmp/gi-reconstruction-migration`). Native cache, extent, resize and cleanup checks
passed (`reconstruction-lifecycle.json`).

Standalone GPU tests cover bilinear, cubic, both EASU implementations, selected
array output, enabled/disabled history writes, and packed tint parameters.
The final combined graph/denoiser/reconstruction regression suite passed 24 tests.

```sh
ORDINARYLIGHT_TEST_VULKAN_GRAPH=1 .venv/bin/python -m pytest -q tests/test_reconstruction.py tests/test_vulkan_graph.py tests/test_relax_temporal.py tests/test_relax_spatial.py tests/test_vulkan_graph_recording.py
```

The FSR 2 temporal upscaler remains a separate native bridge awaiting independent
resource-binding extraction; the reconstruction component consumes its HDR output.


FSR2 graph extraction: `fsr2-metrics.json` compares the existing
`fsr2-upscale/capture.py` sequences against `/tmp/fsr2-upscale`. All nine
24-frame sequences (bilinear, FSR1 and FSR2 across camera/glass/target motion)
matched byte-for-byte. Reproduce with:

```sh
PYTHONPATH=. OUT=/tmp/gi-fsr2-graph .venv/bin/python artifacts/denoiser-motion/fsr2-upscale/capture.py
```

`fsr2-lifecycle.json` records the native lifecycle checks after migration to
`VulkanFsr2`: EASU cache reuse and FSR2 cache bypass, active scale changes, resize
and disposal. `tests/test_fsr2_graph.py` checks the independent runtime component
without a viewer, including odd extents, prepared depth/motion/reactive values,
a two-slot temporal stream, reset, output ownership and replay rejection.


`ray_generation_smoke.py` now enables split secondary intersection and checks both
`VulkanRayGeneration` and `VulkanIntersection` through the native tile API. Glass
and opaque scenes pass repeated tracing interleaved with presentation and disposal.
Analytic GPU tests are in `tests/test_ray_generation_graph.py` and
`tests/test_intersection_graph.py`, including a combined dependency-ordered graph.

### Resolve recording performance comparison

`resolve_performance.py` compares the pinned pre-migration native resolve method
with the current shared-operation adapter. Run from the repository root with
`PYTHONPATH=. .venv/bin/python artifacts/denoiser-motion/gi-composition/resolve_performance.py`;
add `--force-recording` to invalidate native command keys each frame.

Results are in `resolve_performance.json`: 1280x720 output, 50% GI scale,
FSR1-Shader, glass-detail fixture, 12 warmup + 30 measured frames per run,
legacy/shared/shared/legacy order. Queue-idle waits measure serialized frame
latency, not pipelined presentation throughput. GPU timestamps are native frame
measurements. Runs were sequential on a shared desktop without clock isolation.

Pooled static cached medians: GPU 10.187 ms legacy, 10.186 ms shared. Resolve CPU
recording is zero on cache hits. Forced moving-frame recording medians: resolve
CPU 0.468 vs 1.075 ms across eight calls/frame; GPU 6.506 vs 6.389 ms; synchronized
wall time 42.383 vs 41.779 ms. Overall latency differences are within run-to-run
variation; the repeatable finding is approximately 0.607 ms extra CPU recording
per rebuilt frame (0.076 ms/call). Cache reuse hides that overhead normally.

These measurements compare the resolve migration only, not a feature-equivalent
standalone replacement for the full native GI pipeline. Caching prepared resolve
bindings/recordings is the next optimization to investigate.
