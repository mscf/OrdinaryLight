# Moving glass preparation cost

640×480 hidden GLFW swapchain, glass-detail fixture, 8 warm-up frames,
32 glass translations then 32 stationary frames. Standard custom-material
wavefront GI, four ReSTIR streams and transmission history cap enabled.
Capture occurs after each phase, outside frame timing. Median excludes the
first two frames of each phase. This is submission wall time with swapchain
pacing, not isolated GPU duration or a desktop-viewer FPS guarantee.

Unprofiled moving median: 48.03 ms before, 33.78 ms after (30% reduction).
Stationary medians varied from 13.72 to 18.78 ms; desktop/GPU scheduling is
noisy, including occasional long stalls. Do not interpret this as a precise
universal speedup. Profiling adds substantial overhead: the initial moving
profile measured 76.69 ms. In that profile emissive count/weight calculations
cost 0.919 seconds over 32 frames, versus 0.080 seconds after caching.

Changes cache emissive scalar statistics by scene revision, skip invariant
material/light/texture-binding uploads for transform-only edits, and reuse
packed area lights and the unchanged material signature during signature
updates. World-space attributes, custom attributes, area lights, current and
previous geometry, TLAS rebuilding and synchronization remain intact. No
shader, sampling, denoiser-history or command-cache policy was changed here.
Additional scene_partial_prepare_ms, scene_partial_wait_ms, and
scene_partial_buffers_ms counters expose preparation costs in last_timings.

Moving-end and settled HDR captures were byte-identical before/after.
The remaining cost includes command re-recording and synchronous resource
updates. This is a first optimization pass, not a claim of restored 60 FPS.

Reproduce from repository root (requires native Vulkan/display):

```sh
BASELINE=1 PROFILE=0 OUT=/tmp/glass-before PYTHONPATH=. .venv/bin/python artifacts/denoiser-motion/glass-performance/capture.py
PROFILE=0 OUT=/tmp/glass-after PYTHONPATH=. .venv/bin/python artifacts/denoiser-motion/glass-performance/capture.py
```

Set PROFILE=1 for cProfile output. BASELINE loads only the previous update,
signature and emissive-statistic methods from the pinned Git commit; it does
not change working files or undo the separate stock-transmission changes.
