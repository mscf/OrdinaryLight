# Transform-only command and staging reuse

Follow-up to glass-performance. Same 640×480 fixture, 32 moving then 32
stationary frames, eight warm-up frames, history cap and four ReSTIR streams
on. Unprofiled native submission wall times, medians excluding the first two
frames per phase. GPU/display scheduling makes these indicative measurements,
not guaranteed viewer FPS or isolated GPU durations.

| Configuration | Moving median | Tile recording mean/frame | Upload mean/frame |
| --- | ---: | ---: | ---: |
| Before (75c5e181) | 34.51 ms | 6.20 ms | 7.15 ms |
| Command reuse | 32.07 ms | 0.30 ms | 7.30 ms |
| Command + staging reuse | 25.77 ms | 0.29 ms | 1.47 ms |

30/32 moving frames reused commands, versus 1/32 before. Both variants'
moving-end and settled HDR captures were byte-identical to the baseline.
TLAS rebuild remained about 1.3 ms/frame in the baseline and final runs.

Transform-only updates retain the same buffers and descriptors. Commands can
therefore remain cached, but their key now includes light counts and total
emissive weight (baked push constants), plus scene revision when object effects
bake projected bounds. History resets are unchanged. Geometry/shading edits,
resource replacement and swapchain teardown still invalidate commands.

Synchronous scene uploads use a single growable staging slab, retained in the
uploader's normal buffer ownership list. Each call waits for its copy before
returning, so the next call can safely overwrite the slab. No asynchronous
upload or GPU-wait removal is attempted. Growth releases the previous slab;
normal teardown releases the retained allocation.

Additional validation: 600 native frames with forced swapchain recreation at
moving frames 10, 60, 120 and 240, without device loss. Translating and scaling
an emissive bar yielded byte-identical HDR with reuse versus forced recording.
Unit coverage checks slab reuse, offsets, fresh payloads, growth and size errors;
existing swapchain teardown regression checks invalidation before destruction.

Run from the repository root with a Vulkan/display-capable environment:

```sh
PROFILE=0 OUT=/tmp/reuse PYTHONPATH=. .venv/bin/python artifacts/denoiser-motion/glass-command-reuse/capture.py
FORCE_RECORD=1 SCALE_LIGHT=1 SUBJECT=target PROFILE=0 OUT=/tmp/lights PYTHONPATH=. .venv/bin/python artifacts/denoiser-motion/glass-command-reuse/capture.py
FRAMES=300 RECREATE=1 PROFILE=0 OUT=/tmp/stress PYTHONPATH=. .venv/bin/python artifacts/denoiser-motion/glass-command-reuse/capture.py
```

`wavefront_command_record_ms` in renderer timings measures secondary-command
recording/cache lookup separately from GPU stage timestamps. Captures and raw
stress outputs from this run are under /tmp/motion-next-*.

The actual Qt raster_feature_viewer also completed 600 animated frames with
the checkbox enabled and renderer restarted; no presentation failure. Its
final status is saved in viewer.log. The displayed extent is the requested
extent, and the final FPS sample is not a controlled before/after benchmark.
