# Developer tools

See [the shared-primary ReSTIR experiment](diagnostics/shared_restir.md) for
independent path-SPP/reservoir controls, measured results, and GPU comparisons.

These modules exercise implementation details and are not public examples.

Run the interactive Vulkan presentation harness with:

```bash
python -m tools.wavefront_present
```

Run a diagnostic by module name, for example:

```bash
python -m tools.diagnostics.wavefront
```

To investigate slow GI startup, settings changes, or fullscreen transitions,
run the raster workbench from this component directory with lifecycle logging:

```bash
python tools/raster_feature_viewer.py --target wavefront-gi \
  --showcase optical-screen-rough-reflection --profile-lifecycle \
  2> /tmp/ordinarylight-lifecycle.log
```

Press F11 to enter/leave fullscreen, or apply a settings change. JSON records
in the log separate renderer construction/retirement, surface recreation,
swapchain waits/destruction/allocation, and uncached or slow GI frames.
`wavefront_command_record_ms` measures CPU command recording;
`gpu_frame_ms` is a delayed GPU timestamp and can describe an earlier frame.
These diagnostics do not enable HDR readback or the 2 FPS diagnostic mode.

`shader_compile` records external compiler invocations with a source hash;
in-process shader-cache hits do not invoke the compiler. `compute_pipeline_create`
times Vulkan compute-pipeline creation, and `gi_command_finalize` measures the
driver's command-buffer finalization separately. `gi_command_cache_miss` lists
changed cache-key fields and whether FSR 2 or a custom pipeline builder requires
recording regardless of the key. An `invalidated` key can also be a frame slot's
first recording, not necessarily a repeated invalidation. Since fast cached
frames are omitted, the log's miss count is not a cache-miss percentage.

Window sizes must settle for 150 ms before the viewer submits a frame at a
new extent, avoiding GPU resource recreation for intermediate resize events.

To compare CPU recording costs with the previous cache-key behavior:

```bash
python -m tools.diagnostics.command_recording
python -m tools.diagnostics.command_recording --spp 1 \
  --output /tmp/command-recording-spp1.json
```

This uses Rough reflections, shared ReSTIR with two reservoirs, denoising,
960 × 540 output, 32 stationary warm-up frames and 96 frames of deterministic
camera motion. Both cases use per-frame previous-camera snapshots, preventing
the host from overwriting a camera still being read by an in-flight submission.
Only cache-key handling differs: inactive history limits no longer invalidate
recorded commands. The diagnostic checks finite HDR and exact final-image parity.
Readback happens after measurement; reported render/present call time excludes
window event processing and is not the Qt viewer's FPS.

On the RTX 5090 Laptop GPU, the 2-SPP run measured median host recording at
24.68 ms before and 0.31 ms after, with cache hits increasing from 14.6% to 100%.
Mean render/present call time fell from 22.78 ms to 6.54 ms; median GPU time
remained approximately 6.74 ms. At 1 SPP, active temporal history still changed
the recorded constants: mean call time fell from 12.27 ms to 10.29 ms, while
median recording remained approximately 13.2 ms. Both final HDR images matched
their respective baselines exactly. FSR 2, custom pipeline builders, active
history-policy changes, and other real command changes can still require recording.

The Qt animation path also exposed redundant invalidation on ReLAX history
validity. That flag is uploaded through the per-frame policy buffer, so changes
now reuse commands while preserving history resets. At 2052 × 1764, a 160-frame
Qt viewer run with the same shared 2-SPP settings measured recording median/max
of 169.02/197.73 ms before and 0.14/0.43 ms after, excluding the first ten frames.
To regression-test history resets against forced command recording:

```bash
python -m tools.diagnostics.command_recording --history-resets
```

This alternates camera jumps and stops, checks exact final HDR parity, and
requires more than 90% command-cache hits in the optimized case. Restart the
Python viewer process to load code changes; Apply/restart only rebuilds its renderer.

### GI ray batches

The direct GI viewer exposes **GI ray batch size** (131,072, 524,288, or
1,048,576 rays). Larger batches use more GPU memory and reduce the number of
dispatches and synchronization barriers at high resolutions. They preserve
SPP, resolution, lighting samples, and the bounce limit. Apply/restart after
changing this setting, or launch with `--ray-batch-capacity 524288`.

On the RTX 5090 Laptop GPU, Rough reflections at a fixed camera pose and
2052 × 1764 output, shared ReSTIR (two reservoirs, two path SPP), eight bounces,
and ReLAX produced the following medians over 32 frames after 32 warm-up frames:

| Ray batch capacity | Batches/frame | Intersect + shade ms | Total GPU ms |
|---:|---:|---:|---:|
| 131,072 | 56 | 36.47 | 51.70 |
| 524,288 | 14 | 24.64 | 40.84 |
| 1,048,576 | 8 | 23.46 | 39.73 |

Both larger batches matched the repeated baseline's final HDR image exactly.
The first baseline run differed slightly from its identical-settings repeat;
the comparison therefore includes both runs rather than assuming determinism
across initial renderer setup. This is one scene and camera pose, not a general
quality proof or a guaranteed speedup at other resolutions. A separate split
intersection/shading run was slower (55.63 ms total GPU time).

Reproduce from this component directory with
`python -m tools.diagnostics.secondary_batches`. The diagnostic saves per-bounce
timings, final HDR arrays, and compares the larger batches with the repeated
baseline. A 160-frame live Qt run also verified the new 524,288-ray control;
median recording remained 0.15 ms after warm-up.

### Volume-free secondary shading

Runtime-compiled Ordinary Shade secondary kernels now specialize away volume
integration and volume shadow attenuation when the scene has no visible volumes.
This applies to shared-primary ReSTIR and existing custom-material compilation;
the packaged shader path retains its existing behavior. The material pipeline
signature includes this choice, so adding a volume restores the full shader.
Resolution, sampling, bounce limits, and surface lighting remain unchanged.
The specialization's control flow is authored in the typed OrdinaryShade
functions in `scripts/generate_core_shaders.py`. The generator exposes its
compile-time `WAVE_SURFACE_ONLY` switch alongside the existing volume switches;
the runtime compiler selects that flag without rewriting GLSL function bodies.

At the same full-resolution Rough scene pose and 524,288-ray batches, the
repeated baseline measured 24.39 ms for `intersect_shade` and 40.57 ms total GPU.
Automatic surface-only shading measured 10.82 ms and 27.62 ms respectively.
Final HDR output matched the repeated baseline exactly. These are fixed-pose
GPU measurements, not guaranteed viewer FPS. Adding/removing a visible volume
was also tested with a live renderer, including shader reselection and finite HDR.

Reproduce with `python -m tools.diagnostics.secondary_surface`. Set
`wavefront_scene_specialization=False` through RendererConfig for a control run.
Restart the viewer process to load the optimization; no new UI setting is needed.

Other measured candidates were not enabled: material bucketing was slower;
Russian roulette saved roughly 3% total GPU time in this scene while changing
sampling; the handwritten shader was faster but changed mean HDR brightness by
about 21%. Skipping unused primary-specular bookkeeping had no measurable benefit.
