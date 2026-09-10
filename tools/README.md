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

### GI viewer defaults

Selecting Wavefront GI uses shared-primary ReSTIR, two direct-light reservoirs,
two path samples per pixel, and 524,288-ray batches, with native resolution and
ReLAX enabled. This is the measured high-resolution Rough reflections preset,
not a guarantee of optimal performance for every scene. Two path samples reduce
indirect noise; `--path-spp 1` is faster and enables temporal reservoir reuse.
Shared-primary GI requires `glslangValidator` or `glslc` on PATH.

Controls retain your adjustments when switching targets. Startup overrides are
`--restir-reservoirs`, `--path-spp`, `--ray-batch-capacity`, and
`--no-shared-primary-restir` (which defaults to one path sample). These also
apply to `--readback`. Apply/restart after adjusting live GI controls.

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

### Experimental low-bounce GI

Select **GI — 1 sample / 4 bounces (experimental)** in the rendering target
menu, or run `python tools/raster_feature_viewer.py --target wavefront-gi-fast`.
This mode fixes shared-primary ReSTIR on, path samples to one, and the renderer's
maximum bounce count to four. It uses the existing OrdinaryShade wavefront
shaders, including ray-traced primary visibility. Rasterized primary visibility
and an indirect-light cache are not implemented by this experiment.

Resolution, reservoirs, denoising, and ray batch controls remain available.
Normal GI sample/shared-primary settings are restored when switching back.
The mode is also available with `--readback`; its fixed sample/bounce budget
wins over the path-sample and shared-primary startup settings.

Compare with **Wavefront GI** at the same camera and resolution, with animation
disabled. Let the denoiser settle after switching, then compare total GPU and
`intersect_shade` timings. Expect more sampling noise and missing deeper
illumination/reflection/transmission paths, especially indoors and through glass.
The four-bounce value is the renderer's existing path-depth limit, not four
extra indirect bounces after the primary surface. The terminal secondary hit
contributes emission but stops before direct-light evaluation: the original
two-bounce preset could therefore leave reflected non-emissive surfaces black.
Three permits lighting at the first reflected surface; the current four-bounce
preset adds one more level for reflected illumination. Repeated reflections and
glass paths can still need more depth. No speedup is guaranteed.

### Resize and shutdown diagnosis

Use `--profile-lifecycle` to distinguish swapchain recreation from renderer and
Vulkan device teardown. `swapchain` reports waiting, destruction, surface setup,
and allocation separately. `vulkan_runtime_close` now reports `wait_idle_ms`,
`pipeline_cache_ms`, `pools_ms`, `destroy_device_ms`, and `destroy_instance_ms`.
Scene/settings/target changes that retire a presenter also destroy its owned
runtime, so driver device-destruction latency affects those transitions.

A local RTX 5090 profile of Rough reflections, windowed 1026 × 882 and fullscreen
3366 × 2382, measured 53–130 ms swapchain recreation and a 217 ms first fullscreen
frame in standard GI. Shutdown took 2.27 seconds, including 2.15 seconds inside
`vkDestroyDevice`; fast GI showed 3.09 seconds overall with 3.00 seconds there.
These runs did not reproduce multi-second fullscreen stalls. The 150 ms resize
settling interval and subsequent frame recording are additional to swapchain
recreation. Driver timings vary between runs.

Closing now hides the viewer and FPS overlay immediately while its existing
worker completes orderly teardown. The native surface stays alive until cleanup
finishes; process exit still waits for the driver. A post-change live run completed
cleanup normally (2.38 seconds overall, 2.27 seconds device destruction). This is
an improvement to visible close response, not a reduction in driver teardown time.

The viewer also logs `fullscreen_requested` (settings, DPI, pending work and
completed-frame count) and `fullscreen_frame_ready` (first completed frame at a
changed native extent). This excludes an old-size frame already in flight.

For a repeatable test through the actual entry point and F11 key handler, run
from the repository root:

```
python components/OrdinaryLight/tools/diagnostics/fullscreen.py \
  --target wavefront-gi-fast --showcase optical-screen-rough-reflection \
  --report /tmp/ordinarylight-fullscreen-report.json
```

The report includes the Python executable, debugger/monitoring state, Qt backend,
selected launch environment variables, and lifecycle events. Use the same command
in an integrated terminal and a standalone terminal to compare environments.
`--first-toggle-seconds 30` tests established rendering. Add `--cold-cache
--first-toggle-seconds 3 --cycles 1` to exercise F11 while cold GI pipelines are
still compiling. Cold mode creates its own temporary cache directory and leaves
existing caches untouched; its path is recorded in the report.

Two cold-start standard-GI runs reproduced 22.15 and 22.97 seconds from F11 to
the fullscreen frame, with zero completed GI frames when F11 was pressed.
Shared-primary initialization was compiling unused generic primary/shade
pipelines before its scene-specialized pair. Skipping that generic pair reduced
the same cold-start test to 7.61 seconds. Necessary driver compilation remains.
This reproduces a startup-specific delay, not a demonstrated steady-state stall.
The fast GI preset subsequently completed four warm F11 transitions in
0.31–0.45 seconds on the local system.

Raster resize recovery now treats out-of-date/suboptimal acquire/present results
as a request to rebuild on the next frame. A suboptimal acquired image is still
submitted so its acquisition semaphore is consumed. Both cached and uncached
presentation paths complete their submission bookkeeping before recreation.
The previously reproducible first-F11 raster exception no longer occurred over
eight live switches. GI and legacy ray-query acquisition recovery also return to
the caller for a current extent instead of recursively rebuilding with stale
sizes; deterministic repeated-error tests cover that bounded recovery.

### Raster-to-GI fullscreen regression

The missing reproduction sequence was **start in raster → select GI → F11**,
not direct GI startup. In the fast GI preset, four successive F11 transitions
measured 22.48, 22.33, 22.46, and 22.32 seconds; shutdown took 33.36 seconds.
Swapchain destruction alone took approximately 22 seconds per transition.
This occurred with thousands of completed frames and a warm shader cache.

Replacing only the Vulkan surface or native X11 window did not resolve it.
Releasing the previous Vulkan instance and creating a fresh instance/surface
before constructing the next Vulkan renderer did. The Qt window object, scene,
camera and controls are retained. All consumers of the old instance must be
closed before this handoff. The same reset applies after an intervening WebGPU
mode, so earlier Vulkan presentation state cannot be carried through that path.

Validation covered six F11 transitions after raster → fast GI and four after
raster → standard GI, all below one second; subsequent GI shutdown took
0.16–0.18 seconds. This is separate from the cold-start compilation issue above.
Reproduce this sequence without manually operating the menu:

```
python components/OrdinaryLight/tools/diagnostics/fullscreen.py \
  --switch-to wavefront-gi-fast --showcase optical-screen-rough-reflection \
  --first-toggle-seconds 15 --cycles 2 \
  --report /tmp/ordinarylight-raster-to-gi-fullscreen.json
```

The initial mode defaults to raster. `--switch-to` changes the actual viewer
selector after 30 completed frames; the diagnostic then sends actual F11 key
events. Use `--switch-to wavefront-gi` to cover standard GI.

### Primary-stage cost and volume-free specialization

`python components/OrdinaryLight/tools/diagnostics/primary_costs.py` compares the
four-bounce, one-SPP GI preset at a fixed Rough reflections camera, 2052 × 1764,
with 32 warmup and 64 measured frames per case. It saves per-stage timings and
HDR images under `/tmp/ordinarylight-primary-costs`. Each case owns a fresh
window/presenter. `--cases` selects experiments. The baseline now uses current
production settings; pass `--no-primary-specialization --primary-variant surface-only`
to reproduce the original control below, retaining secondary specialization.

The fused `primary` stage includes camera-ray intersection, materials/textures,
ReSTIR and light visibility, guides, and continuation setup. Initial measurements:

| Experiment | Primary ms | Total GPU ms | Interpretation |
|---|---:|---:|---|
| Repeated baseline | 5.18 | 16.10 | Two reservoirs, four candidates |
| Depth-one diagnostic | 1.82 | 7.81 | Omits lighting AND continuation; not equivalent quality |
| One candidate | 4.69 | 15.63 | Changes lighting sampling |
| One reservoir | 4.46 | 14.49 | Changes lighting sampling |
| Native textures | 4.65 | 15.17 | Three-texture scene; total-time gain was not decisive |

The depth-one result is a hit/material/guide floor, not a standalone intersection
measurement. Its difference from baseline includes BSDF continuation and queue
work as well as lighting. These measurements do not establish a bandwidth versus
register-pressure bottleneck. HDR differences from changed sampling are not a
quality verdict, so reservoir, candidate and native-texture defaults were retained.

A separate repeated comparison tested removing volume integration and volume
shadow attenuation from primary shaders when `scene.visible_volumes` is empty:

| Case | Primary ms | Total GPU ms |
|---|---:|---:|
| Baseline repeat | 5.24 | 16.17 |
| Specialized repeat | 3.91 | 15.03 |
| Baseline after experiment | 5.28 | 16.25 |

The specialized repeat's final HDR matched the baseline exactly. Its first run
had RMSE 0.00116 versus baseline; the report retains both measurements rather
than assuming deterministic first-use output. The repeated comparison represents
about 25% less primary time and 7% less total GPU time for this pose, not a general
FPS guarantee.

The specialization is authored in typed OrdinaryShade and is enabled by default
for scene-compiled primary pipelines. Visible-volume changes invalidate pipeline
selection and restore volume processing. Set
`wavefront_primary_scene_specialization=False` in RendererConfig for a control
run; `wavefront_scene_specialization=False` disables the broader scene
specializations as well. No shader estimator or sampling budget is changed.

Validation rebuilt all 458 manifest shader outputs and passed 100 focused tests.
The multiple-scattering volume check compiled identical primary/secondary shader
bytes with the option on or off, confirming the volume-free path was not selected.
A repeated 160 × 90 GPU run also matched both control HDR images exactly. An
initial run showed a 0.01367 maximum difference between its two controls and the
same difference against the enabled case; this was checked with the shader-byte
comparison and repeat run rather than attributed to the optimization.

The follow-up investigation covers candidate coefficient reuse, texture/material
guards, register demand, workgroup size, visibility rays, and further scene
specialization. See [the measured results](diagnostics/primary_optimization_results.md).
Use `--primary-variant surface-only` for the pre-follow-up baseline, `opaque` to
add opaque material specialization alone, and `current` (default) for both retained
specializations. `--reference path.npy` compares another run's HDR; match scene,
extent, GI mode, warmup and frame counts. `--gi-mode full` tests the normal GI preset.
`--pipeline-statistics` captures driver executable statistics in a separate run;
`--primary-workgroup-rows 4` tests an 8×4 workgroup without changing the viewer.
`--no-primary-brdf-cache` disables only the primary BRDF preparation reuse for
a matched comparison with the retained scene and estimator specializations.

### Experimental GI animation and recording spikes

The four-bounce GI preset uses one path sample and active temporal ReSTIR reuse.
Camera motion changes its history limit. Previously this limit (and history
validity) participated in the command-cache key, repeatedly recording all GI
passes. Normal GI's two-SPP default does not use the same temporal reservoir
history, which masked the issue in that mode.

Shared-primary now uploads the exact per-frame history validity and limit in a
16-byte suffix to its per-slot camera buffer. The existing 64-byte camera prefix
is unchanged, and each slot's fence protects the upload. Only the shared-primary
shader variant reads the suffix; offline uploads without policy fall back to push
constants. Legacy command-policy consumers keep their original cache keys.
History limits, rejection decisions, sampling counts, and shading are unchanged.

A 45-second test through the actual `raster_feature_viewer`, starting in raster,
switching to experimental GI after 30 frames, with Rough reflections animation
and the default window size, measured these results after 50 warmup frames:

| Metric | Before | After |
|---|---:|---:|
| Record median | 5.216 ms | 0.078 ms |
| Record p95 | 6.094 ms | 0.128 ms |
| Record maximum | 39.504 ms | 0.895 ms |
| Command-cache hit rate | 30.06% | 100% |

A separate deterministic 960×540 comparison forced the original push-constant
shader to re-record and compared it with uploaded policy while alternating
camera jumps and stops. Final HDR images matched exactly; the uploaded-policy
path reused commands on every measured frame. Focused validation passed 118
tests and eight subtests, including camera snapshot isolation and compilation
with/without the extended camera policy.

To collect animated viewer timings (without HDR readback or a diagnostic FPS cap):

```bash
python components/OrdinaryLight/tools/diagnostics/record_stutter.py \
  --showcase optical-screen-rough-reflection --seconds 45 \
  --report /tmp/ordinarylight-record-stutter.json
```

The report includes individual frames, record percentiles, cache hits, per-frame
history policy, resolution, and garbage-collection intervals. The option
`--target wavefront-gi-fast` skips the initial raster-to-GI switch. To reproduce the image
comparison:

```bash
python components/OrdinaryLight/tools/diagnostics/command_recording.py \
  --spp 1 --fast --uploaded-policy --history-resets
```
