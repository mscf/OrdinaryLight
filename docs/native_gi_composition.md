# Native GI composition

## Development requirement

New rendering features and changes should be implemented as independently
composable stages with explicit inputs, outputs, resource ownership and graph
dependencies. Keep preparation separate from GPU allocation and native viewer
recording. Integrate the same stage into native execution rather than adding a
second viewer-only implementation. Validate independent composition as well as
any affected native path. This is the requested default for future work.

The native viewer records its GI work through `RenderPipeline` and `RenderStage`
in `ordinarylight.pipeline.gi`. The default stages, in order, are:

| Stage | Work | Logical outputs |
| --- | --- | --- |
| `gi.trace` | Tiled transport, ReSTIR, signal preparation | HDR, guides, reservoirs |
| `gi.indirect` (optional) | Indirect reservoir candidates/application | HDR, reservoirs |
| `gi.denoise` (optional) | ReLAX temporal accumulation, spatial filtering, composition | HDR, denoiser history |
| `gi.upscale` (optional) | FSR 2 preparation and temporal upscaling | Upscaled HDR |
| `gi.reconstruct` | Spatial upscaling when selected, tone mapping, display encoding, object effects | Display, reconstruction history |

`create_gi_pipeline()` accepts a prepared recorder for each stage and returns a
validated ordered pipeline. Recorders receive `context['frame']`, a `GiFrame`.
`RenderStage.reads` and `.writes` describe logical dependencies; these names are
not Vulkan allocation handles and do not automatically generate barriers.

## Integrating an application stage

Both `VulkanGlfwPresenter` and `VulkanSurfacePresenter` expose
`set_gi_pipeline_builder(builder, reuse_commands=False)`. The builder receives the prepared default
pipeline and the current frame, and returns a `RenderPipeline`. For example,
an application can insert a compute effect between denoising and display
conversion without copying HDR to the CPU:

```python
from ordinarylight.pipeline import RenderStage


def compose(default, frame):
    def record(context):
        current = context['frame']
        # Application code binds its own descriptors and records the required
        # read/write barriers and dispatch. Existing images stay in GENERAL.
        effect.record(
            command=current.command,
            image=current.images['hdr'].image,
            image_view=current.images['hdr'].view,
            extent=current.render_extent,
        )

    return frame.process_hdr(default, RenderStage(
        'application.hdr_effect',
        reads={'gi.hdr'}, writes={'gi.hdr'}, recorder=record,
    ))


presenter.set_gi_pipeline_builder(compose)
# Render normally through presenter.present_wavefront(...).
# Restore the standard composition:
# presenter.set_gi_pipeline_builder(None)
```

`frame.process_hdr()` inserts after denoising and before upscaling or display
conversion, including when FSR 2 is selected.

The supplied stages can be assembled into a new `RenderPipeline` alongside
application stages. Keep the default stages required by the active renderer
configuration and preserve their dependency order: the current backend commits
history for that configuration after submission. This hook does not yet support
arbitrary removal/replacement of temporal stages with different history policies.

## Ownership, synchronization and lifetime

- `GiFrame.command` is an already recording secondary command buffer on the
  presenter's device. Record commands only; do not begin/end/submit it.
- `resources` and `previous_resources` are read-only snapshots of backend frame
  bindings. The native objects remain owned by the presenter. Do not destroy,
  rebind, retain across frames, or write previous-frame resources.
- `images` provides named `GiImage` views compatible with `VulkanResource.image`.
  The old `resources` mapping remains for compatibility; new consumers should
  use named images. `require_open()` rejects retired image allocations.
- `render_extent` and `output_extent` distinguish the active internal dimensions
  from the display dimensions. Allocation dimensions can differ.
- Custom stages must supply their own Vulkan synchronization and restore borrowed
  image layouts. Logical dependency validation alone is not synchronization.
- History flags describe whether previous-frame inputs may be reused. GPU ordering
  and retirement remain the presenter's responsibility.
- Set/change the builder only between presentation calls, without concurrent
  rendering. Changing it invalidates cached commands and temporal validity.
- Custom builders default to recording every frame. With `reuse_commands=True`,
  the renderer reuses each slot's commands. Callbacks then run on recording,
  **not on every frame**. Store changing inputs in persistent GPU resources.
  Call `invalidate_gi_commands()` after changing embedded constants or bindings.
  Changing buffer contents alone does not require invalidating commands.

### Named images and application graph submission

`frame.images['hdr']` contains linear scene radiance. Before `gi.denoise` it is
raw transport; afterward it is denoised, before exposure, tone mapping and display
encoding. Denoising overwrites this image. To retain raw radiance simultaneously,
enable `RendererConfig(wavefront_raw_hdr_output=True)` at construction. This adds
`gi.snapshot_hdr` after transport/indirect application and before denoising. Its
`frame.images['raw_hdr']` is a distinct persistent RGBA16F image per frame slot.
It costs 8 bytes per allocated pixel per slot and one GPU image copy per frame;
there is no automatic host readback. The default has no snapshot cost.

Other image keys expose native diagnostics: `diffuse`, `specular`,
`normal_roughness`, `view_z`, `motion`, `identity`, `temporal_diffuse`,
`temporal_specular`, `atrous_diffuse`, `atrous_specular`, `diffuse_history`, and
`specular_history`. Their contents are meaningful only after the producing stage
and when the corresponding denoiser configuration is enabled. `ray_distance`,
`packed_normal_class`, and `material_signature` expose transport guides.
**These are not the requested general application primary-hit ABI:** identity
is the existing denoiser identity, and material signature is not object identity.
They do not provide independent instance/slot/face IDs.

Image `width` and `height` describe actual allocation dimensions, which can be
1×1 for disabled guides. The active image rectangle is `[0, render_width) ×
[0, render_height)`, from `frame.render_extent`, within enabled full-size images.
Do not dispatch over allocation padding. A GI pixel `(x,y)` covers normalized
coordinates `[x/W,(x+1)/W) × [y/H,(y+1)/H)` before upscaling. Output pixel centers
map to GI coordinates by `(output_xy + 0.5) * render_extent / output_extent - 0.5`.
Reconstruction filters neighboring radiance; there is no one-to-one primary hit
per upscaled output pixel. Existing guides also do not represent every sampled
camera ray when SPP exceeds one. No retraced or reconstructed guide should be
treated as the requested exact sampled-ray identity export.

An application can record a compiled `VulkanGraph` in an inserted stage using
`context['frame'].record_graph(compiled_graph)`. Prepare persistent kernels and
bindings for each slot; graph passes declare their resource accesses and record
GPU work. The native frame records graph barriers and publishes `submitted()`
callbacks only after successful submission, including cached command replay.
All prior completions must belong to the same runtime queue. Imported semaphore
operations are rejected; this is not an external-queue synchronization API.
Operation preparation must not allocate, submit, or wait. Keep borrowed GI images
in GENERAL layout. Graph resources and kernels must stay alive until submitted
work retires; invalidate native commands before replacing recorded bindings.

`invalidate_gi_history()` rejects progressive, ReSTIR, denoiser and reconstruction
history without dropping pending synchronization or reallocating images.
`invalidate_gi_commands()` preserves history. Neither method makes in-flight
buffers safe to overwrite. Use completion ordering or explicit `wait_idle()` at
replacement boundaries; it waits the runtime queue, not unrelated queues.
`use_scene_resources()` publicly binds an existing native resident **triangle**
scene from the same runtime, retaining application ownership. It currently waits
before binding replacement and does not accept arbitrary external TLAS layouts.

`reconfigure()` supports samples, bounce count, exposure, render scales and
stationary delay. Invalid changes fail before mutation. Sample-capacity/render-
scale changes retire and recreate frame allocations at the next present call.
Changes to adaptive controller bounds and other settings explicitly require
presenter recreation. Camera and output size continue to be arguments to
`present_wavefront()`; resizing retires old image views.

## Native lighting without presentation

`ordinarylight.runtime.VulkanWavefrontPipeline` accepts an application-owned
`VulkanRuntime`, an uploaded native `VulkanSceneResources`, and `RendererConfig`.
It shares native preparation, transport, denoising, recording caches and history
with `present_wavefront()`. It creates no swapchain and runs no reconstruction,
tone mapping or display encoding. FSR2 and external video interop are unsupported
in this session; downstream application operations may provide display processing.

```python
from ordinarylight.runtime import VulkanWavefrontPipeline, VulkanOutput
from ordinarylight.pipeline.graph import VulkanGraph

with runtime.upload_scene(scene) as resident:
    with VulkanWavefrontPipeline(runtime, resident, config=config) as gi:
        with VulkanOutput(runtime) as output:
            frame = gi.prepare(camera, (output_width, output_height))
            # Prepare a tone target for each persistent slot; reuse it on subsequent
            # frames with the same image allocation and active extent.
            with output.prepare(frame.images['hdr'], extent=frame.render_extent) as tone:
                graph = VulkanGraph().add('lighting', frame.operation)
                # Insert application averaging here. Declare the HDR image and
                # primary-hit buffer uses so the graph supplies dependencies.
                graph.add('tone', tone.operation())
                completion = graph.compile().execute(runtime)
                # Optional: append output.present_operation(tone) before compile.
                completion.wait()  # Explicit diagnostic/retirement boundary only.
```

`prepare()` advances no history until the operation is submitted. At most one
frame may be prepared at a time. Submit it once or call `frame.cancel()` before
changing configuration, rebinding scene resources, or preparing another frame.
Prepare a fresh operation each frame; native rendering commands are reused
internally. `render(camera, extent)` is the convenience form that submits lighting
and returns the prepared frame and its completion without waiting for readback.

Preparation may allocate or wait at frame-slot reuse or resize boundaries. Native
stage recording performs no allocations, readbacks, queue submissions or CPU waits
between transport and denoising. Application graph submission still allocates its
primary command buffer and fence. GPU consumers must be submitted on the same
runtime queue before that output slot is reused. Retire downstream consumers
before resizing or closing. Borrowed view validation rejects retired allocations;
it does not preserve old frame contents after slot reuse.

`VulkanOutput.prepare()` accepts both RGBA16F native HDR and RGBA32F application
HDR with the same display transform. Its optional `extent` selects the active
rectangle and excludes native allocation padding. Keep prepared output targets
and application kernels per slot and recreate them when their views retire.

### Exact sampled primary hits

Enable `RendererConfig(wavefront_primary_hits=True)` at pipeline construction.
The native primary camera dispatch writes `frame.buffers['primary_hits']`; it
never retraces a visibility ray. This opt-in output forces wavefront execution
when the strategy is `auto`. Other explicit execution strategies are rejected.

`ordinarylight.wavefront.PRIMARY_HIT_DTYPE` describes a 96-byte record:

| Field | Type | Meaning |
| --- | --- | --- |
| `position_distance` | float4 | World position xyz, ray distance w; w is -1 on miss |
| `geometric_normal` | float4 | World geometric normal xyz |
| `shading_normal` | float4 | World shading normal xyz, including native normal mapping |
| `identity` | uint4 | TLAS instance index, instance-local triangle, packed triangle, reserved zero |
| `ray_origin` | float4 | Actual world camera origin xyz, horizontal jitter w |
| `ray_direction` | float4 | Actual unit world direction xyz, vertical jitter w |

Miss identities are four `0xffffffff` values. If volume attenuation terminates
transport before surface evaluation, the shading normal remains the geometric
normal. Padding components otherwise carry no additional meaning.

The active records are sample-major `[frame.sample_count, render_height,
render_width]`, tightly packed at the start of the buffer. Index is
`sample * W * H + y * W + x`. Buffer byte capacity may exceed the active range.
Each SPP contributor has its own ray and hit; HDR averages their contributions.
The image/pixel mapping described above applies at GI resolution, before any
upscale filter. There is no unique hit for a reconstructed output pixel.

This currently exports native **triangle** identity. Application-defined slot and
face identity from a custom primitive still requires the custom hit ABI below;
the denoiser identity image is not a replacement for that ABI.

### Resident updates

`resident.notify_content_changed(after=(), invalidate_history=True)` publishes
in-place buffer or acceleration updates. Producers must already be submitted on
the same runtime queue; supplied completions are validated without CPU waiting.
Keep packing, capacity and compiled shader capabilities fixed. Set
`invalidate_history=False` only when previous geometry and history remain valid.
That preserves native command caches. The method increments `content_revision`
without changing `binding_revision` or allocation handles. Graph producers can
also precede lighting in the same submission through declared resource uses.

`resident.replace_resources(buffers={name: allocation}, acceleration=resource)`
replaces same-capacity native-layout bindings at an explicit queue-idle boundary.
Buffers must be same-runtime `VulkanBuffer` storage allocations. A TLAS must be a
same-runtime acceleration `VulkanResource` with an owner supporting lifetime
leases. The resident scene retains replacements; the application still owns them
and must close the scene before freeing its replacements. Replacement refreshes
native borrowers and rejects history. External kernels borrowing the resident
scene must close first. `binding_revision` changes invalidate compiled graphs;
recreate bindings and recompile their graphs. This does not translate arbitrary
custom geometry into the native triangle ABI.

### Remaining resident-wavefront integration

Custom intersections and evaluated native materials now work in primary and
fused OrdinaryShade secondary stages through `NativeGeometryProgram`. The native
GI GPU test imports the application's AABB TLAS and buffers, exports exact
primary slot/face identity, and receives indirect light from a custom emitter.

Still required: absorbing or rough boundaries,
inline/split/bucketed continuation and indirect reservoir reuse. Custom denoiser
history and local rejection now use the application-provided previous position. See
[vxl8r_native_gi.md](vxl8r_native_gi.md) for verified behavior and remaining work.

## Independent spatial denoising

`ordinarylight.runtime.VulkanRelaxSpatial` is an independently prepared Vulkan
component. It uses the same ReLAX spatial and composition SPIR-V and push-constant
packing as the native viewer, without importing the GI core or requiring a scene,
camera, window, temporal history, or presenter.

```python
from ordinarylight.runtime import VulkanRelaxSpatial
from ordinarylight.pipeline.graph import VulkanGraph

with VulkanRelaxSpatial(
    runtime,
    diffuse=diffuse, specular=specular,
    normal_roughness=normal_roughness, view_z=view_z,
    material=material, output=hdr,
    iterations=3, color_weight=4.0,
) as spatial:
    graph = VulkanGraph()
    graph.add('filter', spatial.filter_operation(after=(producer_completion,)))
    graph.add('compose', spatial.compose_operation())
    # The graph infers filter -> compose from actual image resource uses.
    graph.compile().execute(runtime).wait()
```

Alternatively, `spatial.operation()` contains both steps. Applications can consume
`spatial.filtered_diffuse` and `spatial.filtered_specular` after filtering and omit
composition. Those images belong to the stage and remain valid until it closes.

All supplied images must be distinct storage images on the same `VulkanRuntime`:

| Argument | Vulkan format | Meaning |
| --- | --- | --- |
| `diffuse`, `specular` | RGBA16F | Linear lobe radiance; alpha metadata is preserved by filtering |
| `normal_roughness` | RGBA16F | Normal in xyz and roughness in w, in the viewer's guide convention |
| `view_z` | R32F | View-space depth; zero denotes background |
| `material` | R32_UINT | Material identity used to prevent filtering across material boundaries |
| `output` | RGBA16F | Initialized HDR; background is preserved where depth is zero |

The stage borrows these images and owns four scratch images and its kernels.
Input lobe images are not overwritten. An optional `extent=(width, height)` selects
an active region no larger than any binding; otherwise output dimensions are used.
Preparation fixes bindings and settings; prepare a new stage after resize or a
binding change. The caller owns producer ordering and initialization. Graph passes
declare image accesses and layouts, so `VulkanGraph` supplies barriers. Closing the
stage waits for its submitted work, releases its kernels and scratch, and leaves
caller images open. Executing a compiled graph after stage closure is rejected.

## Independent temporal denoising

`VulkanRelaxHistory` and `VulkanRelaxTemporal` expose the temporal stage without a
presenter. Each history owns two RGBA16F temporal radiance images and two R32F
history-length images. It borrows the normal/roughness, depth, material and primitive
identity images for that frame. Keep those guides paired with their history until
all consumers finish; use a frame ring when producing new guides.

`VulkanRelaxTemporal` borrows current diffuse/specular signals, RGBA16F motion,
a previous history and an output history. Current guides come from the output
history. Previous/output histories must be distinct, with equal extents. Inputs
must have exactly the history dimensions. Motion xy is previous-minus-current in
pixels, and z is expected previous depth, matching the native signal convention.

For each ring direction, prepare a temporal stage and connect it to the spatial
stage:

```python
from ordinarylight.runtime import VulkanRelaxTemporal, VulkanRelaxSpatial
from ordinarylight.pipeline.graph import VulkanGraph

# histories[slot] borrows guides for this slot; the other history is previous.
current = histories[slot]
with VulkanRelaxTemporal(
    runtime, diffuse=diffuse, specular=specular, motion=motion,
    previous=histories[1 - slot], output=current,
) as temporal, VulkanRelaxSpatial(
    runtime, diffuse=current.diffuse, specular=current.specular,
    normal_roughness=current.guides[0], view_z=current.guides[1],
    material=current.guides[2], output=hdr,
) as spatial:
    graph = VulkanGraph()
    graph.add('temporal', temporal.operation(after=(producer_completion,)))
    graph.add('spatial', spatial.operation())
    completion = graph.compile().execute(runtime)
    completion.wait()
```

For interactive use, keep stages and compiled graphs alive instead of preparing
and closing them every frame. The temporal stage prepares immutable valid/reset
uniform buffers and selects one at execution time, avoiding per-frame uniform
uploads. Settings and bindings are fixed at preparation; recreate the stage when
they change.

History validity is committed by the graph's successful-submission callback, not
by constructing or recording an operation. Failure before submission does not
advance it. Previous-frame completions become queue dependencies automatically.
`history.reset()` invalidates reuse while preserving in-flight dependencies;
`temporal.operation(reset=True)` starts a new chain for that operation's output.
Submit successive temporal frames separately: chaining multiple frame advances
inside one submission is not supported by this history-validity contract.

The caller must reset history after discontinuities such as camera cuts or scene
replacement. Recreate histories/stages after resize. Do not overwrite guides or
history images before their consumers finish. Closing a history rejects live
kernel borrowers; close temporal/spatial consumers first. Stage close waits for
its submitted work; caller-owned signal and guide images remain open.

## Independent upscaling and display reconstruction

`ordinarylight.runtime.VulkanReconstruction` owns its pipeline and descriptors,
and borrows explicit HDR, guide, camera, history and output resources. It uses the
same generated shader as the viewer. Bilinear, clamped cubic, AMD EASU and
OrdinaryShade EASU are selected with `ReconstructionSettings.upscale_filter`.
Tone mapping, display encoding and packed object effects remain fused into this
stage to preserve the existing behavior and dispatch cost.

```python
from ordinarylight.runtime import VulkanReconstruction
from ordinarylight.pipeline.reconstruction import ReconstructionSettings
from ordinarylight.pipeline.graph import VulkanGraph

with VulkanReconstruction(
    runtime, hdr=hdr, position=position, normal=normal, material=material,
    previous_color=previous_color, previous_position=previous_position,
    previous_normal=previous_normal, history_color=history_color,
    previous_camera=previous_camera, current_camera=current_camera,
    outputs=display,
) as reconstruction:
    graph = VulkanGraph().add('display', reconstruction.operation(
        settings=ReconstructionSettings(upscale_filter='fsr1-shade'),
        after=(hdr_producer_completion,),
    ))
    graph.compile().execute(runtime).wait()
```

Keep the prepared component alive for repeated frames. It owns no images; close
waits for submitted work and releases its kernel while leaving caller resources
open. Bindings are immutable; recreate it when allocation identities change.
`extent=(width, height)` on `operation()` selects an active HDR region within the
prepared input without reallocating descriptors or pipelines.

| Binding | Format and convention |
| --- | --- |
| `hdr` | RGBA16F linear HDR |
| `position`, `previous_position` | R32F ray distance in the corresponding camera basis, not XYZ or ReLAX view-space depth |
| `normal`, `previous_normal` | R32_UINT: bits 0–14 and 15–29 encode normalized octahedral coordinates; bits 30–31 encode surface class |
| `material` | R32_UINT signature; bits 29–31 select effect slot 1–4, zero means no effect, and `0xffffffff` is treated as no effect |
| `previous_color`, `history_color` | B10G11R11_UFLOAT_PACK32 linear HDR history at output resolution when temporal reconstruction is enabled |
| `previous_camera`, `current_camera` | Storage buffers of at least 64 bytes: four float32 vec4 records in origin, forward, right, up order, using the native perspective camera basis |
| `outputs` | One RGBA8_UNORM or BGRA8_UNORM image, or 1–8 images with matching extent/format; BGRA requires formatless storage writes |

All images need storage usage on the same runtime. Write targets must not alias
inputs. Guide/history extents are validated when their features are enabled.
Format-correct dummy bindings can be used for disabled history/guide features;
they must still be valid allocations. The shader skips history writes when
reconstruction history is disabled and skips material reads when no effects are
active. The caller must initialize inputs actually consumed by the selected policy.

`ReconstructionSettings` and `ReconstructionEffect` describe the existing 256-byte
push-constant ABI without Python scene objects. The viewer computes effect
rectangles and supplies these values. History selection remains explicit through
`history_valid`; this reconstruction history is separate from ReLAX history.

For multiple outputs, `current_camera.forward.w` selects the array index (rounded
and clamped to 0–7). Set `operation(output_index=...)` to the same index so the
graph declares the correct output hazard. A single output is repeated across all
eight descriptors. Advanced parent passes can use `external_output=True` when
they provide synchronization for a dynamically selected output themselves.
Ordinary application graphs should keep the default and declare their selected
output normally.

FSR 2's **temporal upscaler is not implemented by this component**. The `fsr2`
setting consumes already upscaled, full-output-resolution HDR from an upstream
`VulkanFsr2` stage and performs display conversion.

## Native viewer migration

The native viewer now records denoising through `VulkanRelaxTemporal` and
`VulkanRelaxSpatial`, temporal upscaling through `VulkanFsr2`, and display
reconstruction through `VulkanReconstruction`.
The tracing executor no longer creates their pipelines/descriptors or records
private dispatch sequences. Signal preparation remains part of tracing.

`NativeDenoiserGraph` supplies borrowed views of the existing frame images,
history images, scratch images and policy buffers. No extra scratch allocation or
image copy is introduced. The graph declares version-zero reads of raw signals
before the spatial filter reuses that storage. The existing viewer policy uploads
still happen after the frame-slot fence, preserving motion-dependent rejection.

Denoiser bindings are prepared once per executor/configuration and survive changes
to the active render extent. Each operation can select a smaller active extent
within the prepared allocation. Spatial constants/workgroups change without
recreating kernels. Temporal extent changes require the externally managed policy
buffer to contain the matching extent. Window resize and executor replacement
retire the adapter before destroying its borrowed resources.

The native reconstruction adapter borrows the existing images and camera buffers.
FSR 2 supplies its HDR output as an explicit reconstruction input rather than
modifying the tracing executor's descriptors. The presenter still owns acquired
swapchain-image transitions: its cached reconstruction commands dynamically select
an output descriptor, while the surrounding primary command handles that image's
barriers. This avoids embedding a stale acquired-image handle in cached commands.

The native command cache remains enabled under its existing rules (FSR 2 and
custom GI builders still bypass it). Successful submission publishes the graph's
history and layout updates, including when commands are replayed from the cache.
The native queue and existing history semaphores provide inter-frame ordering.

## Recording a graph into an application command buffer

`compiled.prepare_recording(runtime)` prepares a graph without submitting it.
The result exposes `record(command)`, `resources`, `dependencies`, semaphore
requirements, and `submitted(completion)`.

The application must serialize access on `runtime.lock`, honor dependencies and
semaphores, retain all bindings/stages through completion, and call `submitted`
only after its queue submission succeeds. Recording alone does not publish layout
or history state. A failed recording cannot be retried or published. Cached
commands may publish subsequent completions, provided bindings, imported layouts,
and command-embedded policy remain compatible. General application code should
prefer `compiled.execute(runtime)` unless it owns the surrounding submission.

## Scope and remaining extraction

Temporal/spatial denoising and spatial upscaling/display reconstruction now use
reusable components in the viewer, alongside the `VulkanFsr2` temporal upscaler.
The remaining extraction includes broader tracing/presentation graph composition. The entire
native rendering pipeline is not yet one independently composed resource-backed
graph.

## Validation

The extraction preserves the recorded native algorithms and their internal
barriers. Six 24-frame glass/target/camera sequences at 640×480, 50% scale, with
AMD EASU and OrdinaryShade EASU produced byte-identical display output to the
captures made before this extraction.

`artifacts/denoiser-motion/gi-composition/smoke.py` exercises application-stage
insertion, restoring the default pipeline, and 25% rendering with bilinear,
OrdinaryShade EASU and FSR 2. CPU contracts are covered by
`tests/test_gi_pipeline.py`.

The independent spatial stage matched native HDR exactly on 30 camera-motion
frames at 161×121: six frames for each iteration count from one through five.
See `artifacts/denoiser-motion/gi-composition/spatial_compare.py` and
`spatial-metrics.json`. Opt-in GPU tests in `tests/test_relax_spatial.py` cover
combined and split graph operations, inferred ordering, odd extents, background
preservation, repeated execution, and lifetime rejection.

The independent temporal-plus-spatial graph matched the native viewer on 16
161×121 camera/glass-motion frames: temporal radiance, history lengths and final
HDR all matched exactly. See `temporal_compare.py` and `temporal-metrics.json` in
`artifacts/denoiser-motion/gi-composition`. GPU tests in
`tests/test_relax_temporal.py` also cover repeated ring execution, history reset,
failed submission, shared read-only guides, and consumer lifetime rejection.

After native migration, all six original 24-frame EASU sequences remained
byte-identical (144 frames total). `viewer_lifecycle.py` verifies cache replay,
100/50/25% active extents without reallocating denoiser bindings, resize in both
directions, and disposal for EASU and FSR 2. The spatial mode produced five cache
hits in its initial eight frames; FSR 2 preserved its existing cache bypass.
Results are in `migration-metrics.json` and `viewer-lifecycle.json` alongside the
other composition artifacts.

Reconstruction migration preserved all six 24-frame baseline sequences exactly
(`reconstruction-metrics.json`). Standalone GPU tests cover four spatial filters,
array output selection, packed tint parameters and enabled/disabled history
writes. Native cache, 100/50/25% extents, resize and cleanup also passed for EASU
and FSR 2 (`reconstruction-lifecycle.json`).


## Reusable FSR 2 temporal upscaler

`ordinarylight.runtime.VulkanFsr2` owns the AMD context, preparation kernels,
depth/motion/reactive scratch images and full-resolution HDR outputs. Its inputs
are explicit images from the same `VulkanRuntime`, with no scene or viewer dependency.
The optional native bridge is still required (`scripts/build_fsr2.py`).

```python
from ordinarylight.runtime import VulkanFsr2
from ordinarylight.pipeline.graph import VulkanGraph

upscaler = VulkanFsr2(
    runtime,
    inputs=[dict(hdr=hdr, view_z=view_z, motion=motion,
                 normal_roughness=normal_roughness)],
    extent=(640, 360), output_extent=(1280, 720),
)
jitter = upscaler.jitter(frame_index)  # use this when preparing the camera
operation = upscaler.operation(
    slot=0, jitter=jitter, fov_y=1.0472, dt_ms=16.667, reset=camera_cut,
)
graph = VulkanGraph().add("upscale", operation)
# A downstream operation can read upscaler.outputs[0] in this same graph.
graph.compile().execute(runtime).wait()
# Close downstream output borrowers before upscaler.close().
```

HDR is linear RGBA16F with sampled-image usage. The three guides require storage
usage: view depth is R32F, motion and normal/roughness are RGBA16F. Inputs can be
larger than the active render extent. Motion xy is pixel displacement including
current jitter; z contains previous view depth. Normal/roughness w supplies
roughness for the existing conservative reactive mask. The depth range is fixed
at 0.1–10000, with reversed finite depth prepared internally. FOV is in radians;
frame time is explicit milliseconds. The output is RGBA16F at output resolution.

An input list allows a frame ring to share one serial temporal context. Create a
fresh operation for each frame and submit it before recording the next. Multiple
frames from one context in the same graph are rejected. Cached command replay is
unsupported because SDK dispatch mutates CPU state during recording. Submission
publishes history only after success; failed or abandoned recording forces a reset
on the next dispatch. The component tracks the previous completion across slots.
Application-owned external command submission must honor graph dependencies and
keep the component/resources alive until GPU completion.

The native viewer borrows its existing inputs and uses these graph passes without
image copies. Reconstruction borrows the component's actual output allocations,
so lifetime checks prevent destroying FSR2 while reconstruction still uses them.
The adapter initializes the output layout once because its FSR2 and reconstruction
graph recordings are prepared separately before submission. Resize retires
reconstruction bindings before replacing the upscaler.

`tests/test_fsr2_context.py` covers failed/abandoned recordings and submission
tokens without a GPU. `tests/test_fsr2_graph.py` runs the component independently
with odd extents and a two-slot ring, checking prepared values, HDR output,
reset, command replay rejection and resource ownership. Native lifecycle checks
also cover scale changes, resize and cleanup.

After FSR2 migration, all nine 24-frame baseline sequences matched byte-for-byte
(216 frames, including 72 FSR2 frames). See `fsr2-metrics.json` and
`fsr2-lifecycle.json` in the composition artifacts.


## Shared presentation transfer

`ordinarylight.runtime.blit_operation(source, target)` provides a resource-backed
nearest-neighbor blit for RGBA8/BGRA8 UNORM display images. It supports different
source/target sizes and channel ordering, with no tone mapping or color-space
conversion. Source and target must be distinct, same-runtime images with transfer
source/destination usage. The graph infers producer/consumer ordering and records
the required layout barriers; both images finish in GENERAL by default.

```python
from ordinarylight.runtime import blit_operation

graph.add("display_transfer", blit_operation(display_image, destination_image))
```

For an acquired swapchain image, `present=True` leaves the destination in
PRESENT_SRC_KHR. Acquisition, its wait semaphore, the render-finished signal,
queue presentation and resource lifetime remain the caller's responsibility.
`VulkanOutput.present_operation` wraps this transfer with that lifecycle for
application graphs. The native viewer's fallback presentation path uses the same
transfer operation with its own acquisition and frame fences. Direct storage
presentation still uses the existing zero-copy path.

`tests/test_blit_graph.py` checks exact pixels for nearest scaling and RGBA/BGRA
conversion, inferred ordering, final layouts and alias/usage rejection. Native
composition smoke tests cover bilinear, EASU and FSR2 fallback presentation;
application presentation tests cover cancellation, repeated frames and resize.
Tracing, direct-storage swapchain management and full submission lifecycle
extraction remain outstanding.


## Scene-independent primary-ray generation

`ordinarylight.runtime.VulkanRayGeneration` exposes the split primary generator
as graph passes: queue-header reset followed by camera-ray and path-state
initialization. It borrows four same-runtime storage buffers and owns its kernel:

- `camera`: 64 bytes matching the native four-vec4 camera ABI.
- `rays`: a 16-byte queue header plus `capacity * RAY_DTYPE.itemsize`.
- `paths`: `capacity * HOT_PATH_STATE_DTYPE.itemsize`.
- `media`: `capacity * MEDIUM_STACK_DTYPE.itemsize`.

The ray queue also requires transfer-destination usage. Buffers must be distinct
and remain open until the stage is closed. The operation initializes the active
tile only; medium-stack slots beyond the initial IOR entry and other tracing
queues/secondary state remain the caller's responsibility.

```python
from ordinarylight.runtime import VulkanRayGeneration

rays_stage = VulkanRayGeneration(runtime, camera=camera_buffer,
    rays=ray_queue, paths=path_states, media=medium_stacks, capacity=4096)
graph.add("primary_rays", rays_stage.operation(
    extent=(640, 360), tile_origin=(0, 0), tile_extent=(64, 64),
    sample_index=0, sample_count=1, capture_secondary=True,
))
```

The camera ABI stores origin/frame index, forward, right, and up/projection in
four vec4s. right.w holds optional packed half-precision jitter coordinates;
zero selects the existing stochastic jitter. Camera preparation remains explicit.
The operation preserves the existing pixel/sample/RNG metadata and secondary
capture flag. It tracks completion for standalone submissions; external recorders
must retain the stage and buffers until their submission completes.

The native split tile/readback path now uses this component. It allocates full
medium-stack storage even for opaque scenes, because split generation initializes
one entry per ray. Returning to fused opaque tracing can shrink that allocation;
all affected scene descriptors are rebound when the buffer changes.

The viewer's fused primary/megakernel/hybrid paths still generate rays inside their
transport shaders. They are not migrated by this extraction. Intersection,
material shading, fused transport and scene bindings remain the next boundaries.

`tests/test_ray_generation_graph.py` checks analytic rays for an offset tile,
queue counts, metadata, capture flags, initial IOR and buffer lifetime without a
scene. `artifacts/denoiser-motion/gi-composition/ray_generation_smoke.py` exercises
native split tracing interleaved with fused presentation for glass and opaque
scenes, including medium-buffer replacement and cleanup.


## Split triangle intersection

`ordinarylight.runtime.VulkanIntersection` owns the ordinary closest-hit kernel
and borrows explicit ray/hit queues, a TLAS resource and a vertex-buffer resource.
Resident scenes already expose the required binding contract:

```python
from ordinarylight.runtime import VulkanIntersection

intersection = VulkanIntersection(runtime,
    tlas=resident_scene.resource("tlas"),
    vertices=resident_scene.resource("vertex"),
    rays=ray_queue, hits=hit_queue, capacity=4096,
)
graph.add("intersect", intersection.operation())
```

It composes with `VulkanRayGeneration` through the shared ray queue; no manual
stage ordering is needed. Ray and hit queues contain a 16-byte header followed
by `capacity` records (`RAY_DTYPE` and `HIT_DTYPE`, both 48 bytes). Hit storage
requires transfer-destination usage because the operation resets its header,
including for a zero-ray indirect dispatch. It preserves native ray/path IDs,
world-space hit position/distance, geometric normal, triangle index and barycentrics.
A miss uses distance -1 and primitive index 0xffffffff.

`operation(indirect=dispatch_buffer)` uses three caller-prepared uint32 dispatch
arguments at offset zero; the buffer needs indirect usage and enough 64-thread
workgroups for the active ray queue. The default dispatch covers capacity and
lets the shader check the queue count. Resources stay caller-owned; completion
tracking protects stage teardown. Resident scene leases prevent closing a scene
while the component uses it. Recreate bindings if the TLAS or vertex allocation
is replaced, and synchronize geometry updates before execution.

This contract is specific to the existing triangle scene ABI: TLAS instance custom
indices are triangle offsets, and the matching vertex buffer contains three
world-space vec4 vertices per triangle. The query treats triangles as opaque;
material response, alpha handling, medium transitions and further transport
belong to shading. This does not replace arbitrary acceleration-structure builders
or imply support for unrelated TLAS layouts.

Native non-bucketed split intersection now records this operation with the existing
GPU-generated indirect arguments. The adapter recreates bindings when native TLAS
or vertex handles change. Bucketed intersection and fused material/transport
shaders retain their existing paths. Legacy descriptor-layout slots remain reserved
in the executor, but the ordinary intersection pipeline is created by the runtime
component only when needed.

`tests/test_intersection_graph.py` checks analytic hit/miss records, both direct
and indirect dispatch, empty-queue reset, scene/buffer lifetimes and composition
with primary-ray generation. The native ray-generation smoke script also enables
split secondary intersection and checks that both extracted components are used.
The next extraction is shading/material-resource binding; full fused transport
and submission ownership remain backend-specific.


## Split shading contract

`ordinarylight.wavefront.shading` now defines the buffer portion of split shading's
version-1 ABI and its 56-byte push constants. Native descriptor updates and dispatch
parameter packing use these same helpers:

- `SHADE_BUFFER_BINDINGS`: immutable semantic-name to descriptor-slot mapping.
- `SHADE_WRITABLE_BUFFERS`: path, outgoing ray, medium, secondary-path and optional
  work-counter buffers that shading can modify.
- `shade_buffer_bindings(buffers)`: validates named bindings and returns an immutable
  slot mapping. Custom attributes and profiling counters are optional; required
  resources cannot silently be omitted.
- `WavefrontShadeSettings(...).pack()`: validates counts, light weight and sampling
  probabilities and packs the existing native shader ABI unchanged.

Settings describe bounce limits, analytic/emissive light counts and sample counts,
emissive selection weight, environment sampling, Russian roulette, fused
intersection, subgroup enqueue, secondary NEE and secondary-state capture. They
contain no renderer, scene or device objects. Scene-specific counts and weights
are supplied explicitly by preparation or by the native adapter.

This is a prerequisite for a reusable shading component, not that component itself.
TLAS binding 8, sampled texture array binding 13, sampled volume array binding 21
and custom material descriptor sets remain outside the buffer helper. Shader
variants must agree with optional descriptors. The next work is explicit sampled
resource-array support and material binding ownership, followed by migrating the
actual shading dispatch. The current fused/custom/bucketed algorithms are preserved.

`tests/test_shading_contract.py` checks exact packing compatibility, invalid
parameters, immutable mappings and optional descriptor behavior. The native split
smoke test exercises glass and opaque shading using this contract.


## Combined sampled-image arrays

`VulkanKernel(..., sampled_image_arrays={binding: pairs})` now accepts nonempty
arrays of `(VulkanResource.sampled_image(image), VulkanResource.sampler(sampler))`
pairs. These produce combined-image-sampler descriptors, as used by the native
material shaders. Each descriptor consumes one image and one sampler; allocation
counts and device sampler/sample-image limits include every array element.
Repeated pairs are allowed, and shared owners are retained once per kernel.

```python
kernel = VulkanKernel(runtime, spirv, buffer_bindings,
    sampled_image_arrays={13: tuple(
        (VulkanResource.sampled_image(image), VulkanResource.sampler(sampler))
        for image, sampler in texture_pairs
    )})
```

Array binding numbers cannot overlap scalar or storage-image-array bindings.
The shader's sampler dimension must match the supplied view (2D textures or 3D
volume views); this API does not allocate or convert views. Descriptors default to
GENERAL; `sampled_image_layouts` can select SHADER_READ_ONLY_OPTIMAL per array
binding. Declare matching compute shader reads in the graph, merging duplicate
image uses. Samplers need lifetime protection
but no image-memory hazard declaration. Descriptor-array indexing features remain
subject to the runtime's enabled device capabilities.

Kernel ownership now honors resource owners' `retain(consumer)` and
`release(consumer)` methods. Resident scene resources expose that protocol, so
kernels borrowing scene buffers or acceleration structures prevent premature
scene destruction. Owners without that protocol remain externally leased bindings.

`tests/test_sampled_arrays.py` verifies distinct 2D array elements, repeated
image/sampler pairs, exact sampled values, inferred producer ordering, invalid
bindings and lifetime rejection. Existing reconstruction/storage-array and scene
intersection tests also pass. Native material descriptor migration and resident
sampled texture/volume view adapters remain to be implemented; the viewer still
uses its existing sampled descriptors and read-only image layouts.


## Resident sampled texture and volume bindings

`resident_scene.sampled_resources("textures" | "volumes", count=None)` returns
stable `(sampled image, sampler)` resource pairs. Textures follow native ordering:
sRGB then linear for each scene texture. Volumes expose 3D views in visible-volume
order. The native uploader's dummy resources remain part of the returned table.
Optional `count` pads with the first pair, matching fixed-size native shader arrays;
it rejects truncation and padding an empty table. Without native textures enabled,
the texture table is empty.

```python
pairs = resident_scene.sampled_resources("volumes", count=16)
kernel = VulkanKernel(runtime, spirv, buffers,
    sampled_image_arrays={21: pairs},
    sampled_image_layouts={21: vk.VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL})
```

These views lease their owning scene. Closing the scene while a kernel uses any
pair is rejected. Imported layout is SHADER_READ_ONLY_OPTIMAL: keep graph uses and
descriptor layouts consistent with that layout when sharing with native rendering.
Merge duplicate image hazards after padding. A new resident scene requires new
bindings; do not retain old pairs across replacement or scene close.

The native executor now obtains its sampled descriptors from this same accessor,
preserving read-only layouts and existing sampler/image ordering. Shader pipelines
and descriptor sets are still executor-owned. `VulkanDescriptorSet` also accepts
array/layout arguments for the subsequent shading migration.

GPU tests now cover real resident 3D volume sampling, padding, read-only layout
preservation, scene leases, and the sRGB/linear 2D texture ordering. Native glass
and opaque split-tracing smoke checks passed after the descriptor-accessor change.


## Native compute shading descriptors

Ordinary split compute shading now uses `VulkanDescriptorSet` for immutable
buffer, TLAS, sampled-texture and sampled-volume bindings. The adapter consumes
the shared shading buffer map and resident sampled-resource accessor. It preserves
the shader's fixed descriptor layout, including the reserved custom-attribute slot
(using a valid dummy buffer when no custom attributes are supplied).

Descriptors are rebuilt at scene-binding changes, including medium-stack
replacement. The native parent waits for prior work before retiring old sets and
owns the leases on scene/queue allocations. These internal descriptor consumers
are included in renderer teardown accounting. Profiling keeps separate counter
bindings per frame slot; native textures keep the paired sRGB/linear descriptor
order. Existing material programs and pipeline layouts remain in use.

SER continues using the existing sets because its buffers/images declare additional
ray-generation shader visibility. Bucketed shading also retains its separate
sets. This migration does not yet extract material shader compilation, custom
material sets, or dispatch into an independent shading component.

The native smoke script now checks the descriptor migration across glass/opaque,
native-texture and profiling variants, interleaving split tracing with presentation
and verifying resource replacement and cleanup. The next step is reusable shading
dispatch with explicit hazards and the existing shader variants.


## Resource-backed shading dispatch

`ordinarylight.runtime.shade_operation(kernel, settings, *, indirect=None,
ray_capacity=None, after=())` dispatches a prepared split-shading kernel. It uses
`WavefrontShadeSettings` and the shared buffer ABI, merges aliased resource reads,
and declares writable path/medium/outgoing-ray/counter/secondary buffers, TLAS
reads, sampled image reads and indirect-argument dependencies.

The kernel supplies `bindings`, `sampled_image_arrays`, `sampled_image_layouts`,
`require_open()` and `bind(command, constants)`, as provided by `VulkanKernel`.
Use the existing split-shading shader with matching variants and a 56-byte push
constant range. Exactly one dispatch argument is required: `ray_capacity` for a
bounded 64-thread dispatch, or a caller-prepared 12-byte indirect-argument buffer.

```python
graph.add("shade", shade_operation(shading_kernel, settings,
    indirect=dispatch_arguments), after=("generate", "intersect"))
```

Generation and shading both write path/medium state, so declare their order (or
resource versions) explicitly. Initialize outgoing queue headers and secondary
state before shading. This operation does not reset queues, prepare scene data,
compile custom material programs, or manage resource lifetime. Keep its kernel,
images and buffers alive through completion; runtime kernel ownership protects
borrowed allocations. An attached `VulkanMaterialResources` set-1 bundle contributes its read dependencies; other custom descriptor sets remain outside this contract.

The native ordinary non-bucketed compute path now uses this operation with its
existing shader pipeline, immutable descriptor sets and GPU-produced indirect
arguments. The parent retains resources and owns submission; sampled image layouts
remain read-only. SER and bucketed shading still use their existing dispatch paths. Material set 1
is supported as described below.

`tests/test_shading_graph.py` runs the real split material shader in an independent
primary-generation/intersection/shading graph. It checks environment misses and
emissive hits, direct and indirect dispatch, terminated path flags and empty outgoing
queues. Native smoke checks cover glass/opaque scenes, native textures, profiling,
resource replacement and cleanup. Material shader preparation, additional material
sets and fused transport remain the next boundaries.


## Material set 1 in reusable kernels and shading

`VulkanKernel(..., material_resources=bundle)` includes a
`VulkanMaterialResources` descriptor layout at set 1, retains the bundle and binds
it with the kernel. The kernel's shader must be compiled with the bundle's resource
declarations. Cross-runtime bundles are rejected. Closing a bundle while a kernel
uses it is rejected, just as for attached renderers.

Material bundles expose `uses`, their merged graph read dependencies, and
`bind_graph(command, pipeline_layout)`, which binds after those dependencies have
been recorded. The latter deliberately avoids checking still-uncommitted image
layout fields. Existing `bind` and `synchronize` behavior remains available for
callers outside a graph.

`shade_operation` automatically includes `kernel.material_resources.uses` when a
bundle is attached. A texture or buffer producer can therefore precede shading in
the same submission. Generic custom kernel passes must explicitly include
`bundle.uses` in their pass uses. The graph records required transitions and
visibility; no separate `synchronize()` call is needed between same-graph producer
and consumer. Application controls still need to invalidate relevant rendering
history when material contents change.

Native non-bucketed, non-SER shading with extra material resources now uses this
graph path as well. Its adapter binds the already-prepared set-1 bundle and uses
the same read dependencies. Material shader compilation and preparation remain
separate work; this change preserves the current compiler and selected shaders.

`tests/test_material_kernel_graph.py` checks set-1 sampling from a transfer-written
image, repeated producer updates, graph ordering, layout publication and bundle
lifetimes. Existing camera material tests compare resource-backed shading with
constant equivalents and cover texture, buffer, uniform and scattering programs.


## Device-independent external material layout

`ordinarylight.materials.MaterialResourceLayout.from_programs(programs)` prepares
external material declarations without a GPU runtime or resource allocations. The
immutable layout sorts named declarations, assigns uniform/buffer slots and texture
image/sampler pairs, validates conflicts, and emits the same GLSL accessors used by
runtime binding. Camera shading defaults to descriptor set 1 and first binding 0.
Other executors can choose their own `descriptor_set` and `first_binding`.

```python
from ordinarylight.materials import MaterialResourceLayout
from ordinarylight.shaders.compiler import compile_wavefront_material_shader

layout = MaterialResourceLayout.from_programs(programs)
spirv = compile_wavefront_material_shader(
    "wavefront_shade.comp", programs,
    attribute_layout=attributes, attribute_binding=16,
    material_resources=layout,
)
```

This prepares shader bytes before device creation. It still requires the existing
GLSL compiler toolchain. Resource values are bound later through
`VulkanMaterialResources`; that bundle exposes its matching `resource_layout`.
The runtime binder and native shader compilation now use the same pure declaration
contract. Uniform size, buffer alignment, texture allocation types and runtime
compatibility remain checks at allocation binding time.

`tests/test_material_resource_layout.py` checks ordering, conflicts, immutability,
and source/SPIR-V preparation with runtime construction explicitly forbidden.
Material kernel and camera comparisons cover the later allocation/execution side.
This establishes the declaration boundary; shader variant selection, full scene
preparation and fused transport packaging remain separate work.


## Prepared split-shading stage

`ordinarylight.wavefront.prepare_shading` returns an immutable `PreparedShading`
without creating a GPU device. It contains SPIR-V, the selected `ShadeVariant`,
material declarations (if any), buffer and sampled-array requirements, the
64×1×1 workgroup size and 56-byte push-constant size.

```python
from ordinarylight.wavefront import ShadeVariant, prepare_shading

prepared = prepare_shading(variant=ShadeVariant(
    native_textures=True, profiling=False,
    overlapping_volumes=False, scattering_volumes=False,
))
# Later, after creating a runtime and binding scene/queue allocations:
kernel = prepared.create_kernel(runtime, buffer_and_tlas_bindings,
    sampled_image_arrays=sampled_pairs,
    sampled_image_layouts=sampled_layouts)
graph.add("shade", shade_operation(kernel, settings, indirect=dispatch_args))
```

Stock variants load existing packaged shader bytes. Flags cover OrdinaryShade
shading, native textures, profiling, overlapping volumes, single/multiple
scattering and empty-space skipping. Multiple scattering requires scattering.
Custom `programs` use the existing compiler with explicit `attribute_layout`,
optional `MaterialResourceLayout`, and optional material modifier. The prepared
object records those external resource declarations for later binding validation.

`create_kernel` checks required buffer/TLAS bindings, fixed sampled-array counts,
and material-layout agreement before constructing a `VulkanKernel`. It preserves
resource ownership checks in that kernel. Pipeline feature support still comes
from the chosen runtime; this is an in-memory native preparation object, not a
new browser export format.

The native base/custom shading pipelines use this preparation path. Native volume
variant naming also delegates to `ShadeVariant`. Pipeline-statistics collection
remains in the native pipeline-creation step. Transport, shader compilation and
runtime allocation are now distinct steps for ordinary split shading; fused
primary/continuation transport still has its own executor preparation.

Validation covers all 96 packaged combinations, native volume-name agreement,
custom SPIR-V equality with the previous compiler entry point, independent
prepared-kernel hit/miss graphs, camera material comparisons, and native
texture/profiling smoke checks. Next work is preparing the tracing stages around
this component and composing a complete independent split path.

## GPU queue dispatch preparation

`runtime.VulkanQueueDispatch(runtime, queue=queue, arguments=arguments)` borrows
an existing wavefront queue and a storage/indirect argument buffer. Its operation
runs the packaged dispatch-preparation shader, writing
`(ceil(min(count, capacity) / 64), 1, 1)`. Empty queues produce zero X groups.
The producer must supply a valid queue capacity within allocation and device
limits; this stage does not reset queues or validate GPU-written headers.

Add its operation to the same `VulkanGraph` as ray generation, intersection and
shading. Pass the argument buffer to `intersection.operation(indirect=arguments)`
and `shade_operation(..., indirect=arguments)`. Resource dependencies order the
queue producer, argument preparation and consumers, including the compute-write
to indirect-read barrier. No queue-count readback is needed. As with the other
stages, buffers are leased until the stage is closed and submitted work completes.

The standalone shading tests now execute this complete single-bounce sequence
for both hit and miss rays. Queue tests cover empty, exact workgroup boundary,
partial workgroup and count-above-capacity cases. The native executor still uses
its existing dispatch preparation; this addition does not change viewer output.
Multi-bounce queue recycling, output resolution and native scheduling migration
remain separate work.

## Independent multi-bounce scheduling

`runtime.split_trace_graph` builds a graph from a generation operation, two ray
queues, two `VulkanQueueDispatch` stages, two `VulkanIntersection` stages, two
prepared shading kernels, a shared indirect argument buffer, shading settings and
queue capacity. Stage index zero reads queue zero and writes queue one; index one
reverses those bindings. Path, medium and secondary state remain shared.

Each bounce resets the destination queue header, prepares dispatch arguments,
intersects and shades. The schedule uses explicit buffer versions so reads select
the correct bounce's data when allocations are recycled. It executes
`settings.max_bounces` iterations; empty queues dispatch zero intersection and
shading groups. The result remains in the path buffer. `reset_ray_queue` is also
available independently and preserves queue record storage.

The caller owns all stages and allocations, initializes any required secondary
state, and keeps them alive until completion. Supply whole-buffer split-stage
bindings and a generation operation that initializes queue zero and shared path
state. This helper covers separate intersection and ordinary shading; fused,
SER and material-bucket scheduling remain native executor paths.

GPU tests cover hit/miss paths, reflected continuation, empty later bounces and
repeat execution. Multi-bounce path bytes are compared with separate direct
intersection/shading submissions that do not use indirect dispatch or graph
resource versions. The viewer still uses its existing bounce scheduler. Output
resolution and native integration remain subsequent steps.

## Path-to-HDR resolution

`runtime.VulkanPathResolve` binds the packaged native path-resolution shader to
caller-owned `paths`, RGBA16F `hdr`, `secondary_paths`, `reservoirs`, `camera` and
`seeds` allocations. Construction takes path capacity and reservoir extent;
buffer sizes, format, usage and runtime ownership are validated before execution.
The stage retains these allocations through its kernel until closed.

Its `operation(path_count=..., sample_index=0, sample_count=1,
capture_secondary=False, sampled_indirect=False)` scatters path radiance using
pixel indices in path metadata. Sample zero replaces addressed pixels with
radiance divided by sample count; subsequent samples add their contributions.
Unaddressed pixels are preserved. Callers must ensure pixel indices are unique
within a dispatch and initialize any pixels that no paths cover.

Optional secondary capture updates signals, seeds and reservoirs on the final
sample. Sampled-indirect mode preserves the already sampled diffuse/specular
signals. All shader bindings remain explicit even when capture is disabled;
secondary storage covers path capacity, reservoirs use 24 bytes per reservoir
pixel, seeds use four bytes per reservoir pixel, and camera storage is 64 bytes.

Add resolution after the final trace stage (the graph infers this from path
reads). Capture also writes secondary state, so graphs with existing explicitly
versioned secondary writes must connect the next secondary version. The simple
HDR-only path can be appended directly to `split_trace_graph`.

Tests execute standalone generation, multi-bounce tracing, resolution and image
readback in a single graph, comparing HDR values with path radiance at half-float
precision. Additional tests check swapped pixel mappings, sample accumulation,
final-sample capture, sampled-signal preservation and resource leases. The native
viewer still uses its existing resolve recording. Extracting ReLAX signal-image
preparation and connecting the independent denoising stages remain next steps.

## ReLAX signal-image preparation

`runtime.VulkanRelaxPrepare` exposes the packaged native signal-preparation
shader without a renderer. It borrows path/secondary records, current and previous
64-byte camera buffers, previous world-space vec4 vertices, packed normal/material
images, and six output images: diffuse, specular, normal/roughness, view depth,
motion and identity. Formats and extents are checked at binding time; the kernel
retains all allocations. Current and previous cameras may share a buffer.

`operation(path_count=..., sample_index=0, sample_count=1, sampled_indirect=False,
transmission_motion_cap=False, planar_mirror_guides=False)` preserves the native
32-byte constants contract. Ordinary signals come from resolved secondary state;
sampled-indirect signals accumulate each sample. Final-sample preparation scatters
geometry and motion guides. Previous geometry must cover primitive indices in
valid secondary records. Pixel indices must be unique within a dispatch.

Outputs can bind directly to `VulkanRelaxTemporal` or `VulkanRelaxSpatial`. Callers
initialize uncovered pixels and identity storage: the existing shader does not
write identity for invalid primary hits. Geometry packing, previous-frame data and
history reset policy remain caller responsibilities. This extraction does not
change the native viewer's preparation recording.

GPU tests validate diffuse/specular scatter, normal/roughness, static-camera
reprojection, depth, identity and shared camera leases in ordinary and sampled
modes. The same graph feeds the independent spatial denoiser and checks its HDR
result. Native migration and a complete example with temporal history remain
subsequent integration work.

## Runnable temporal composition example

Run the independent denoising example from the repository root:

```bash
.venv/bin/python -m examples.composed_denoising --output /tmp/composed-denoising.npz
```

This deliberately uses one synthetic static-surface pixel with known radiance,
not a rendered scene. It makes the signal and history contracts inspectable:
caller-supplied path records feed preparation, a two-slot temporal history ring,
spatial filtering and HDR readback. No native renderer or presentation window is
created. Vulkan and the GLSL compiler used by `compile_compute` are required.

Six frames should report HDR `(3, 6, 9)` and history lengths `1, 2, 3, 4, 1, 2`.
Both histories are reset before frame four. The NPZ contains `hdr` and
`history_length` arrays; it is a numerical artifact, not an image gallery.
`tests/test_composed_denoising.py` checks these results automatically.

Because geometry and camera are static, this example shares immutable guide
values between history slots. Moving-scene integration must preserve separate
previous-frame guides, camera and vertex data. The next integration boundary is
feeding these stages from a rendered scene across moving frames, rather than
synthetic records.

## Rendered camera-motion example

```bash
.venv/bin/python -m examples.composed_motion --output /tmp/composed-motion.npz
```

This example traces a large emissive triangle through the independent split
pipeline, resolves its paths, prepares motion/signals, and runs temporal and
spatial denoising. It uses separate guide images for each history slot. A camera
translation produces quarter-pixel horizontal reprojection each frame; the center
pixel's history should grow from one to six. The default extent is 16x16.

The NPZ stores all frames of linear `hdr`, `history_length` and `motion`. A PPM
preview of the final frame uses a simple Reinhard mapping and display gamma. The
flat surface is a diagnostic fixture rather than a visual quality benchmark.

Trace and denoising are separate synchronized submissions on the same runtime;
path data stays GPU-resident. This keeps explicit recycled trace versions local
while demonstrating composition through shared allocations. Only final diagnostic
images are read back. Camera uploads and secondary-state clearing occur between
completed frames.

The example derives packed geometric normals from primary intersections through
the scheduler’s `primary_guides` operation. It still uses a single material key. Roughness is zero, matching the current split shader's
primary-position guide encoding; arbitrary material roughness and instance
identity need a general primary-guide producer. Geometry is static, so previous
vertices are fixed. Object motion, disocclusion and general primary-guide
production remain follow-up work. `tests/test_composed_motion.py` checks traced
radiance, quarter-pixel motion, expected previous depth and growing history.


## Primary-hit guide insertion

`split_trace_graph(..., primary_guides=operation)` inserts a caller-owned
`VulkanOperation` after primary intersection and before shading. The operation
can consume the primary hit queue and path metadata while those values are still
live, before subsequent bounces recycle their buffers. Its resource accesses
participate in explicit buffer versioning and normal graph image dependencies.
The default remains unchanged when the operation is omitted.

The moving-camera example uses this hook to pack actual geometric hit normals;
its GLSL producer is an example-specific operation, not a universal material-guide
implementation. It leaves material key zero for the single-material fixture.
Tests preserve primary hits through later empty bounces, confirming both hit and
miss data survive queue recycling. General material/instance metadata and shading
normal evaluation remain the next pieces to expose through this boundary.


## Scene-derived primary metadata

`wavefront.prepare_primary_metadata(scene)` creates an immutable byte table in
resident triangle order: material ID, stable scene-local instance ID, base
roughness bits and one reserved word. This preparation creates no GPU resources.
Rebuild it after topology, visibility or material changes; keep triangle ordering
aligned with the resident scene.

`runtime.prepare_primary_metadata_shader()` separately compiles the small stage.
Bind its SPIR-V and the uploaded table through `VulkanPrimaryMetadata`, alongside
paths, captured secondary paths and an R32_UINT material guide image. Its
operation runs after tracing and before signal preparation, updating captured
roughness and instance identity while preserving primitive and barycentric data.
Invalid primary records write material key zero. Callers retain responsibility
for initializing uncovered pixels and matching the scene table.

The camera-motion example now applies this stage between resolve and ReLAX
preparation, replacing its fixed material key and zero-roughness restriction.
Tests cover two distinct materials/instances and reversed primitive lookup, plus
camera-motion integration. This table represents **base material roughness**;
texture-modulated/custom material evaluation, normal mapping and transmissive
primary captures still require corresponding evaluated guide producers. The
stage does not claim support for those paths yet. Native viewer behavior remains
unchanged.

## Native resolve integration

The native executor's `record_path_to_hdr` now delegates through
`targets.vulkan.path_resolve_graph` to the same `path_resolve_operation` used by
`VulkanPathResolve`. Push constants, resource access declarations, graph barriers
and dispatch are shared. The adapter borrows existing native descriptors and
allocations; it introduces no new images, buffers or pipeline objects.

Native descriptor placeholders can alias secondary storage. The shared operation
combines aliased access declarations, preserving native behavior. The parent
continues to own submission and lifetime. HDR remains in GENERAL layout, and the
operation has no temporal history state to publish after submission.

Validation: 11 standalone resolve/tracing tests; four native glass/opaque,
texture/profiling split-trace smoke cases; cached frame, extent-change, resize and
close checks for FSR1-Shader and FSR2; and four moving-camera HDR captures matching
the previous native resolve recording byte-for-byte. Native resolve pipeline and
descriptor creation are still executor-owned. Signal-preparation recording is the
next duplicate native path to migrate.

## Native signal-preparation integration

`VulkanWavefrontExecutor.record_relax_prepare` now delegates through
`targets.vulkan.relax_prepare_graph` to `relax_prepare_operation`, the same
recording contract used by `VulkanRelaxPrepare`. Sample accumulation flags,
transmission motion caps, planar-mirror guides, push constants, access declarations
and dispatch are shared. Native frame images, camera/geometry buffers, pipeline
and descriptors remain borrowed from the executor; no additional GPU allocations
are introduced. All image layouts remain GENERAL and history publication belongs
to the later temporal stage.

Validation includes four independent signal/motion/history tests, native cached
frames and extent/resize/close checks with FSR1-Shader and FSR2, and four
moving-camera HDR frames matching the previous native signal recorder
byte-for-byte. This removes duplicate recording logic, not native pipeline or
descriptor creation. Moving that preparation/ownership boundary and caching
prepared recordings remain outstanding before this portion is fully extracted.

## Native preparation ownership and graph reuse

Native signal preparation now instantiates `VulkanRelaxPrepare` per frame slot.
The component creates and owns its pipeline and descriptor set; legacy executor
signal-preparation pipeline/layout/set creation is disabled. Native allocations
are borrowed, active extent may be smaller than allocation extent, and bindings
are replaced when handles/views change. Swapchain teardown retires these stages
before destroying images. Lifecycle checks assert the old pipeline is absent,
sets are empty, and stage identity survives active-resolution changes.

Compiled signal graphs are cached per slot by extent, path count, sample index/
count and guide policy flags. Each slot keeps at most 32 variants with LRU
replacement. Binding changes clear that slot's cache; teardown clears all caches.
Each use still prepares a fresh single-use `VulkanGraphRecording`, preserving
validation and recording safety rather than replaying its internal callable.

Validation forces native Vulkan command rebuilding and checks compiled graph
identity is reused, then exercises resolution changes, resize and close with
FSR1-Shader and FSR2. Four standalone signal/history/motion tests pass. This change
removes repeated operation construction/graph compilation; no new performance
claim is made without measuring the remaining recording and binding-check costs.
Resolve pipeline/descriptor ownership is the next migration boundary.

## Native resolve ownership

Native resolve now creates a reusable `VulkanKernel` with explicit borrowed
bindings and the packaged resolve SPIR-V. The executor no longer creates its
resolve image pipeline, layout or descriptor sets. Descriptor-pool indexing was
adjusted to remove the old slots. Native dormant alias bindings remain supported
by the shared resolve operation; the stricter standalone `VulkanPathResolve`
allocation contract is unchanged.

Each frame slot caches the kernel by resource handles/views and at most 32
compiled graphs by extent, sample constants and capture flags. Resource changes
retire the old kernel/cache; swapchain teardown closes all resolve kernels before
image destruction. Each dispatch creates a fresh validated graph recording.

Validation: 11 standalone resolve/shading tests, four native texture/profiling
smoke configurations, and FSR1/FSR2 lifecycle checks asserting legacy resolve
pipeline/sets are absent and reusable kernels close on resize. Earlier legacy
performance measurements predate this ownership migration; their old recorder
requires native descriptor bindings that are no longer allocated and must be
adapted before rerunning that comparison.

## Compact indirect-reservoir initialization

`runtime.clear_indirect_reservoirs(buffer, count=...)` clears the existing
six-uint32 (24-byte) compact reservoir ABI. The caller supplies a transfer-
destination buffer and retains it until completion. Empty ranges are allowed;
trailing bytes remain untouched. The operation declares transfer writes and
explicitly makes the cleared range available for subsequent compute access,
including native consumers recorded outside its graph.

Native compact-reservoir initialization now uses this operation. The dedicated
clear compute pipeline/layout/descriptors are no longer created; candidate and
debug descriptor allocation offsets and binding entry points still work without
the old clear sets. Candidate generation itself remains native-owned.

Validation covers empty/partial clears, FSR1/FSR2 lifecycle checks, and the
`indirect_clear_smoke.py` fixture with compact storage and candidate reuse
explicitly enabled. Default and custom GI graph captures match in that fixture.
This is initialization extraction, not completion of ReSTIR candidate/reuse
modularization, and no speedup is claimed for a once-per-allocation clear.

## Indirect candidate recording

`runtime.indirect_candidates_operation` consumes an explicitly bound kernel for
`wavefront_indirect_candidates.comp.spv` (bindings 0–12, 36-byte constants).
Inputs include current/previous compact reservoirs, current/previous screen-space
guides and cameras, HDR, a profiling counter buffer and scene TLAS. The operation
validates extents, image formats, buffer capacities and writable-buffer aliasing.
Caller-owned history validity, frame index, spatial reuse and history limit are
explicit parameters. The caller retains allocations through submission.

Profiling adds counter clear and host-read visibility passes around dispatch;
normal graph hazards provide transfer-to-compute ordering. Current/previous
camera read bindings may share storage. The native candidate recorder delegates
to this operation while retaining its existing pipeline/descriptor ownership.
No temporal validity is published by this stage itself.

Two standalone GPU tests exercise empty geometry with profiling off/on. The
indirect-enabled native smoke fixture passes. Four moving-camera HDR frames match
the old candidate recorder with temporal/spatial reuse and profiling enabled;
this is rendered-output parity, not a byte comparison of reservoir state.
Candidate pipeline ownership, reservoir-state comparisons and remaining debug/
application paths still need migration and validation.

## Candidate pipeline ownership and reservoir parity

Native candidate generation now owns its pipeline and descriptors through a
resource-bound `VulkanKernel` per frame slot. The legacy candidate layout,
pipeline and sets are no longer allocated. A change in buffers, image views or
TLAS replaces the kernel; swapchain/executor teardown retires kernels before
resources. Debug image rebinding remains independent of candidate descriptor
creation. Candidate operation/history policy is still shared with standalone use.

`candidate_parity.py` compares the pinned legacy recorder and migrated recorder
using equivalent kernel bindings, with temporal/spatial reuse and profiling
active. Four moving frames match exactly in both HDR and nonzero reservoir words.
The diagnostic reads reservoirs through a compute copy, avoiding a requirement
for transfer-source usage on the production reservoir allocation.

Candidate source and reservoir extents are independent: native reduced-resolution
rendering can retain a larger reservoir grid. Both require positive dimensions;
image extents and reservoir byte capacity are validated separately.
`candidate_lifecycle.py` exercises this configuration across resize and shutdown.
Standalone empty-geometry/profiling tests remain in `test_indirect_candidates.py`.
Candidate graph caching, debug/application extraction and broader ReSTIR
integration remain follow-up work.

## Indirect visualization and application

`runtime.indirect_apply_operation` exposes the packaged compact-reservoir output
shader with explicit reservoir, HDR, seed and material bindings. It supports
radiance/history/validity/acceptance diagnostics and `apply` correction mode.
Extent, buffer capacity, image format, history limit and strength are validated;
application declares HDR read/write, diagnostics declare HDR writes. Callers own
resources and submission completion.

Native output recording now uses this operation and resource-bound VulkanKernel
pipeline/descriptor ownership. The legacy output pipeline/layout/sets are no
longer allocated. Kernels are replaced on binding changes and retired before
swapchain resources. Candidate generation and output remain independently
composable operations.

Three standalone GPU tests cover empty-reservoir HDR preservation and diagnostic
values. `indirect_output_smoke.py` exercises all five modes in the native fixture,
with identical default/custom graph HDR captures and successful reset/close.
These checks do not claim new lighting quality or performance improvements.
Broader direct-light ReSTIR, fused tracing and specialized scheduling remain
outside this compact indirect-reservoir extraction.

## Fused-primary settings boundary

`wavefront.PrimarySettings` names the native fused-primary push-constant tail,
including direct-light ReSTIR enable/history/candidate/spatial/MIS settings,
lighting sample counts, roulette, secondary capture and effect ranges. `pack`
accepts the existing 32-byte tile prefix and returns the unchanged 176-byte ABI.
This is device-independent value preparation; it validates binary representation
without changing native policy selection or shader semantics.

Native primary recording now constructs this object rather than assembling an
anonymous positional struct. Binary-offset tests cover ReSTIR and effect fields;
four native smoke configurations pass, and candidate parity still matches HDR
and compact reservoir words across four moving frames. Fused shader binding,
variant preparation and dispatch remain executor-owned. This establishes their
settings contract, not a completed standalone fused tracing stage.

## Fused-primary descriptor contract

`wavefront.primary_bindings(native_textures=False, profiling=False)` returns
immutable named descriptor requirements, including binding numbers, descriptor
kinds, counts and conservative access intent. It covers primary/hybrid/megakernel
shared set zero, current/previous ReSTIR buffers and guides, secondary state,
reserved custom attributes, and fixed sampled texture/volume arrays. It contains
no native handles. Runtime policy can disable writes; specialized shader variants
may use only a subset of this layout.

Native primary descriptor-layout creation now consumes this contract. Three CPU
ABI/settings tests and four native texture/profiling smoke configurations pass.
This is a common resource description, not yet a standalone fused executor:
allocation binding, variant preparation and fused dispatch remain outstanding.

## Fused compute dispatch and validation freeze

`runtime.primary_operation(kernel, constants, workgroups=...)` now exposes fused
primary, hybrid and megakernel compute entry points as a `VulkanOperation`.
`constants` is the 176-byte result of `PrimarySettings.pack(tile_constants)`.
The caller selects the compiled shader variant and matching dispatch geometry.
The operation consumes the named primary descriptor contract, declares scene,
TLAS, queue, path, guide, reservoir, sampled-image and material-set dependencies,
and combines access masks and byte ranges for aliased resources. Invalid binding
sets, array counts, resource kinds, runtimes and constant sizes are rejected.

An independent application can supply a `VulkanKernel` with these bindings,
`push_constant_size=176`, the required sampled arrays and optional material set,
then insert this operation in its own `VulkanGraph`. Preparation of shader bytes
and settings remains separate from constructing that kernel on a device.
Resources and the kernel must remain alive until execution completes.

The native viewer uses the same operation through `NativePrimaryKernel`, borrowing
its existing descriptors and selected pipeline. This preserves shader variants,
custom material binding and pipeline statistics without allocating GPU resources
while recording. The reserved custom-attribute binding always has a valid buffer.
The parent still owns descriptor updates, resource replacement, command caching,
submission and lifetime. Native image layouts stay GENERAL; sampled scene images
stay read-only. This adapter is not an independently owned native fused executor.

Feature extraction is paused at this boundary. SER/SBT dispatch, persistent
continuation, material-bucket scheduling and general evaluated primary-guide
production remain specialized backend work. They are not claimed to be fully
modularized. Performance and validation results for this freeze are recorded in
[the validation report](../artifacts/denoiser-motion/gi-composition/validation_freeze.md).

### Importing an existing acceleration structure

`runtime.import_scene(scene, acceleration=tlas_resource, buffers=bindings)`
creates resident scene resources without building a BLAS or TLAS. The supplied
`VulkanResource` must identify a same-runtime acceleration structure with an
owner implementing `require_open`, `retain`, and `release`. That owner must also
keep every referenced BLAS and its required dependencies alive. Both uploaded
native scenes and `VulkanTransportScene` provide this ownership contract.

`scene` supplies native material/lighting metadata and packing capabilities. It
may be empty; no dummy triangle is created. Optional `bindings` map names from
`resident.bindings` to application-owned `VulkanBuffer` allocations using the
native scene ABI. Supplied allocations are bound directly, without copying or
uploading replacement contents; omitted bindings are uploaded from `scene`.
Buffers must have storage usage, share the runtime, and fit the packed metadata.
When packing materials, use `scene.triangle_material_data(programs, default)`
with the same material program order/default as the renderer. Program IDs are
part of the ABI, including when the material parameters otherwise match.

Imports lease their owners until replacement or close; close attached pipelines
before the imported scene, and the imported scene before its external owners.
The constructor performs initialization at a resource boundary. Publish already
submitted same-queue content/AS updates using `notify_content_changed`; replace
handles using `replace_resources`. Neither method implies cross-queue ownership
transfer. CPU changes to the metadata require a new resident snapshot.

### Typed native custom geometry

Declare an intersection function using
`@ordinaryshade.function(name="nativeIntersectCandidate")`. Its parameters are
`(origin: vec3, direction: vec3, t_min: f32, t_max: f32, primitive: u32,
instance: u32, instance_offset: u32)` and its result is `NativeIntersection`.
The callback owns traversal within the candidate primitive. Return a miss with
`nativeIntersectionMiss()`, or a valid distance, unit geometric/shading normals,
application `identity: uvec4`, texture coordinates, and previous world position.
The query fills the committed native `address` independently of application IDs.
Shading normals must share the geometric normal's hemisphere. Both ray vectors
and returned normals use world space. Native custom surface queries include
visibility mask bits 1 and 2.

Declare the material callback using
`@ordinaryshade.function(name="nativeEvaluateMaterial")`, taking
`(hit: NativeIntersection, cone_width: f32)` and returning `NativeMaterial`.
The cone width is a world-space footprint; the application evaluates its own
material data. The returned parameters use OrdinaryLight's native BSDF.
`NativeIntersection`, `NativeMaterial`, and `nativeIntersectionMiss` are exported
from `ordinarylight.geometry`.

```python
from dataclasses import replace
from ordinarylight.geometry import (
    NativeGeometryBuffer, NativeGeometryProgram, VulkanNativeGeometryResources,
    NativeMaterial,
)

program = NativeGeometryProgram(
    intersect_cell, evaluate_cell_material,
    buffers=(NativeGeometryBuffer("cells", CellRecord),
             NativeGeometryBuffer("cell_materials", NativeMaterial)),
)
with VulkanNativeGeometryResources(runtime, program, {
    "cells": cell_buffer, "cell_materials": material_buffer,
}) as geometry:
    config = replace(base_config, geometry_resources=geometry)
    with runtime.import_scene(metadata, acceleration=app_tlas, config=config) as resident:
        with VulkanWavefrontPipeline(runtime, resident, config=config) as gi:
            frame = gi.prepare(camera, output_extent)
            graph = VulkanGraph().add("lighting", frame.operation)
            # Add GPU consumers of frame.images and frame.buffers here.
            completion = graph.compile().execute(runtime)
```

Use a base configuration with `wavefront_ordinaryshade_shade=True`,
`wavefront_fused_secondary=True`, `wavefront_custom_inline=False`,
`wavefront_material_bucketing=False`, and indirect reservoir reuse disabled.
Denoising is supported with `denoiser_enabled=True`, `temporal_history=True`,
and `progressive_accumulation=True`. Alternate continuation paths remain
explicitly rejected. Custom scenes currently exclude native volumes and
custom triangle attribute/material programs. Emissive custom surfaces work via
path sampling and optional application-owned emitter sampling/PDF callbacks.
Smooth lossless optical boundaries are supported as described below. Ordinary triangle rendering retains its existing modes.

Buffer declarations are typed read-only std430 arrays at set 2, assigned in list
order. The supplied buffers must match the declarations and share the runtime.
Each value can be a `VulkanBuffer` or a storage-buffer `VulkanResource` view,
including `transport_scene.resource("materials")` or a bounded `.byte_range()`.
The view's owner must implement `require_open`, `retain`, and `release`; the
bundle leases that owner and preserves the descriptor's byte offset and range.
Uniform-buffer and non-buffer descriptors are rejected. This permits borrowing
resident scene buffers without copying or accessing private allocations.
Programs are immutable; resource bundles retain allocations until replacement or
close. Buffers can be written by application GPU stages between frames. Publish
already-submitted same-queue changes with
`geometry.notify_content_changed(after=(completion,), invalidate_history=False)`
to retain command recordings. `geometry.replace_buffers({"cells": new_buffer})`
leases replacement allocations, waits at the explicit resource boundary, and
invalidates native commands/history. Submit or cancel a prepared frame before
publishing changes. Update the application's acceleration structures coherently;
voxel occupancy and dirty-slot tracking remain application-owned. Rebuild any
application graph that references replaced allocations directly.

Custom history uses `NativeIntersection.previous_position`: xyz is the previous
world-space position of the same surface point, and w greater than 0.5 declares
valid correspondence. Non-finite positions or w at most 0.5 reject temporal
history locally. Static geometry should return its current world position with
w=1. The application can invalidate dirty slots without a global reset by
returning w=0 for those hits and publishing buffer changes with
`invalidate_history=False`. This does not make concurrent buffer writes safe;
follow the same-queue update and lifetime rules above.

All four application identity words participate in a 32-bit fingerprint used by
the denoiser's identity guide. This fingerprint can collide; consume the exact
`uvec4` primary-hit identity for slot/face averaging. Denoiser guides describe the
last contributing camera sample per GI pixel; primary-hit records retain every
contributing sample. Both use the active GI pixel extent, before upscaling.
The extra 32-byte-per-pixel history buffer is device-local and is allocated and
written only when denoising or signal capture is enabled.


### Custom optical boundaries

Pass an optional typed callback as `NativeGeometryProgram(..., boundary=callback)`:

```python
@osh.function(name="nativeEvaluateBoundary")
def evaluate_boundary(hit: NativeIntersection) -> NativeOpticalBoundary:
    record = cells[hit.address.y]
    return NativeOpticalBoundary(record.parameters.xy, record.metadata.z != osh.u32(4294967295))
```

`NativeOpticalBoundary` is exported from `ordinarylight.geometry`. Its `ior` pair
is `(outside, inside)` and `enabled` selects a smooth lossless dielectric instead
of the material BSDF. The geometric normal must point outside. IORs must be
finite and positive; disabled or invalid boundaries fall back to material
shading and opaque visibility. Application identity is available at every
callback, independently of the acceleration primitive address.

Primary and fused secondary stages use the same exact unpolarized Fresnel
sampler, including total internal reflection and radiance-mode eta-squared
transmission. The explicit media pair also handles a camera starting inside a
solid. No boundary-buffer upload or CPU work is inserted between GPU stages.
Optical callbacks must agree on the media shared by adjacent/nested faces.
The native stack retains IOR tracking for subsequent native material surfaces;
it does not validate application boundary topology or overlapping media.

All native shadow and ReSTIR visibility queries use the same boundary callback.
An enabled index-matched boundary passes a straight visibility ray; a refractive
boundary blocks it. Direct-light connections do not trace refractive caustics.
Path samples can still carry light through the refractive boundary. Boundary
transport currently excludes absorption, scattering, and rough interfaces;
material attenuation parameters do not add those effects to this callback.
Existing material-based transmission remains available when no boundary is enabled.


### Application-owned emitter sampling

Attach `NativeEmitterProgram(count, select, evaluate, pdf)` with
`NativeGeometryProgram(..., emitters=emitters)`. All four callbacks are typed
OrdinaryShade functions validated when the geometry program is compiled:

| Exported function | Signature | Meaning |
| --- | --- | --- |
| `nativeEmitterCount` | `() -> u32` | Current number of area emitters |
| `nativeSelectEmitter` | `(selector: f32) -> u32` | Select an emitter using a uniform variate |
| `nativeEvaluateEmitter` | `(emitter: u32, coordinates: vec2) -> NativeEmitterSample` | Map two uniform variates to an emitter point |
| `nativeEmitterPdf` | `(hit: NativeIntersection) -> f32` | Joint emitter-selection/area PDF of an emissive hit |

`NativeEmitterProgram` and `NativeEmitterSample` are exported from
`ordinarylight.geometry`. A sample contains world-space `position`, unit `normal`,
nonnegative linear `emission`, positive finite `area_pdf`, and `two_sided`.
The area PDF includes the probability of selecting the emitter; do not include
the area/environment domain probability or a solid-angle Jacobian. OrdinaryLight
applies those factors. The hit-PDF callback must describe the same distribution;
return zero for hits outside its support. Its exact instance/slot/face identity
is independent of the enclosing acceleration primitive. Emission and sidedness
must agree with the material callback.

These callbacks own the **complete area-emitter distribution**, replacing native
triangle-emitter selection for this pipeline. Include native triangle emitters
in the application distribution if they should receive next-event samples.
Emissive surfaces omitted from it still contribute through BSDF path hits.
Point/directional/spot and environment lighting retain their existing interfaces.
When unified area/environment sampling is active and both domains exist, the
custom area domain uses probability 1/2. Otherwise its probability is 1 or 0.

Emitter IDs are contiguous in `[0, count)` with `count <= 0x01fffffe`; the upper
value is reserved for the environment candidate. ReSTIR stores these IDs and
half-precision sample coordinates, then calls the evaluator again during reuse.
Keep IDs and the coordinate-to-surface mapping stable while retaining history,
and handle coordinates rounded to either endpoint. Reset history when the
mapping or sampling distribution changes. Sampling, PDF, and count callbacks
read the same persistent application buffers as the geometry callbacks.
No proxy triangles or CPU emitter-count readback is required. Use the existing
resource update/publication methods to make changes visible safely.

Primary and fused secondary next-event sampling and emissive-hit evaluation use
complementary unit-count power-heuristic weights. The sample count still controls
estimator averaging; it does not change the MIS partition. This also keeps the
partition consistent across ReSTIR candidate counts. Delta optical events retain
unit emissive-hit weight. All emitter visibility uses the shared geometry and
optical-boundary query contract.

GPU comparisons check ordinary NEE, multiple light samples, ReSTIR candidates,
temporal/spatial reuse, mixed area/environment domain selection, and GPU-resident
emitter-count updates against a path-only reference. A second scene hides the
diffuse receiver behind a reflection to exercise secondary NEE. GPU work counters
verify that shadow queries and accepted reservoir history actually occur.
