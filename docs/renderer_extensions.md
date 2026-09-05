# Building rendering algorithms on OrdinaryLight

OrdinaryLight now provides an independent Vulkan runtime, resident scene
snapshots, explicitly ordered GPU passes, versioned transport GLSL, application
history, and external-HDR output. A renderer can use these without constructing
the built-in GI algorithm. Existing GI constructors and scientific viewers keep
their defaults.

This is a first explicit execution interface. It uses one queue, conservative
barriers and host waits for completion dependencies. It does not implement an
optimizing render graph, concurrent queues, automatic voxel connectivity, or
SDF construction.

## Runtime and ownership

```python
import ordinarylight as ol
from ordinarylight.renderers.gi import VulkanGlobalIlluminationRenderer

with ol.VulkanRuntime() as runtime:
    # No GI shaders have been compiled here.
    with runtime.buffer(4096) as voxel_state:
        ...
    renderer = VulkanGlobalIlluminationRenderer(runtime=runtime)
    try:
        assert renderer.compute_context is runtime
        # Existing render_frame/render_gpu entry points remain available.
    finally:
        renderer.close()  # borrowed runtime remains open
```

The runtime owns device discovery/creation, queue, command pool, extension
functions and pipeline cache. Allocations, scene snapshots, kernels, and
renderers own their subordinate resources and retain the runtime. Close those
consumers before closing the runtime; an early close raises without destroying
the device. Outstanding submission completions are waited and retired at runtime
close. Use context managers or explicit `close()`; garbage collection is not a
resource management strategy.

The initial runtime profile requires a hardware Vulkan 1.2 adapter with
acceleration structures and ray queries, even for standalone compute. This is
not yet the general compute-only profile for devices without ray queries.
`runtime.capabilities` distinguishes enabled ray-pipeline, native-texture,
external-memory, presentation, and custom-intersection facilities. Ray queries
are available; built-in scene upload supports triangles. `custom_intersections`
is true for the new `VulkanTransportScene` AABB callback path, including an
analytic SDF sphere. Existing camera GI scene upload remains triangle-only.
See [transport foundations](transport_foundations.md) for the common hit contract,
non-camera multi-bounce integrator, and dielectric semantics.

`config=RendererConfig(...)` retains the existing device feature policy.
A borrowed renderer can change rendering options but cannot enable device
features absent from its runtime. The same runtime accepts existing
`VulkanComputeSequence(..., context=runtime)`. GI `compute_context` now returns
this public runtime; the high-level `Renderer` forwards it. Raster/WebGPU
compute contexts retain their existing interfaces.

`runtime.buffer(size, data=...)` provides a persistent, host-coherent storage
buffer (also transfer source/destination); `upload()` and `read()` synchronize
before host access. `runtime.image(width, height)` creates a device-local
RGBA32F single-mip, single-layer 2-D color image. Vulkan usage/format overrides
are explicit. Other image dimensionalities and transient heaps are not part of
this first allocation API. Scene upload still supports its existing 3-D volume
images.

All use of a shared device queue, command pool and pipeline cache must be
serialized. Runtime submit/allocation operations and the ordered pass executor
use `runtime.lock`. When mixing existing GI or reflected-compute calls with
application work, hold that same reentrant lock around the complete operation.
Existing scientific workbenches already serialize compute and rendering on one
worker. A runtime owns at most one presentation surface; it is not a multiwindow
swapchain manager. Existing four-panel viewers retain their per-panel devices.

## Resident scene snapshots

```python
with runtime.upload_scene(scene) as resident:
    renderer.use_scene_resources(resident)
    try:
        # renderer.render_frame(scene, camera, ...) uses the resident upload
        tlas = resident.resource("tlas")
        materials = resident.resource("material")
    finally:
        renderer.close()  # unbinds, does not close resident
```

`VulkanSceneResources` lives in `ordinarylight.targets.vulkan.scene` and is also
returned by `runtime.upload_scene()`. The built-in renderer uses the same scene
upload implementation. No GI shader compilation happens in standalone upload.
Custom material programs describe packing but must be compiled into the
application's shader separately.

`bindings` maps `vertex`, `previous_vertex`, `material`, `light`, `area_light`,
`attribute`, `custom_attribute`, `texture`, `texture_binding`, `volume_header`,
`volume_scalar`, `volume_transfer`, and `triangle_volume` to borrowed buffers.
`resource(name)` returns a typed pass view, including `tlas`. Native texture and
volume image/view/sampler objects are exposed through `scene_sampled_textures`
and `scene_sampled_volumes`. Never destroy these native handles yourself.

`revisions` records uploaded scene, geometry, shading and transform revisions.
`content_signatures` separately fingerprints packed material/texture content
and analytic/emissive lighting content, allowing different history invalidation.
Changing emissive geometry can change the lighting signature as well as geometry.
Texture changes also conservatively invalidate lighting, including environment maps.
Material-program configuration remains an additional application invalidation
dependency. `primitive_ids` is a read-only array of stable scene instance IDs in
packed primitive order; it is not a voxel ID array. The application supplies
its own mapping to voxel, probe or lightmap IDs.

Standalone snapshots are immutable: after scene mutation, upload a replacement
and bind it. A renderer rejects stale borrowed snapshots, including changed GPU
volume sources. Close/unbind all borrowing renderers before retiring a snapshot.
Renderer-owned scenes retain existing partial uploads/refits and GPU volume
refresh behavior. A renderer may be closed before its borrowed scene; a runtime
may not be closed before either.

Packing is transport ABI v1: positions are world-space vec4 records; vertex
attributes are normal/texcoord/tangent vec4 records; materials and analytic/area
lights match `transport.shader_source("types")`; volumes match the existing
`VOLUME_HEADER_ABI`. Scene TLAS custom indices are packed primitive offsets.
External code must use that offset together with the per-instance primitive
index, rather than treating a TLAS instance index as a material index.

## Ordered GPU passes

`ordinarylight.pipeline.vulkan` provides `VulkanResource`, `VulkanResourceUse`,
`VulkanPass` and `VulkanPassPipeline`. The original logical `RenderStage` and
`RenderPipeline` API is unchanged.

```python
import vulkan as vk
from ordinarylight.pipeline.vulkan import (
    VulkanResource, VulkanResourceUse, VulkanPass, VulkanPassPipeline,
)

use = VulkanResourceUse(
    VulkanResource.buffer(voxel_state),
    stage=vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
    access=vk.VK_ACCESS_SHADER_READ_BIT | vk.VK_ACCESS_SHADER_WRITE_BIT,
)
completion = VulkanPassPipeline([
    VulkanPass("accumulate_voxels", (use,), kernel.bind,
               workgroups=(voxel_group_count, 1, 1)),
]).execute(runtime, after=(previous_completion,))
completion.wait()
```

A pass records native Vulkan commands. If `workgroups` is supplied, its recorder
binds the kernel and descriptors; the executor records `vkCmdDispatch` with
those exact group counts. Group counts have no relationship to image dimensions.
Recorders must not submit, destroy resources, or mutate tracked layouts.

Each use declares a typed buffer, image or acceleration structure, Vulkan stage
mask and access mask. Image uses additionally declare the required layout. Uses
within a pass must combine read/write access into one declaration per native
resource. The executor emits barriers before passes and publishes writes at the
end. Image layout state commits only after successful submission. Buffers and
images remain allocated between executions. Views retain their owners through
submission; closing an allocation waits for device work, and subsequent use of
closed resources is rejected.

Completion dependencies must belong to the same runtime. They currently wait on
the host before recording; submission itself returns a fence-backed
`VulkanCompletion`. Waiting releases its fence/command buffer, and is idempotent.
Raw `runtime.submit(recorder, resources=..., after=...)` is also available; that
lower-level recorder owns its barriers and layout bookkeeping.

`VulkanKernel` in `ordinarylight.runtime` builds a compute pipeline with immutable
set-0 storage-buffer/storage-image/AS bindings and optional push constants. It
uses the runtime pipeline cache. `compile_compute()` accepts complete GLSL and
caches SPIR-V by exact source. The reflected `VulkanComputeSequence` remains the
convenient path for OrdinaryShade buffer kernels. Sampler arrays, uniform
buffers and custom descriptor layouts can use native recorders; they are not
silently inferred by `VulkanKernel`.

## Transport ABI v1

`ordinarylight.transport.shader_source(component)` returns include-expanded GLSL
for `types`, `contracts`, `lighting`, `volumes`, `sampling`,
`dielectric`, or `sdf_sphere`. Files are packaged under
`ordinarylight/shaders/transport_v1`. For GLSL includes, use the parent `shaders`
directory as the include root. Built-in GI includes these same components through
its existing adapter filenames. No OrdinaryShade rewrite is required.

Lighting requires these explicit declarations before inclusion:

- `MaterialData`, `PointLightData`, `AreaLightData` from `types`.
- `point_lights[]`, `area_lights[]`, and `scene_tlas` with the application's
  descriptor bindings.
- `sampleSceneTexture(int, vec2)` and
  `float volumeShadowTransmittance(vec3, vec3, float)` adapters.
- `OL_TRANSPORT_POINT_LIGHT_COUNT`, `OL_TRANSPORT_AREA_LIGHT_COUNT`,
  `OL_TRANSPORT_AREA_LIGHT_WEIGHT`, `OL_TRANSPORT_ENVIRONMENT_SAMPLES`, and
  `OL_TRANSPORT_SECONDARY_AREA_LIGHT_SAMPLES` expressions.

Optional `WAVE_*` feature definitions select existing OrdinaryShade, profiling,
and ReSTIR specializations; undefined features follow GLSL's disabled `#if`
behavior. Enabling them requires their corresponding generated functions,
profiling callback or reservoir buffers. Their advanced layouts remain the
existing wavefront specialization ABI, not an automatic extension binding map.
The basic lighting component has no camera, image extent or output buffer.
It provides existing BSDF sampling/evaluation, analytic/area/environment light
sampling and ray-query visibility functions.

`OL_TRANSPORT_RAY_ORIGIN(hit, normal)` can override shadow-ray offsets using the
intersection normal while passing a different shading normal to scattering.
The default preserves GI's existing normal offset. `contracts` defines
`OrdinaryLightSurfaceSample`, matching the 96-byte `SURFACE_SAMPLE_DTYPE`:
position, geometric normal, shading normal, incoming direction, application
identity and explicit inside/outside medium IDs. Medium membership uses the
geometric normal; it is never inferred from a shading normal. This is a contract,
not an automatic conversion from an SDF or voxel connectivity graph.

The `volumes` component exposes existing scalar-volume transport with
`WAVE_VOLUME_*_BINDING` macros for its five bindings (four buffers and the 16-entry
3-D sampler array). It retains existing opt-in overlap/scattering/empty-space
flags and the volume-header ABI. It does not construct contiguous dielectric
regions or implement arbitrary custom intersections.

Run [the surface-sample example](../examples/runtime_surface_samples.py) for a
concrete non-camera algorithm: it uploads a scene, samples lighting at two
application-defined surfaces, and writes radiance to reversed application IDs.
Its untextured/vacuum adapters are intentional fixture restrictions; they must
be replaced for textured surfaces or media.

## Application history and external HDR

`SampleHistory` stores values under arbitrary hashable identities. Each entry
lists invalidation domains. For example, diffuse voxel history can depend on
`geometry/materials/lights`, and specular history additionally on `camera`.
`invalidate("camera")` then preserves diffuse history. `identities=` restricts
invalidation to a local set. Returned retired values still belong to the
application; retire GPU resources after their last completion. The store does
not guess scene revisions or free resources implicitly.

```python
from ordinarylight.runtime import VulkanOutput

with VulkanOutput(runtime) as output:
    with output.tone_map(hdr_image, after=producer_completion) as frame:
        # frame.image stays on the GPU; frame.completion describes readiness.
        ...
    output.present(hdr_image, after=producer_completion,
                   surface_size=(width, height))
```

The producer supplies a same-runtime linear RGBA32F storage image and mandatory
completion. The tone-mapping SPIR-V is packaged, so this service does not require
a GLSL compiler at runtime. Tone mapping applies the existing ACES approximation, linear-to-sRGB
conversion, and opaque alpha into RGBA8. `VulkanOutput` does not initialize or
enter GI. `read(frame)` is an explicit diagnostic readback. Native presentation
uses a GPU blit into a compatible RGBA/BGRA UNORM surface, with no CPU pixel copy.
The initial presentation path waits synchronously and recreates outputs per call;
it is not yet a cached frames-in-flight replacement for the scientific viewers'
existing optimized presentation paths.

Create a runtime with `glfw_window=...`, or paired
`external_instance=.../external_surface=...`, for presentation. Integer Qt handles
are accepted. `VulkanSurfacePresenter(..., runtime=runtime)` can also borrow that
same surface runtime, but two active presenters must not manage its swapchain
concurrently.

With `VulkanRuntime(headless_surface=True)`, `output.export(hdr, after=...)`
returns the existing `GpuFrame` contract: dedicated opaque-FD RGBA8 memory,
ready semaphore and import metadata. The image is released to the external queue
family before signaling readiness. External consumers must finish before
closing the frame. This initial independent path does not offer release
semaphores or NV12/P010 conversion; those existing GI output facilities remain
available through their original API. No cross-runtime/cross-device image import
is implied by passing an HDR image.

## Validation

`tests/test_runtime_extensions.py` contains CPU contract checks and opt-in GPU
checks (`ORDINARYLIGHT_TEST_VULKAN_RUNTIME=1`). GPU tests cover non-camera indexed
lighting with GI construction prohibited, external-HDR tone-map values and FD
export, and shared scene/runtime lifetimes. Existing scientific regression
fixtures remain in `lattice-test/rt_viewer` and `lattice-test/geometry_viewer`.

Validation on the RTX 4070 Laptop GPU for this extraction:

- Core suite: 540 passed, 34 skipped, 74 subtests passed.
- Extension GPU checks: application-ID radiance, HDR color values/FD export,
  and rendering from a borrowed scene; standalone native HDR presentation and
  resize also passed. FD tests do not validate a CUDA/NVENC consumer.
- All 440 rebuilt wavefront SPIR-V variants matched existing binaries exactly.
- DMC RT histograms matched the portable reference exactly on GI, Vulkan Raster,
  and WebGPU: 439/73 baseline, 435/77 with changed NDT, 337/175 with changed
  parameter basis (binned/excluded, 512 trials per scenario).
- Likelihood viewer: 11 tests and live initial/update/capture smoke checks on
  all three targets passed. Native Vulkan presentation retains its no-CPU-pixel
  path; WebGPU retains its explicitly labeled fallback.

The raster integration initially reproduced a native crash. Vulkan subpass
attachment pointers were being freed before render-pass creation by the Python
binding; keeping their owning structs alive fixed the failing RT regression.
