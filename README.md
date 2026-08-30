# Ordinary Light

`ordinarylight` is a general-purpose path tracer with a renderer-neutral Python
scene API. Its current public surface includes:

- meshes, instances, materials, textures, lights, volumes, and hierarchical
  animation resources;
- perspective, orthographic, and panoramic cameras;
- semantic loader, camera, light, animation, output, renderer, target, and integration
  namespaces;
- a portable CPU reference renderer; and
- a Vulkan wavefront GI renderer using hardware ray queries, BLAS/TLAS
  acceleration structures, native presentation, and headless HDR readback.

## Repository layout

- `ordinarylight/`: installable renderer package and bundled shaders
- `examples/`: supported examples, plus explicitly separated legacy experiments
- `tools/`: developer presentation harnesses and low-level diagnostics
- `tests/`: unit tests and opt-in GPU gates
- `artifacts/`: retained historical benchmark evidence; not runtime data
- `assets/`, `shaders/`, and `scripts/`: fixtures and build tooling

The installable package is organized by domain rather than as a collection of
flat modules. Public models live in focused namespaces such as
`ordinarylight.cameras`, `lights`, `materials`, `scene`, `volume`, and
`effects`; renderer algorithms live in `ordinarylight.renderers`; execution
APIs live in `ordinarylight.targets`; shader sources and compilation helpers
live in `ordinarylight.shaders`. Each namespace re-exports its normal public
API, and the top-level `ordinarylight` exports remain available for concise
application code.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[vulkan]'
```

Inspect Vulkan support:

```bash
python -m ordinarylight
```

On a suitable RTX system the selected adapter must be a hardware adapter and
advertise `VK_KHR_acceleration_structure`, `VK_KHR_deferred_host_operations`,
and `VK_KHR_ray_query`.

Render the reference image:

```bash
python -m examples.reference
```

The core package can be installed without Vulkan for loaders, scene tooling,
the reference backend, and headless API development. See
[`docs/getting_started.md`](docs/getting_started.md) for supported examples.

This writes `/tmp/ordinarylight_reference.ppm`, which most image viewers can
open. Pass `--output PATH` to choose another location.

## Python rendering API

Application and notebook code can use the high-level renderer without knowing
about wavefront dispatches, swapchains, or Vulkan resource ownership:

```python
import matplotlib.pyplot as plt
import ordinarylight as ol

scene = ol.loaders.gltf.load("assets/TransmissionTest.glb")
camera = ol.PerspectiveCamera(
    position=(0.0, 1.5, -5.0),
    target=(0.0, 1.0, 0.0),
)

with ol.Renderer(max_bounces=8, samples_per_pixel=4,
                 wavefront_execution_strategy="auto") as renderer:
    hdr = renderer.render(scene, camera, (1280, 720))

# RGB is linear HDR radiance. Display mapping stays explicit and belongs to
# the application, which is important for quantitative scientific colors.
plt.imshow(hdr[..., :3].clip(0.0, 1.0))
```

`Renderer.render()` returns a `(height, width, 4)` `float32` NumPy array and
automatically advances its deterministic sampling sequence. Use
`reset_sequence()`, an explicit `frame_index=`, or a caller-owned `out=` array
for reproducible capture and allocation control. `device`, `last_timings`, and
the immutable `last_statistics` record provide introspection without exposing
private renderer objects. The lower-level
`renderers.gi.VulkanGlobalIlluminationRenderer` and `VulkanGlfwPresenter` remain available for
specialized validation and direct-window applications.

### Renderer and target organization

Concrete algorithms live exclusively beneath `ordinarylight.renderers`:

- `renderers.gi.VulkanGlobalIlluminationRenderer`
- `renderers.raster.VulkanRasterRenderer`
- `renderers.raster.WebGpuRasterRenderer`
- `renderers.hybrid.HybridRenderer`
- `renderers.reference.CpuReferenceRenderer`

Each derives from `RendererImplementation` and publishes immutable
`implementation_info()` metadata describing its renderer family and execution
API. Third-party renderers do not need to inherit this class: the public
`RendererProtocol` protocol remains structural.

Platform APIs are a separate concern exposed through `ordinarylight.targets`.
For example, `targets.vulkan` owns Vulkan device discovery, configuration, and
native presenters, while `targets.webgpu` and `targets.cpu` identify their
shader format and execution properties. There is intentionally no
`ordinarylight.backends` namespace: renderer algorithms and execution targets
should not be conflated.

Vulkan renderer selection is explicit and inspectable. The default `"auto"`
preference chooses hardware ray-query global illumination when available and
uses Vulkan rasterization as its compatibility fallback. An explicit request
never silently changes renderer class:

```python
with ol.Renderer(renderer_preference="auto") as renderer:
    print(renderer.renderer_selection)
    # requested, selected, fallback, and a fallback reason when applicable

with ol.Renderer(renderer_preference="gi") as renderer:
    ...  # fail rather than fall back if GI is unavailable

with ol.Renderer(renderer_preference="raster") as renderer:
    ...
```

Release wheels contain CI-compiled SPIR-V and WGSL for the built-in raster
program, so raster fallback does not import Ordinary Shade at runtime. Python
shader definitions remain authoritative: a source checkout whose definitions
are newer than its artifacts recompiles them with the companion project rather
than silently using stale output.

Raster and GI share Ordinary Light's scene-level API and are kept semantically
equivalent where the techniques overlap. Indirect illumination, physically
traced reflection/refraction, and multiple scattering are intentionally
GI-specific rather than raster parity requirements.

For GUI event loops, notebooks, and output pipelines, submit without blocking
the calling thread. Submissions made through one renderer remain ordered and a
`RenderJob` can be polled, given a completion callback, waited on, or awaited:

```python
job = renderer.render_async(scene, camera, (1280, 720))
while not job.done():
    service_application_events()
hdr = job.result()

# In asyncio code:
hdr = await renderer.render_async(scene, camera, (1280, 720))
```

For correctness tests or systems without Vulkan ray tracing, select the
portable CPU renderer explicitly while retaining the same high-level contract:

```python
implementation = ol.renderers.reference.CpuReferenceRenderer(
    samples_per_pixel=4, max_bounces=4,
)
with ol.Renderer(implementation=implementation) as renderer:
    hdr = renderer.render(scene, camera, (640, 360))
```

Headless stills are NumPy arrays. Compressed video remains an output concern
and does not add a GUI dependency:

```python
with ol.outputs.FFmpegVideoWriter("render.mp4", (width, height), fps=30) as video:
    for camera in cameras:
        video.write(ol.outputs.to_sdr(renderer.render(scene, camera, (width, height))))
```

The video writer also accepts a binary file-like object for notebook or server
streaming. FFmpeg is optional and checked only when a writer is created.

NVIDIA systems can keep the complete H.264 path on the GPU. Install
`ordinarylight[video-gpu]`, enable external interop, request NV12, and pass the
managed frame directly to NVENC:

```python
config = ol.RendererConfig(external_image_interop=True)
with ol.Renderer(config=config) as renderer, \
     ol.outputs.NvencVideoWriter("render.h264", (width, height), fps=30) as video:
    for index, camera in enumerate(cameras):
        video.write(renderer.render_gpu(
            scene, camera, (width, height), frame_index=index,
            pixel_format="nv12",
        ))
```

Vulkan performs tone mapping and NV12 conversion, CUDA imports the stable
external buffers and semaphore pairs once, and NVENC consumes the two planes
without host readback or upload. A binary file-like destination can be used in
place of the path for server streaming. See `examples/nvenc_zero_copy.py` and
the ownership details in `docs/public_api.md`.

For 10-bit HEVC, select P010 on both sides of the contract:

```python
with ol.outputs.NvencVideoWriter(
    "render.h265", (width, height), codec="hevc", pixel_format="p010"
) as video:
    video.write(renderer.render_gpu(
        scene, camera, (width, height), pixel_format="p010"
    ))
```

P010 is tone-mapped directly from the linear HDR render target into BT.709
limited-range 10-bit codes in the high bits of 16-bit samples. It remains
GPU-resident through Vulkan conversion, CUDA import, and NVENC.

Long-running streams can insert recovery points without rebuilding NVENC or
the Vulkan/CUDA interop pool. This emits an IDR with repeated codec headers
every two seconds and also supports an immediate application request:

```python
with ol.outputs.NvencVideoWriter(
    stream, (width, height), fps=30, bitrate="6M",
    keyframe_interval_seconds=2.0,
    repeat_headers_on_keyframe=True,
) as video:
    video.write(frame)
    video.request_keyframe()  # applies to the next successful frame
    video.write(next_frame)

# Or mark a particular frame directly:
video.write(frame, force_idr=True, repeat_headers=True)
```

`forced_keyframe_count` reports IDRs requested through the periodic, queued,
or per-frame controls. Failed encode attempts do not consume a queued request.

Scene changes do not require reconstructing the renderer. For long-running
viewers and video streams, use `renderer.replace_scene(next_scene)` to retain
the Vulkan device, pipelines, and external video buffers. Runtime samples,
bounces, exposure, and render scale can be updated with
`renderer.reconfigure(...)`; structural changes explicitly require recreation.

Asset ingestion is organized under `ordinarylight.loaders`. Format modules own a
uniform `load()` entry point, so glTF is available as
`ol.loaders.gltf.load(path)`; the descriptive
`ol.loaders.load_gltf(path)` and historical `ol.load_gltf(path)` aliases refer
to the same function. Future format support belongs in this namespace rather
than in the renderer or scene modules.

Native applications can use the optional high-level viewport without pulling
windowing into the core import path:

```python
from ordinarylight.integrations.glfw import NativeViewport

with NativeViewport(scene, camera, config=ol.RendererConfig()) as viewport:
    viewport.run()
```

Call `viewport.step()` from an existing application loop, or supply a
`controller(viewport, dt)` callback to update the camera and scene before each
frame. Both modes expose the latest immutable `RenderStatistics` through
`viewport.last_statistics`.

Explicit named outputs return an immutable mapping-like `RenderFrame`. Color is
linear HDR RGBA; variance is unbiased per-pixel luminance variance across the
samples in that render call:

```python
frame = renderer.render(
    scene, camera, (width, height), samples=8,
    outputs=("color", "variance", "depth", "normal",
             "instance_id", "material_id"),
)
hdr = frame.color                 # float32 [height, width, 4]
variance = frame["variance"]      # float32 [height, width]
print(frame.metadata["timings"])
```

`renderer.available_outputs` provides capability discovery. Depth, normals,
and variance are renderer products rather than values inferred from the final
image. `depth` is positive primary-ray distance with `inf` for background;
`normal` is the unit world-space geometric normal with zero for background.
Both primary-hit products use sample zero when rendering multiple samples.
`instance_id` contains the stable `Instance.id`; `object_id` is its compatibility
alias. `material_id` contains a stable scene-local identity shared by instances
using the same `Material` object. These are `uint32` images and use `0xffffffff`
for background. These products support
general renderer workflows such as picking, compositing, editor selection,
debugging, and segmentation. `motion` is a `(height, width, 2)` `float32`
current-minus-previous screen displacement in output pixels. It accounts for
camera motion, rigid transforms, and equal-topology vertex deformation between
successive calls that request motion. The first compatible call, background,
new objects, and topology changes produce zero motion. `reset_sequence()` also
clears motion history.

Use `renderer.capabilities` for renderer-neutral feature and limit discovery:

```python
capabilities = renderer.capabilities
capabilities.require("hardware_ray_tracing", "volumes")
if capabilities.supports_output("motion"):
    outputs = ("color", "motion")
print(capabilities.renderer, capabilities.device, capabilities.limits)
```

Capability names describe renderer behavior rather than Vulkan extensions, so
application code does not need backend-type checks.

Instances and point lights returned by `Scene` are stable resource handles with
monotonic integer IDs. Mutate them through the scene so validation, revision
tracking, accumulation invalidation, and resident GPU scene replacement happen
automatically:

```python
mesh = scene.add_mesh(vertices, indices, material)
scene.update_mesh(mesh, vertices=next_vertices)
scene.update_mesh(mesh.id, material=next_material)
scene.update_mesh(
    mesh,
    transform=(
        ol.Transform.translation((1, 2, 3))
        @ ol.Transform.rotation((0, 1, 0), angle)
        @ ol.Transform.scale(2)
    ),
)
scene.remove_mesh(mesh)

light = scene.add_point_light((0, 3, 0), intensity=10)
scene.update_point_light(light, intensity=20)
```

Repeated geometry should use an explicit mesh resource and lightweight
instances. Each instance has an independent transform, visibility, material,
and stable ID, while Vulkan builds one BLAS for the shared object-space data:

```python
geometry = scene.create_mesh(vertices, indices, default_material)
instances = [
    scene.add_instance(
        geometry,
        transform=ol.Transform.translation((x, 0, 0)),
        material=material,
    )
    for x, material in placements
]

# Validation is atomic and the scene revision advances only once. The resident
# backend performs one packed upload and one TLAS refresh for this batch.
scene.update_instances([
    (instance, {"transform": transform, "material": material})
    for instance, transform, material in updates
])

print(scene.instancing_statistics())
scene.remove_instance(instances[0])
```

For large placement sets, use the column-oriented API instead of constructing
one changes dictionary per object:

```python
# transforms has shape (instance_count, 4, 4).
instances = scene.add_instances(
    geometry, transforms,
    materials=materials,
    names=names,
)

# A single validated scene revision, packed GPU upload, and TLAS refresh.
scene.update_instance_transforms(instances, next_transforms)
scene.update_instance_batch(
    instances, transforms=next_transforms, materials=next_materials,
)

# JSON-compatible structure and identity metadata; geometry arrays stay out of
# the snapshot so inspection does not duplicate large datasets.
description = scene.snapshot()
```

`add_mesh()` remains the convenient one-off form and internally creates a mesh
resource plus one instance. A legacy geometry edit to one of several shared
instances uses copy-on-write; use `update_mesh_resource()` when every instance
should receive the new object-space data. Hidden instances retain their IDs but
are omitted from packed render data and the TLAS.

glTF loading uses the same model. When several glTF nodes reference one mesh
primitive, `load_gltf()` decodes one `MeshResource` and creates independently
named/transformed instances for those nodes. Source mesh, primitive, node,
parent-node, and `extras` metadata are retained in `scene.snapshot()`.
Sparse accessors, normalized skin weights, inverse-bind matrices, morph
targets, node hierarchies, and linear/step/cubic animation tracks are imported
without flattening their semantics. Use `scene.animations` with
`ol.AnimationPlayer`, or call `scene.apply_animation(clip, time)` directly.
Required unsupported glTF extensions fail explicitly; optional unsupported
extensions are recorded in `scene.metadata` for application diagnostics.

## Point, line, and glyph batches

Finite world-space points and line segments use the same general scene model.
The backend lowers each point batch to one shared sphere mesh and each line
batch to one shared capped-cylinder mesh. Arbitrary glyphs reuse a caller-owned
`MeshResource`; no renderer feature is specialized for a particular domain:

```python
points = scene.add_points(
    positions, radii=radii, materials=point_materials,
)
lines = scene.add_lines(
    segment_starts, segment_ends, radii=line_widths,
    material=line_material,
)
glyphs = scene.add_glyphs(glyph_geometry, glyph_transforms)

# IDs and instance handles remain stable across column-oriented animation.
points.update(positions=next_positions, radii=next_radii)
lines.update(starts=next_starts, ends=next_ends)
glyphs.update(transforms=next_glyph_transforms)

lines.remove()
```

These primitives participate in all ordinary material shaders, acceleration
structure sharing, visibility, `instance_id`/`material_id` outputs, snapshots,
and accumulation invalidation. They are finite 3D geometry; screen-space marker
and line-width policies belong in a higher-level integration.

Select **Points, lines, and glyphs** in `ordinarylight-workbench`, or run its
combined structure/4K performance gate:

```bash
ordinarylight-workbench
tests/gates/run_4k_primitives.sh
```

## Structured volumes

Dense scalar fields are first-class scene resources. Arrays are copied into an
immutable `(depth, height, width)` float32 resource, normalized through an
explicit value range, and sampled inside a transformed local unit cube. An
RGBA `Texture1D` supplies the transfer function: RGB is emitted radiance and
alpha is reference opacity per material step.

```python
transfer = ol.Texture1D([
    (0.0, 0.0, 0.0, 0.0),
    (0.1, 0.5, 1.5, 0.04),
    (3.0, 0.8, 0.1, 0.35),
])
volume = scene.add_volume(
    density,
    ol.VolumeMaterial(
        transfer, density_scale=1.0, emission_scale=1.2,
        step_size=0.01,
    ),
    transform=ol.Transform.translation((-1, 0, -1))
              @ ol.Transform.scale((2, 3, 2)),
    value_range=(0.0, 1.0),
    name="density",
)
scene.update_volume(volume, data=next_density)
```

Light-dependent media opt into scattering independently of emission:

```python
cloud = ol.VolumeMaterial(
    transfer,
    density_scale=1.0,
    emission_scale=0.0,
    scattering_scale=0.9,
    scattering_color=(0.65, 0.8, 1.0),
    phase_function="henyey_greenstein",
    anisotropy=0.55,
    # Order 1 is the direct estimator. Higher values enable the separate,
    # bounded multiple-scattering specialization.
    scattering_orders=4,
    scattering_albedo=(0.92, 0.94, 0.98),
    step_size=0.02,
)
```

`phase_function` accepts `"isotropic"` or `"henyey_greenstein"`; anisotropy is
bounded to `[-0.99, 0.99]`. The CPU reference and Vulkan volume marchers apply
inverse-square point-light illumination, a normalized phase response, and an
order-independent midpoint estimate of source-path attenuation through every
crossed medium. Vulkan additionally evaluates emissive triangles and a bounded
environment quadrature and traces opaque visibility for all three light
domains. Direct lighting is evaluated at one representative point per medium
interval, keeping visibility cost independent of density-march resolution.
Scattering is zero by default and its larger kernels are loaded lazily, so
existing surface, emission--absorption, and order-1 scattering scenes retain
their prior shader and cost. Orders 2--8 use an opt-in, energy-bounded local
multiple-scattering closure driven by medium optical depth and RGB scattering
albedo. This efficiently restores energy lost by a direct-only estimator, but
is intentionally not presented as an unbiased spatial volumetric random walk.

Select **Volume multiple scattering** in `ordinarylight-workbench`, or run
`python -m tests.gates.volume_multiple_scattering` for its native HDR and
wavefront/megakernel parity gate.

The CPU reference backend and native Vulkan wavefront backend both implement
front-to-back emission--absorption integration. Vulkan uses ray-traced unit-box
entry proxies, linearly sampled device-local 3D images, and packed
transfer-function buffers; volume data is not tessellated into visualization
geometry. Surface visibility and volume integration use independent ray-query
masks, so an empty medium is exactly invisible while an absorbing medium
attenuates point, area, environment, and ReSTIR direct-light shadow rays.
Sparse volumes have an opt-in conservative 8-voxel brick
occupancy. Brick ranges include the complete trilinear-filtering halo and are
classified against the active transfer function, so skipping is exact rather
than an opacity approximation. Its correctness gate passes, but the current
sparse showcase measured a small performance regression, so it remains
disabled by default while the traversal is redesigned. Dense volumes retain
the ordinary marcher and do not load the skipping specialization. Use
`ol.volume_empty_space_statistics(scene.visible_volumes)` to inspect occupancy,
or set `RendererConfig(volume_empty_space_skipping=True)` for an explicit A/B.
Overlapping media add their extinction coefficients and mix emission by optical
depth, making the result independent of scene insertion order. Their larger
marcher is loaded lazily as a dedicated shader specialization, preserving the
ordinary and single-volume kernels. It is available to the wavefront, hybrid,
megakernel, and persistent execution strategies; SER currently rejects scenes
with overlapping volumes explicitly. Volumes compose with meshes, lights,
instancing, stable `instance_id` output, snapshots, transforms, and
accumulation invalidation.

Use `ordinarylight-workbench` for interactive volume showcases. The remaining
commands are automated regression/performance gates:

```bash
ordinarylight-workbench
python -m tests.gates.volume_scattering
python -m tests.gates.volume_empty_space
tests/gates/run_volume.sh
tests/gates/run_volume_compositing.sh
tests/gates/run_volume_empty_space.sh
```

The **Multiple volumes** workbench showcase presents three partially overlapping media with
distinct transfer functions alongside opaque reference geometry. It uses the
normal native presentation path and reports FPS and GPU time in the title bar.
The **Volume scattering** workbench showcase isolates non-emissive, forward-scattering fog under
two colored point lights so phase directionality and source-path attenuation
can be inspected while the camera orbits.

The compositing gate specifically guards empty-volume parity at a
surface/volume overlap, wavefront/megakernel HDR parity, volumetric shadow
attenuation, and insertion-order parity for overlapping media. Pass
`--overlap-only` for the focused overlap capture.

`revision`, `geometry_revision`, and `shading_revision` allow integrations to
detect exactly which class of state changed; `transform_revision` independently
tracks object-to-world motion. Object vertices remain unchanged, while bounds,
normals, tangents, emissive geometry, CPU reference rendering, and Vulkan input
all use transformed world data. Material, light, and vertex-attribute-only
updates reuse existing acceleration structures and perform one batched partial
GPU-buffer upload when layouts are unchanged. Vulkan keeps one object-space
BLAS per distinct mesh resource and a TLAS record per visible non-empty
instance. Transform-only updates therefore preserve every BLAS, update the
instance data, and rebuild the TLAS
in its existing allocation. Equal-topology vertex deformation updates only the
affected object/world vertex data, refits those mesh BLAS in place, and refreshes
the existing TLAS when the mesh was created with `deformable=True`. Keeping this
capability explicit lets static meshes retain the fastest traversal layout.
Topology, texture-set, and buffer-size changes still
conservatively rebuild the affected resident scene resources.

The **Dense geometry** workbench scene is the instancing showcase and
regression workload. This gate
checks the 4K 50 FPS target and verifies that its forty spheres share a BLAS:

```bash
ordinarylight-workbench
tests/gates/run_4k_instancing.sh
```

The general-purpose renderer API direction, including the requirements needed
by downstream visualization packages, is documented in
[`docs/scientific_api.md`](docs/scientific_api.md). Domain-specific datasets,
plot types, and analysis remain outside this project.

## Dear PyGui surface adapter

Install the optional adapter when integrating with an existing Dear PyGui
application:

```bash
pip install -e '.[gui]'
```

`DearPyGuiSurface` owns a resizable dynamic texture and an image widget.
Any renderer implementing `render_to(...)` can present RGBA8 results to it. The
application should recreate the texture and rerender after viewport resizing
settles, so the render resolution tracks the available UI area.

```python
from ordinarylight.integrations.dearpygui import DearPyGuiSurface

surface = DearPyGuiSurface(800, 450, parent="render_window")
renderer.render_to(scene, camera, surface, samples=4)
```

Run tests:

```bash
python -m unittest discover -s tests -v
```

## Native GLFW integration

Install GLFW when embedding Ordinary Light in a native window application:

```bash
pip install -e '.[window]'
```

Use the Qt workbench for interactive feature inspection:

```bash
ordinarylight-workbench
```

Library callers can configure the renderer without process-global environment
variables and explicitly upload the resident GPU scene:

```python
config = ol.RendererConfig(
    max_bounces=8,
    present_mode="mailbox",
    swapchain_images=6,
    present_pacing=False,
)
with ol.VulkanGlfwPresenter(window, config=config) as presenter:
    presenter.upload_scene(scene)
    while not glfw.window_should_close(window):
        width, height = glfw.get_framebuffer_size(window)
        presenter.present(scene, camera, width, height)
```

`upload_scene()` is optional on the first frame. Passing a different `Scene`
to `present()` safely replaces the resident GPU scene after outstanding work
completes.

Load the included Khronos CC0 transmission test scene from the workbench's
**Load glTF / GLB…** action:

```bash
ordinarylight-workbench
```

`ordinarylight.load_gltf(path)` loads GLB or glTF 2.0 triangle primitives,
resolves active-scene node transforms, and imports base color, emission,
metallic, roughness, transmission, IOR, and volume attenuation parameters.
The direct Vulkan shader uses per-triangle materials and traces secondary
transmission and metallic-reflection rays. Texture sampling, stochastic
roughness, and progressive accumulation are not implemented yet.
Set `WAVE_RENDER_MAX_BOUNCES=N` to select one through sixteen ray segments per
pixel in the direct Vulkan demo; the default is five.

### Qt feature workbench

An optional Qt workbench provides scene loading, switching and unloading,
camera animation, renderer feature controls, accumulation reset, and live
timing statistics:

```bash
pip install -e '.[qt]'
ordinarylight-workbench

# Equivalent from a source checkout:
python -m ordinarylight.integrations.qt_workbench
```

The render surface is embedded directly beside the controls. On Linux, the
current PySide wheels do not consistently expose `QVulkanInstance`, so the demo
creates an undecorated GLFW Vulkan child and adopts its X11 window into Qt with
`QWindow.fromWinId()`. Both toolkits use XWayland in this mode. Presentation is
still directly to the Vulkan swapchain with no CPU pixel copy. Applying
settings recreates the renderer because `RendererConfig` is immutable;
switching scenes reuses it. The public `VulkanSurfacePresenter` also accepts an
externally owned `VkInstance` and `VkSurfaceKHR` for toolkits that expose those
handles directly.

Left-click a visible mesh or volume in the viewport to select it. The portable
picker is independent from that demonstration and is available as
`ordinarylight.pick(scene, camera, viewport_size, pixel)`. It returns a
`PickResult` with a stable object ID, object reference, world-space position,
triangle, and barycentric coordinates. Application code decides what the hit
means and can load metadata, modify the scene, update another UI, or apply a
visual effect:

```python
hit = ordinarylight.pick(scene, camera, viewport_size, pixel)
if hit is not None:
    metadata_panel.show(load_metadata(hit.object_id))
    presenter.apply_object_effect(
        scene, hit.object_id,
        ordinarylight.effects.Outline(color=(1.0, 0.45, 0.05), width=2),
    )
else:
    presenter.clear_object_effect()
```

Applications using the high-level renderer can submit a compact resident-TLAS
query without blocking their event loop:

```python
job = renderer.pick_async(
    scene, camera, framebuffer_size, ui_pixel,
    mapping=ordinarylight.ViewportMapping(
        ui_size,
        framebuffer_size=framebuffer_size,
        render_size=internal_render_size,
        content_rect=displayed_content_rect,
    ),
    options=ordinarylight.PickOptions(
        transmissive="surface", volumes="include",
    ),
)
job.add_done_callback(lambda completed: handle_pick(completed.result()))
```

The Vulkan default policy dispatches a single pixel against the already
resident TLAS and reads back only its compact hit record. Through-glass and
volume-filtering policies currently use the portable CPU traversal because
they may need to continue beyond a rejected first hit. `ViewportMapping`
handles high-DPI framebuffer scaling, letterboxing, and dynamic internal
resolution, and returns no hit for UI pixels outside displayed content.

The optional object-effect layer is enabled with
`RendererConfig(object_effects=True)`. `Outline`, `Tint`,
`EmissiveHighlight`, `Isolation`, `BoundingBox`, and `XRay` run in Vulkan
reconstruction without pixel readback and therefore appear in both window and
encoded output. `BoundingBox` and `XRay` use projected object bounds so they
remain visible through occluders. Up to four effects can be active at once:

```python
renderer.set_object_effects(scene, (
    (hovered_id, ordinarylight.effects.Outline(color=(1, 1, 1))),
    (selected_id, ordinarylight.effects.Tint(color=(1, 0.3, 0.1))),
    (tracked_id, ordinarylight.effects.XRay(color=(0.1, 0.8, 1))),
))
```

The singular v0.3.4 `apply_object_effect()` and `clear_object_effect()` calls
remain supported. Picking never creates selection state or implicitly applies
an effect. Effect changes serialize with rendering and GPU-video submissions,
so remote applications do not need presenter or Vulkan access. No callback
framework is imposed: ordinary Python code handling `PickResult` is the
arbitrary action API.

The display-space color functions used by the built-in effects and the shared
ACES transform used by reconstruction, tone mapping, and P010 output are
generated from typed Python helpers by the independent Ordinary Shade companion
project. Generated GLSL is checked into Ordinary Light,
so installing or running the renderer does not require the compiler. Ordinary
Shade remains an optional development-time dependency and has no dependency on
Ordinary Light.

`wavefront_indirect_clear.comp`, `wavefront_prepare_indirect.comp`,
`wavefront_resolve.comp`, `wavefront_tone_map.comp`, and
`wavefront_tone_map_image.comp`, plus `rgba_to_nv12.comp`, `hdr_to_p010.comp`,
`denoise_atrous.comp`, the final display `tone_map.comp`, and primary-ray
`wavefront_generate.comp` and `wavefront_intersect.comp` are complete compute
entry points authored in Ordinary Shade, together with the subgroup-compacted
`wavefront_intersect_bucketed.comp` variant and the packed-reservoir
`wavefront_indirect_reuse_probe.comp`. HDR path accumulation and production
indirect-reservoir seeding in `wavefront_path_to_hdr.comp` are generated as
well, as is temporal/spatial reservoir validation and merging in
`wavefront_indirect_candidates.comp`. Their generated
Vulkan GLSL retains the original descriptor bindings, storage layouts, access
qualifiers, push-constant offset, and workgroup sizes. The second stage uses
Ordinary Shade's fixed structured storage records for queue state and indirect
dispatch arguments. The resolve stage adds typed structured runtime arrays,
write-only output, vector swizzles, and integer-vector construction.
The tone-map stage demonstrates composable typed helper functions: its entry
point links the same Ordinary Shade ACES and sRGB helpers used to generate the
shared color library, then lowers packed RGBA output per backend.
The image-output variant adds portable storage-image dimensions and dynamically
bounded nested loops while preserving arbitrary source-to-output scaling.
The NV12 stage adds reusable helper graphs, signed constants, rounding, shifts,
and bitwise packing while preserving the streaming backend's 4×2 conversion
and pitched two-plane memory layout.
The P010 stage reuses that layout model with shared HDR color helpers, scaled
source sampling, 10-bit limited-range conversion, and MSB-aligned 16-bit pairs.
The à-trous denoiser adds image ping-pong, variance gating, nested fixed-range
sampling, loop continuation, and normal/depth/color bilateral weighting without
changing its descriptor or push-constant ABI.
The display stage adds short-circuit boolean expressions, conditional
expressions, loop termination, and a composed FPS glyph overlay while retaining
the native 8×8 output path.
Primary-ray generation adds typed heterogeneous storage blocks with runtime
structure arrays, nested queue records, uniform camera data, projection modes,
and portable geometry/trigonometry helpers. Its medium-stack binding uses an
equivalent flat scalar view of the existing 64-byte-per-path layout.
The intersection stage adds a typed Vulkan acceleration-structure resource,
ray-query lifecycle and committed-hit accessors, and structured hit-queue
output. Ordinary Shade rejects this Vulkan-only feature for WGSL with an
explicit diagnostic.
The bucketed variant adds declared subgroup capabilities, ballot masks,
elected-lane atomics, broadcast and exclusive-bit offsets, typed material
records, and structured local values. Its Vulkan subgroup extensions are
emitted only when the entry point declares that requirement.
The indirect-reuse probe adds nested structure values, resource-sharing helper
functions, `void` helpers, half/unorm packing, RGB9E5 conversion, and explicit
storage-buffer barriers. It validates the production six-word reservoir ABI
without retaining the former preprocessor-dependent probe wrapper.
The path-to-HDR stage then applies those same typed reservoir structures and
packing helpers in production, preserving its six descriptor bindings,
representative-pixel downsampling, accumulation behavior, and finite-candidate
rejection.
The indirect-candidate stage adds typed unsigned storage images, reprojection
result values in place of output parameters, ray-query visibility testing,
bounded history merging, spatial-neighbor validation, and optional atomic
profiling while preserving its thirteen descriptor bindings.
`wavefront_indirect_debug.comp` completes the indirect-reuse group with typed
acceptance/validity visualizations and the material-aware, bilinearly filtered
reuse correction path. A structured correction result replaces the original
GLSL output parameters while retaining all five display modes.
`wavefront_reconstruct.comp` is also a complete generated stage. It composes
reprojection, history validation, diffuse filtering, tone mapping, and the
built-in object effects while preserving both the RGBA and formatless BGRA
output variants. Ordinary Shade models the eight swapchain outputs as a typed
storage-image descriptor array. The effect header is expressed as flattened
typed fields with the same byte layout as the former `uvec4[4]` push-constant
view, so the host ABI is unchanged.
The buffer-output `ray_query.comp` baseline is generated too, including camera
projection, multi-bounce reflection and transmission, medium tracking, custom
material injection, ray-query traversal, and packed output. Its generated
material insertion markers preserve runtime specialization by Ordinary Light's
existing Python material API.
Its accumulated `ray_query_image.comp` counterpart is generated as well. The
typed stage preserves point/spot/directional and area-light visibility,
multiple-importance sampling, primary-surface metadata, adaptive sampling,
ping-pong accumulation and moments, temporal reprojection, and the original
material-aware neighborhood history clamp.
`scripts/generate_core_shaders.py --check` guards the
checked-in sources while the shader manifest guards their compiled SPIR-V.

The final generated wavefront shade entry is the production default. Use
`RendererConfig(wavefront_ordinaryshade_shade=False)` or
`WAVE_RENDER_ORDINARYSHADE_SHADE=0` for an explicit handwritten A/B fallback.
The formal `shade_parity` matrix currently reports strict HDR numerical parity
across opaque, textured, emissive, glass, dense, and volume workloads,
including native textures and overlapping/scattering media. Steady-state shade
dispatch medians are at parity. Ordinary Light persists the Vulkan pipeline
cache by default, which removes most of the larger generated module's pipeline
creation cost after its first successful renderer lifetime. The first-ever
compile for a new GPU/driver combination remains cold. Deterministic Python
`MaterialEvaluation` programs now specialize both the handwritten primary
stage and the generated Ordinary Shade secondary stage, including interpolated
custom vertex attributes, native textures, profiling, and active volume
features. The formal `shade_material_parity` and `shade_attribute_parity` gates
verify the runtime specializations against production HDR output.
`SurfaceResponse` programs use the same staged ABI and may replace scattering
from the primary hit onward with an absorb, diffuse, reflection, or transmission
event plus a Python-authored direction, weight, PDF, and emission. Ordinary
Light retains ownership of continuation queues, medium-stack transitions,
Russian roulette, synchronization, and resource lifetime. The formal
`shade_surface_parity` gate exercises diffuse, mirror, and stochastic nested
Fresnel-glass programs against the handwritten fallback.

The fused-primary migration has begun at its shared camera and path-state
boundary. Perspective, orthographic, and panoramic ray construction, primary
RNG seeding and jitter, initial path identity/flag packing, barycentric hit
reconstruction, geometric and interpolated shading normals, face orientation,
ray-cone spread, and surface classification are generated by
`scripts/generate_primary_shaders.py` and consumed by wavefront, hybrid,
megakernel, persistent, and SER entry points. Primary UV/tangent interpolation,
UV-density footprints, generated triangle tangents, mapped-normal hemisphere
correction, sampled base/emissive/transmission/metallic/roughness/occlusion
application, and tangent-space normal transformation share that generated
boundary. Thin GLSL resource wrappers retain packed/native texture sampling,
profiling counters, and their descriptor ABI. Primary valid/invalid G-buffer
payloads, one- and two-sided emission policy, maximum-bounce termination, and
active-path flag clearing are generated as typed output-state helpers; image
writes and queue storage remain explicit renderer resource operations. Initial
glass handling is typed as well: opaque/transmissive classification, clamped
target IOR, refraction with total-internal-reflection fallback, first medium
entry/depth, medium-stack initialization, and transmission tint/weight.
Post-sampling primary continuation state is generated too: BSDF throughput
application, roughness-driven ray-cone growth, direction normalization,
medium/capture/diffuse/unified-NEE flag packing, previous-PDF retention,
secondary-capture eligibility, continuation-ray offsets, and queue payloads.
The generated Vulkan orchestration boundary additionally owns secondary ray
queries, descriptor reads, medium/path storage, volume and profiling callbacks,
subgroup-compacted continuation queues, custom-material dispatch, group
swizzling, and the persistent-coarse shared-memory scheduler. Those paths
exercise Ordinary Shade's typed external ABI, ray-query capabilities,
workgroup storage, atomics, barriers, and Vulkan invocation reordering; the
handwritten equivalents remain compile-time diagnostic fallbacks.
The enclosing secondary-transport loop is Python-authored as well. It owns the
stop bound, iteration, termination signal, mutable path/medium/RNG/cone state,
and final RNG persistence, while one typed callback supplies the renderer's
per-bounce Vulkan adapter. This keeps portable control composition in Ordinary
Shade without forcing descriptor and backend ownership into the companion
compiler.
The PBR sampler and direct-light estimators remain independent compositional
modules rather than being folded into this state boundary. Point, directional,
and spot lights now use a typed analytic-light module for direction, inverse-
square/range and cone attenuation, surface cosine, shadow distance, incident
radiance, and PBR contribution. Descriptor iteration, visibility ray queries,
and volume-shadow lookup remain thin Vulkan orchestration, with transmittance
passed explicitly into the generated calculation. The PBR sampling
module is now typed too: cosine-hemisphere and GGX half-vector sampling,
Schlick Fresnel, GGX distribution/Smith masking, mixed diffuse/specular PDFs,
BRDF evaluation, reflection, occlusion weighting, and invalid-sample fallback.
The GLSL wrapper owns only the inout RNG sequence and output assignment.
The previous GLSL branches remain available at compile time through
`WAVE_ORDINARYSHADE_PRIMARY_CAMERA=0`,
`WAVE_ORDINARYSHADE_PRIMARY_STATE=0`, and
`WAVE_ORDINARYSHADE_PRIMARY_SURFACE=0`, plus
`WAVE_ORDINARYSHADE_PRIMARY_TEXTURE_STATE=0` and
`WAVE_ORDINARYSHADE_TEXTURE_APPLICATION=0`, plus
`WAVE_ORDINARYSHADE_PRIMARY_OUTPUT=0` and
`WAVE_ORDINARYSHADE_PRIMARY_TRANSMISSION=0`, plus
`WAVE_ORDINARYSHADE_PRIMARY_CONTINUATION=0` and
`WAVE_ORDINARYSHADE_PBR=0`, and
`WAVE_ORDINARYSHADE_ANALYTIC_LIGHTS=0`, plus
`WAVE_ORDINARYSHADE_AREA_LIGHTS=0` and
`WAVE_ORDINARYSHADE_ENVIRONMENT_LIGHTS=0`, plus
`WAVE_ORDINARYSHADE_UNIFIED_NEE=0` and
`WAVE_ORDINARYSHADE_EMISSIVE_MIS=0`, plus
`WAVE_ORDINARYSHADE_SECONDARY_TRANSPORT=0` and
`WAVE_ORDINARYSHADE_SECONDARY_TRANSMISSION=0`, plus
`WAVE_ORDINARYSHADE_SECONDARY_SURFACE=0` and
`WAVE_ORDINARYSHADE_SECONDARY_CONTROL=0`. Every boundary compiles independently, all-on,
all-off, and in mixed generated/fallback configurations; custom material
specialization remains at the same semantic boundary, and the generated branch
is HDR-identical across the complete execution-strategy parity matrix on a
textured real-geometry feature scene. Native-texture validation remains within
`3.5e-8` relative HDR RMSE of the handwritten production shade path. Both the
normal six-bounce matrix and the explicit one-bounce early-termination matrix
remain exactly HDR-identical across execution strategies. The 16-bounce
feature-parity and dedicated nested-glass matrices are exact across strategies
as well.
The analytic-light generated/fallback boundary preserves the same `4.9e-7`
relative HDR RMSE, including the spot-cone polynomial.
Triangle-area sampling and reservoir-candidate evaluation now share typed
barycentric sampling, emitter cosine, solid-angle PDF, power-heuristic MIS, and
radiometric contribution helpers. On the dedicated area-light scene this path
agrees with the complete generated shade graph within `2.2e-8` relative HDR
RMSE.
Environment lighting now shares typed analytic-sky evaluation, spherical HDR
mapping and decoding, cosine-domain PDFs, MIS, octahedral candidate encoding,
and material-weighted contribution math. Texture fetch and visibility remain
explicit renderer inputs. The feature-parity scene remains within `4.9e-7`
relative HDR RMSE after this boundary moved.
Unified area/environment selection is typed as well: primary domain weighting,
secondary sample-budget weighting, digitally shifted secondary-NEE selection,
and luminance target evaluation now share the Ordinary Shade graph. The hash
and bit-reversed sequence remain exactly deterministic and do not consume BSDF
RNG state.
Reciprocal weighting is typed too. Environment misses reconstruct the selected
environment PDF from path metadata, while emissive hits reconstruct triangle
area, sidedness, selection density, and solid-angle density before applying
the power heuristic. The small-emitter stress scene agrees within `2.3e-8`
relative HDR RMSE.
Secondary opaque transport now uses typed NEE compensation, BSDF-throughput
application, roughness-driven ray-cone growth, continuation flags/PDFs, and
Russian-roulette survival and renormalization. RNG advancement and queue writes
remain explicit entry-point responsibilities. A 16-bounce diffuse scene agrees
within `6.1e-8` relative HDR RMSE.
Secondary transmission now shares typed target-IOR selection, relative-IOR
refraction, total-internal-reflection fallback, medium push/pop decisions, and
tinted throughput. Indexed stack storage remains renderer orchestration. The
16-bounce nested-glass scene agrees within `2.4e-7` relative HDR RMSE.
Secondary hit preparation now reuses typed ray-cone advancement, barycentric
reconstruction, geometric and oriented shading normals, UV interpolation and
footprints, tangent reconstruction, and mapped-normal correction. Descriptor
loads and material-program dispatch remain explicit seams. Native-texture
validation agrees within `6.1e-8` relative HDR RMSE.
Secondary control flow now uses typed hybrid stop selection, low-throughput and
maximum-bounce termination, miss accumulation, and indirect-capture eligibility.
Volume integration and capture-buffer writes remain explicit resource work. A
semantic material-application marker keeps Python material specialization
stable across surrounding generated/fallback branches; custom-material HDR
parity remains within `4.9e-7` relative RMSE.
The final fused-loop policy seams are typed as well: SER material hints, NEE
probability clamping, area/environment sample-count bounds, contribution
averaging, emissive sidedness, and capture-position payload construction.
The remaining handwritten fused loop is deliberately limited to Vulkan ray
queries, descriptor/indexed storage, volume-pass invocation, profiling,
reordering intrinsics, RNG/resource mutation, and queue/state writes. A
structural test protects that boundary. The completed fused path remains
bit-exact across wavefront, megakernel, SER, hybrid, persistent-continuation,
persistent, and coarse-tile persistent execution strategies on the native
feature-parity scene.
The backend-orchestration migration is now active too. Ordinary Shade authors
the fused TLAS query and complete intersection payload, SER reorder call,
subgroup-compacted queue reservation, continuation queue validation/payload
writes, overflow atomics, and workgroup dispatch swizzle. Renderer-owned
descriptor blocks are consumed through typed external ABI values rather than
raw source injection. These generated and fallback paths compile independently,
and the generated default remains bit-exact across the complete execution
strategy matrix.
Typed external arrays now cover vertex, attribute, material, medium-stack,
primary-path, and secondary-capture storage. Ordinary Shade's structure system
supports fixed-size array fields for the nested-medium ABI, and generated
functions own the corresponding reads and mutations without redeclaring the
renderer descriptor blocks.
The established generated secondary graph and the new shared PBR wrapper agree
within `4.9e-7` relative HDR RMSE on the 16-bounce feature-parity scene.

Pipeline caches default to
`$XDG_CACHE_HOME/ordinarylight/vulkan/<device-uuid>.bin`, or the equivalent
directory under `~/.cache`. Set `RendererConfig(vulkan_pipeline_cache=False)`
to disable persistence or `vulkan_pipeline_cache_path=...` to select an
application-managed location. The presentation tool exposes the same controls
as `WAVE_RENDER_PIPELINE_CACHE=0` and `WAVE_RENDER_PIPELINE_CACHE_PATH=...`.
Cache loading, invalidation, and saving are best-effort, so read-only containers
continue to work without persistence.

Automatic execution-strategy selection compiles only the pipeline family and
scene specialization that the resident scene can dispatch. If a later scene
resolves to a different strategy or specialization, Ordinary Light replaces
the wavefront executor after synchronization while retaining the Vulkan device,
swapchain/external images, and persistent pipeline cache. On the reference GPU,
this reduced a genuine cold-cache auto first-frame record from 58.6 seconds to
7.62 seconds; the following process uses the persistent cache. The formal
`strategy_transition` gate exercises wavefront-to-megakernel transitions on one
renderer.

The Qt event thread never performs rendering, swapchain recreation, Vulkan
teardown, or scene parsing. A presentation worker owns the Vulkan presenter
for its entire lifetime and accepts at most one outstanding frame, while a
separate loader builds scenes. This backpressure is intentional: it keeps Qt
responsive instead of accumulating stale camera frames.

Interactive showcases are ordinary scripts rather than separate applications.
Built-ins live under `ordinarylight.showcases.catalog`. Set
`ORDINARYLIGHT_SHOWCASE_PATH` to one or more additional script directories and
press **Reload scripts** to discover changes. A script only needs to declare a
`Showcase`:

```python
import ordinarylight as ol
from ordinarylight.integrations.workbench import Showcase

def build():
    scene = ol.Scene()
    # Populate the scene here.
    return scene

SHOWCASE = Showcase("my-scene", "My scene", build)
```

The wavefront renderer additionally imports PNG/JPEG base-color,
metallic-roughness, and emissive textures, `TEXCOORD_0`, glTF wrap modes, and
nearest/linear filtering. Tangent-space normal maps use authored glTF tangents
when present and generated tangents otherwise, including normal-scale and
mirrored-transform handedness. Color textures are decoded from sRGB while
material data maps remain linear. Non-transmissive materials use a shared GGX
metallic-roughness BRDF for direct lighting and path continuation. Test the
bundled textured scene through either
execution strategy with:

```bash
WAVE_RENDER_SCENE=assets/TransmissionTest.glb \
WAVE_RENDER_EXECUTION_STRATEGY=wavefront \
python -m tools.wavefront_present
```

Set `WAVE_RENDER_EXECUTION_STRATEGY=auto` to select the feature-equivalent
megakernel when at least 25% of scene triangles are transmissive, while keeping
the compacting wavefront path for other scenes. Override the threshold with
`WAVE_RENDER_AUTO_MEGAKERNEL_TRANSMISSION_FRACTION`; explicit `wavefront` and
`megakernel` selections always take precedence. The interactive wavefront demo
uses `auto` by default; the library configuration retains explicit `wavefront`
as its conservative default.

Scene specialization is enabled by default. When static inspection proves that
a scene is opaque, the hybrid and megakernel paths omit transmission handling.
Unknown dynamic material programs, transmission textures, and glass fall back
to the general shaders. Disable specialization for A/B testing with
`WAVE_RENDER_SCENE_SPECIALIZATION=0`.

The experimental untextured megakernel removes UV interpolation, footprint calculation,
tangent reconstruction, normal-texture binding checks, and opaque medium-stack
traffic at compile time. Literal diffuse/reflection events from Python-authored
`SurfaceResponse` programs are proven opaque; dynamic event expressions remain
on the general path. It is currently quarantined from runtime selection: the
fused presentation HDR gate exposed missing environment pixels that the older
offscreen parity test did not exercise. Its prior 4K performance measurements
remain historical data, not an active optimization claim.

Opaque-specialized executors also replace the per-path refractive-medium stack
with a one-entry descriptor-safe dummy allocation. This avoids 8 MiB with the
demo's default 131,072-path tile capacity, or 256 MiB at the maximum supported
4,194,304-path capacity. Scene changes resize the stack at a synchronized bind
point, so mixed opaque and nested-glass workflows retain full behavior.

When generalized MIS and the alternate unified/stratified primary estimators
are disabled, the opaque/untextured megakernel automatically selects a
production ReSTIR variant that compiles those unused branches out. The variant
is HDR-exact; current RTX 4070 measurements are performance-neutral within run
variance, so it is retained as code-size and register-pressure hygiene rather
than counted as an FPS improvement.

The hybrid primary/continuation split has the same exact combined
opaque/untextured/production specialization. It reduced dense-4K GPU time from
26.73 to 25.23 ms in a back-to-back test, but remains slower than the fused
megakernel. Queue traffic and additional dispatches outweigh the split
kernel's lower register pressure on the RTX 4070, so the megakernel remains
the production performance target.

The path-termination quality gate compares full paths against unbiased Russian
roulette using matched moving HDR sequences:

```bash
python -m tests.gates.path_termination_quality --scene dense --gate \
  --roulette-start 4 --output /tmp/path_termination_quality
```

It gates relative RMSE, temporal residual, bias, and low-frequency noise.
Bounce-4 termination currently passes; the faster bounce-3 variant remains
opt-in because it materially increases finite-sample variance.

It can also compare maximum path depths directly. The representative mixed
metal/glass area-light scene passes the five-versus-four-bounce gate, while the
nested-glass scene correctly rejects four bounces because low-frequency error
structure increases. Consequently the library default remains five bounces;
the 4K `area_lights` performance profile uses its independently validated
four-bounce setting:

```bash
python -m tests.gates.path_termination_quality --scene area_lights --bounces 5 \
  --candidate-bounces 4 --reference-samples 32 --gate \
  --output /tmp/path_depth_quality
```

The corrected front-facing 4K gate combines that scene-specific depth with the
HDR-exact general untextured specialization and a 2,097,152-path tile. On the
RTX 4070 Laptop GPU it measured 52.21 FPS at 3840x2130 (98.6% of 4K), with a
17.94 ms median GPU time and 19.73 ms median cadence. Native Wayland runs
maximize after the first frame by default so compositor decoration does not
silently shrink the measured framebuffer below the gate's 98% pixel threshold.

`WAVE_RENDER_MEGAKERNEL_SINGLE_WARP=1` selects an experimental 8x4
opaque/untextured megakernel workgroup. It is bit-exact but remains disabled:
on the RTX 4070 test system it increased median dense-4K GPU time from about
25.45 to 26.33 ms, showing that the default 8x8 group provides better latency
hiding for this kernel.

`WAVE_RENDER_MEGAKERNEL_GROUP_SWIZZLE=32` selects an exact 32-workgroup
horizontal scheduling tile for the opaque/untextured production megakernel.
It changes dispatch-to-pixel ordering only and retains the unswizzled shader as
the portable default. A matched dense 3840x2160 RTX 4070 A/B reduced median GPU
time from 21.30 to 20.96 ms and increased measured throughput from 38.10 to
39.32 FPS. Widths 8 and 16 are also available for device-specific tuning.

The optional ray-generation/SER foundation can be checked independently with:

```bash
python -m tools.diagnostics.ser
```

Normal rendering does not enable ray-pipeline or SER device features. The probe
requests them explicitly, compiles and creates a minimal ray-generation
pipeline, builds its shader binding table, dispatches reordered invocations,
and validates their output. This keeps the compute ray-query renderer as the
portable production path while the ray-generation implementation reaches
feature parity.

An opt-in production strategy executes the shared megakernel implementation as
a ray-generation shader and enables NVIDIA invocation reordering:

```bash
WAVE_RENDER_EXECUTION_STRATEGY=ser \
WAVE_RENDER_SER=1 \
python -m tools.wavefront_present
```

The raygen and compute variants include the same path-tracing implementation;
only launch indexing and the SER reorder point differ. The strategy remains
experimental and is never selected by `auto`. `tests.gates.execution_parity`
includes it in the HDR equivalence gate on SER-capable hardware.

Use the fused presentation quality/performance gate for meaningful comparison:

```bash
WAVE_RENDER_GLFW_PLATFORM=wayland python -m tests.gates.ser_quality \
  --width 3840 --height 2160 --window-scale 1.25 --scene glossy_glass
```

The gate rejects black output, rejects framebuffer extents outside 2% of the
target, scores a memory-bounded HDR grid at 4K, and reports median GPU time for
compute, baseline raygen, and SER. Keep `--window-scale` matched to desktop
fractional scaling; otherwise GLFW logical dimensions do not represent native
pixels. SER remains opt-in until a corrected native-4K comparison justifies
selecting it automatically.

GPU-time-driven dynamic resolution can target a full-resolution 4K output
while varying the internal traced extent without reallocating Vulkan images:

```bash
WAVE_RENDER_SCENE=assets/TransmissionTest.glb \
WAVE_RENDER_EXECUTION_STRATEGY=wavefront \
WAVE_RENDER_DYNAMIC_RESOLUTION=1 \
WAVE_RENDER_DYNAMIC_TARGET_MS=16.67 \
WAVE_RENDER_DYNAMIC_MIN_SCALE=0.5 \
WAVE_RENDER_RUSSIAN_ROULETTE=1 \
WAVE_RENDER_RUSSIAN_ROULETTE_START=4 \
WAVE_RENDER_SECONDARY_NEE_PROBABILITY=0.5 \
WAVE_RENDER_SECONDARY_AREA_LIGHT_SAMPLES=2 \
python -m tools.wavefront_present
```

Dynamic resolution enables temporal reconstruction by default in the demo.
`WAVE_RENDER_SCALE` is the maximum internal scale and the title bar reports the
selected scale and `DRS`. The controller normalizes delayed GPU timestamps by
the scale that produced them, reduces resolution quickly when over budget, and
requires sustained headroom before increasing it.
`WAVE_RENDER_SECONDARY_NEE_PROBABILITY` controls unbiased direct-light
evaluation on secondary diffuse bounces. Values below one reduce shadow rays
and compensate retained samples by their selection probability; primary
lighting and transmissive paths are always preserved. Half-rate secondary NEE
is intended for temporal reconstruction and remains opt-in because it raises
single-frame variance.
`WAVE_RENDER_SECONDARY_AREA_LIGHT_SAMPLES` independently controls emissive
triangle samples on secondary diffuse hits. Zero (the default) inherits
`WAVE_RENDER_AREA_LIGHT_SAMPLES`; setting a smaller value preserves primary
surface quality while reducing deeper-path shadow rays.
For emissive-triangle scenes, four primary samples plus two consistent
secondary samples is the preferred quality/performance starting point. It is
usually less noisy and more execution-coherent than selecting four secondary
samples at 50% probability, despite a similar expected shadow-ray budget.

The first ReSTIR-DI building block lives in
`ordinarylight.integrations.restir_di`. Its CPU reference reservoir and the
matching tightly packed 12-byte `wavefront_restir.glsl` history ABI implement
weighted candidate selection, temporal reservoir merging with target
reevaluation, and unbiased final normalization. The shader retains a
full-precision arithmetic reservoir while tracing, then stores a 25-bit light
index, 7-bit sample count, half-precision barycentrics, selected target, and
historical weight sum. Two full-resolution 4K history slots consume about
190 MiB, roughly 63 MiB less than the previous 16-byte layout. GPU history
allocation and primary-surface reuse are kept
as a separate opt-in stage so the existing direct-light path remains unchanged
while that integration is validated.
`WAVE_RENDER_RESTIR_DI=1` now allocates compact frame-slot reservoir history at
the maximum internal render extent and binds current/previous buffers to the
primary execution strategies. `WAVE_RENDER_RESTIR_HISTORY_LIMIT` controls the
validated temporal sample-count cap. The experimental gate builds a weighted
reservoir from primary area-light candidates, reprojects and geometry-validates
the previous frame's selected candidate, bounds accumulated history, and traces
only the resulting selected visibility ray; secondary lighting is unchanged.
With one fresh candidate and a 20-sample history cap, the 4K-output/half-scale
area-light benchmark reduced hard-view shadow rays by roughly 28% and improved
p90 GPU time from about 16.0 to 15.2 ms. It remains off by default pending visual
A/B review of disocclusions and fast camera motion.
`WAVE_RENDER_RESTIR_CANDIDATES` controls fresh primary candidates independently
of `WAVE_RENDER_AREA_LIGHT_SAMPLES`; its default is one because temporal history
provides the remaining candidate population.

`WAVE_RENDER_MATERIAL_BUCKETING=1` enables the experimental coherent wavefront
path. Its intersection shader uses subgroup-coalesced enqueue to split plain
and textured hits, then shades the two queues independently. It currently
serves as an instrumented development path rather than a default optimization:
the extra indirect dispatches remain slower than ordinary wavefront execution
on the TransmissionTest benchmark. Bucketing is now bounce-aware: set
`WAVE_RENDER_MATERIAL_BUCKETING_START_BOUNCE` (default 2) to leave dense early
bounces in their original order and only sort later, more divergent work.
Scheduling changes do not alter sampling or shading arithmetic and must pass
the same HDR quality gates before adoption.

`WAVE_RENDER_EXECUTION_STRATEGY=persistent` selects the dedicated persistent
scheduler development target. Its first milestone owns an independent shader
module, native-texture/profile variants, configuration surface, timing labels,
and parity gates while using the proven in-kernel path loop. It is therefore
bit-exact with the megakernel today (zero HDR error in the execution-parity
scene) and measured 55.7 FPS at 3840x2130. Future GPU queue ownership and work
stealing can now evolve behind this strategy without changing the production
megakernel or the canonical multi-dispatch wavefront implementation.
An initial atomic work-stealing prototype was quality-exact but rejected: a
per-invocation allocator reached 38.6 FPS and subgroup-coalesced two-pixel
claims reached 43.4 FPS, both below the 55.7 FPS locality-preserving baseline.
Future persistent scheduling should therefore retain coherent 2D tiles and
steal only coarse tiles or divergent continuation work.
`WAVE_RENDER_PERSISTENT_COARSE_TILES=1` enables the resulting 8x8 tile-stealing
prototype when the persistent strategy is selected. A 512-workgroup pool
preserves exact pixel RNG and spatial ray coherence, passing at 55.2 FPS with
an 11.9 ms median GPU time. It remains opt-in because static tile ownership
still measured slightly faster (57.5 FPS, 11.5 ms) on the current scene; the
variant is retained for A/B testing on more spatially imbalanced workloads.

`WAVE_RENDER_PERSISTENT_CONTINUATIONS=1` with
`WAVE_RENDER_EXECUTION_STRATEGY=hybrid` keeps the coherent three-bounce inline
prefix, then consumes its compact survivor queue in one persistent continuation
dispatch. It preserves path state and pixel-derived RNG exactly, passes all 12
HDR scene/motion cases, and produced zero error in deterministic execution
parity. At 3840x2130 it improved the hybrid path from 50.9 FPS / 15.1 ms median
GPU to 57.8 FPS / 10.8 ms by removing the remaining per-bounce indirect
dispatches. This is currently opt-in while it is exercised on additional scene
distributions.

`WAVE_RENDER_SCENE_SPECIALIZATION=1` enables an experimental opaque-only
hybrid/continuation shader for scenes using the standard parameter-based
material model. It is never selected for transmission textures, transmissive
materials, or Python-authored `SurfaceResponse` programs, whose scattering
behavior cannot be inferred from static material fields. The variant is
bit-exact on an opaque regression scene, but a 3941x2161 A/B measured 10.48 ms
median GPU versus 10.36 ms for the generic shader (58.7 versus 57.3 end-to-end
FPS, dominated by presentation variance). It therefore remains disabled by
default; current Vulkan compiler optimization already removes the profitable
static branches.

The hybrid strategy keeps dense early paths inside the primary kernel, then
enqueues survivors into the compacting wavefront pipeline. Select it with
`WAVE_RENDER_EXECUTION_STRATEGY=hybrid`; the default cut point is three bounces.
`WAVE_RENDER_HYBRID_INLINE_BOUNCES` accepts odd values from 3 through 15 so the
existing ping-pong queue ABI remains unchanged.

Packed texture data is staged into immutable device-local memory by default,
rather than remaining in host-visible upload memory.
`WAVE_RENDER_DEVICE_LOCAL_TEXTURES=0` selects host-visible memory for the
packed sampler. Set `WAVE_RENDER_NATIVE_TEXTURES=1` to enable the experimental
descriptor-indexed native sampled-image path for a quality/performance A/B.

The packed texture table includes sRGB-correct and linear mip pyramids.
Primary and secondary rays propagate a cone footprint and select explicit LODs;
linear samplers blend both texels and adjacent mip levels.
`KHR_texture_transform` offset, scale, and rotation are represented by reusable
texture bindings and apply independently to every supported texture channel.
Normal-map transforms also adjust the tangent basis. `TEXCOORD_0` and
`TEXCOORD_1` are packed into the same vertex record, and every texture binding
selects its coordinate set independently. Occlusion textures use
their red channel and strength to attenuate indirect diffuse/environment
energy without darkening direct point or area lights.
`KHR_materials_transmission` textures use their linear red channel to modulate
the material transmission factor. Texture-coordinate sets beyond `TEXCOORD_1`
remain unsupported.

This window has no OpenGL context. The compute shader writes an RGBA8 Vulkan
storage image, which is blitted entirely on the GPU into the GLFW swapchain and
presented without NumPy image readback. Presentation rotates across two complete
frame slots, each with its own output image, descriptor set, acquire/render
semaphores, fence, and reusable command buffer. Space pauses rotation,
Left/Right changes its speed, R resets it, and Escape closes the window.

The presenter prefers `MAILBOX`, then `IMMEDIATE`, with portable `FIFO` as its
fallback. Override the choice for comparison with
`WAVE_RENDER_PRESENT_MODE=fifo`, `mailbox`, or `immediate`. The title reports
GPU timestamp duration separately from CPU acquire/submit/present wall time.
The performance demo defaults to six swapchain images because native 4K testing
showed that this reduces Wayland image-acquisition backpressure substantially;
override it with `WAVE_RENDER_SWAPCHAIN_IMAGES=N` when testing memory or latency
tradeoffs.
Periodic statistics are disabled by default because synchronous terminal and
window-title reporting can cause a visible compositor hitch. Enable terminal
instrumentation with `WAVE_RENDER_STATS_INTERVAL=1`; additionally set
`WAVE_RENDER_TITLE_STATS=1` only when live title text is useful.
The tool also disables Python's cyclic garbage collector during the render loop
to avoid periodic collection pauses from high-frequency CFFI wrapper creation;
set `WAVE_RENDER_DISABLE_GC=0` when comparing collector behavior.
On Linux, compare native Wayland against XWayland with
`WAVE_RENDER_GLFW_PLATFORM=x11`; use `wayland` to force the native backend.
Native Wayland resize configure events are settled for 150 ms before recreating
the resolution-dependent Vulkan images, avoiding transient allocation spikes
during maximize. Override this with `WAVE_RENDER_RESIZE_SETTLE_MS` or set it to
zero when diagnosing raw compositor resize behavior.
Use `WAVE_RENDER_TARGET_FPS=120` (or another display-appropriate rate) to pace
submissions and avoid exhausting compositor-owned swapchain images.

## Vulkan scope

### Experimental raster renderers

Ordinary Light has a renderer-neutral raster contract. `RasterProgram` links
Ordinary Shade vertex and fragment functions, `RasterMesh` carries typed
interleaved vertex/index data, and `RasterState` describes topology, culling,
depth, and blending. Two offscreen implementations exercise the same program
semantics:

- `renderers.raster.VulkanRasterRenderer` consumes SPIR-V and records a native Vulkan graphics
  pipeline, color/depth render pass, draw, and readback.
- `renderers.raster.WebGpuRasterRenderer` consumes WGSL through the optional `wgpu` package.

Install `ordinarylight[vulkan]` or `ordinarylight[webgpu]` for the respective
runtime. Compiling Python shader functions additionally requires Ordinary
Shade. Both renderers implement the standard `RendererProtocol` contract and can
render existing `Scene` meshes, instances, materials, and object transforms
through perspective and orthographic cameras. Panoramic cameras require a
later non-linear projection pass.

The portable scene layer evaluates the same `Material` and `MaterialProgram`
objects used by GI, point/spot/directional lighting, native directional/spot
shadow maps, and named depth, normal, and object-ID products. Its validated
render graph composes geometry, shadow, lighting, temporal, and post stages.
Static-scene accumulation, Reinhard/ACES tone mapping, transfer-function volume
slicing, hybrid GI composition, and `Renderer.render_to()` surface presentation
are opt-in.

Raster materials use a stable, vec4-aligned GPU record selected by material ID.
It carries base color, emission, roughness, metallic, attenuation,
transmission, IOR, normal scale, occlusion strength, program kind, feature
flags, and indices for every supported texture channel. Base-color, emissive,
metallic-roughness, normal, occlusion, and transmission textures share a
portable GPU atlas; color channels are decoded as sRGB and material-data
channels remain linear. Authored or generated tangents drive tangent-space
normal mapping. Deterministic raster equivalents are provided for PBR,
diffuse, mirror, glass, and unlit material programs, while GI retains the
program's stochastic transport semantics. Program-set signatures select cached
shader/pipeline variants without changing the scene API.

Auxiliary depth/normal/object-ID products currently use correctness-oriented
CPU implementations. Native MRT and GPU volume passes remain optimization
targets without changing these public semantics.

Run `python tools/raster_feature_viewer.py` to open the extensible Qt raster
feature catalog. Each catalog entry is a small `Showcase` script containing its
scene builder, camera, renderer defaults, description, and tags. The viewer can
live-render Vulkan, WebGPU, or both and currently includes directional and spot
shadow demonstrations, a complete material-texture scene, and a shared
raster/GI material-program scene. Vulkan uses a Qt-owned native window and direct
swapchain presentation by default, avoiding NumPy readback and `QImage`
uploads. Its scene buffers, attachments, descriptor state, recorded command
buffers, and frame synchronization remain resident after the swapchain warms.
Pass `--readback` to compare Vulkan and WebGPU through the diagnostic image
path. The viewer provides automatic camera animation, reusable
`ArcballCameraController` orbit/pan/dolly interaction, and render-resolution
presets from 720p through 4K.

```python
program = ol.RasterProgram.scene(target="wgsl")
implementation = ol.renderers.raster.WebGpuRasterRenderer(
    program, state=ol.RasterState(cull_mode="back"),
)
renderer = ol.Renderer(implementation=implementation)
rgba = renderer.render(scene, camera, (640, 480))
renderer.close()
```

The current GPU slice builds BLAS/TLAS acceleration structures, reads
per-triangle material records at ray-query intersections, and traces up to five
segments for glass transmission and metallic reflection. It writes packed
RGBA8 output either to a readback buffer or directly to a Vulkan swapchain.
Texture sampling, stochastic BRDFs, shadow rays, accumulation, and denoising
remain future work.

## Python material programs

Materials can be authored as restricted Python functions. The decorator runs
the function once with symbolic values and produces a typed, renderer-neutral
`MaterialProgram`; it does not execute Python for each ray:

```python
import ordinarylight as ol

@ol.material
def tinted_glass(ctx):
    tint = ol.mix(ctx.base_color, ol.vec3(1.0), 0.2)
    return ol.MaterialEvaluation(
        base_color=tint,
        emission=ctx.emission,
        metallic=ctx.metallic,
        roughness=ctx.roughness,
        transmission=ctx.transmission,
        ior=ol.select(ctx.entering, ctx.ior, 1.0),
        attenuation_color=ctx.attenuation_color,
        attenuation_distance=ctx.attenuation_distance,
    )

print(tinted_glass.glsl())
```

Use it in either Vulkan path through the renderer configuration:

```python
config = ol.RendererConfig(material_program=tinted_glass, max_bounces=8)
presenter = ol.VulkanGlfwPresenter(window, config=config)
```

Use `ol.select(condition, a, b)` for conditions involving symbolic values;
ordinary Python `if` statements cannot depend on GPU data. Arithmetic,
swizzles, `mix`, `dot`, `normalize`, `reflect`, and `refract` are currently
supported. `MATERIAL_PARAMETER_LAYOUT` documents the material-buffer ABI.

Custom programs are compiled into the complete compute shader when the Vulkan
pipeline is created and cached in-process. This requires `glslangValidator`
(from `glslang-tools` or the Vulkan SDK) or `glslc` on `PATH`. The default
material continues to use packaged, precompiled SPIR-V and needs no compiler.
The **Python-authored materials** workbench showcase exercises compiled custom
material programs:

```bash
ordinarylight-workbench
```

Programs may instead be attached to individual material instances. The
renderer generates one evaluator per distinct program and dispatches by the
per-triangle program ID:

```python
blue = ol.Material(base_color=(0.2, 0.4, 1.0), program=blue_program)
gold = ol.Material(base_color=(1.0, 0.7, 0.1), program=metal_program)
scene.add_mesh(vertices_a, indices_a, blue)
scene.add_mesh(vertices_b, indices_b, gold)
```

The same workbench showcase includes per-mesh material dispatch:

```bash
ordinarylight-workbench
```

For direct control over path scattering, return a `SurfaceResponse` instead of
parameter adjustments:

```python
@ol.material
def mirror(ctx):
    return ol.SurfaceResponse(
        emission=(0.0, 0.0, 0.0),
        weight=ctx.base_color,
        next_direction=ol.reflect(ctx.direction, ctx.normal),
        event=ol.SCATTER_REFLECTION,
        pdf=1.0,
    )
```

`SCATTER_ABSORB` terminates the path. `SCATTER_DIFFUSE`,
`SCATTER_REFLECTION`, and `SCATTER_TRANSMISSION` continue along
`next_direction`, multiplying throughput by `weight / pdf`. Emission is added
at every hit. Transmission events update a bounded nested-medium IOR stack.

Material contexts expose interpolated `normal` and `uv` attributes plus
`random_u`, `random_v`, `bounce`, `current_ior`, and `exterior_ior`.
`fresnel_schlick(...)` and `cosine_sample_hemisphere(...)`
provide common sampling building blocks. Symbolic comparisons combined with
`select(...)` provide probabilistic event selection.

Meshes accept optional per-vertex `normals` and `texcoords` in
`Scene.add_mesh(...)`. Missing normals are generated using area-weighted faces
with identical-position seam welding; missing UVs default to zero. glTF
`NORMAL` and `TEXCOORD_0` accessors are imported, transformed, uploaded, and
barycentrically interpolated at ray hits. Geometric normals remain separate for
front/back classification and robust ray offsets.

Select **Vertex attributes** in the workbench:

```bash
ordinarylight-workbench
```

It renders matching smooth- and flat-normal UV spheres with a high-contrast
Python-authored material driven by `ctx.uv`, demonstrating explicit normals,
texture coordinates, barycentric interpolation, and material-DSL swizzles.

Set `RendererConfig(samples_per_pixel=N)` or pass `samples=N` to `render()` or
`present()` to accumulate between one and 64 jittered samples per pixel in a
frame. The direct demo exposes this as `WAVE_RENDER_SAMPLES=N`.

Direct presentation also supports persistent linear-HDR accumulation across
frames:

```python
config = ol.RendererConfig(
    samples_per_pixel=1,
    progressive_accumulation=True,
)
presenter = ol.VulkanGlfwPresenter(window, config=config)
presenter.reset_accumulation()  # optional explicit invalidation
```

History resets automatically when the camera, scene object, or framebuffer
size changes. `presenter.accumulated_frames` reports its current length. The
workbench exposes pause and reset actions for inspecting convergence.

For interactive presentation or zero-copy video streaming, accumulate only
after the camera and scene have remained unchanged for a short settling
interval:

```python
config = ol.RendererConfig(
    external_image_interop=True,
    progressive_accumulation=True,
    stationary_accumulation=True,
    stationary_delay_seconds=0.15,
    wavefront_interactive_render_scale=0.5,
)
renderer = ol.Renderer(config=config)
frame = renderer.render_gpu(
    scene, camera, (1920, 1080), pixel_format="nv12"
)
print(frame.attributes["accumulation_state"])
```

Camera motion, scene revisions, resolution changes, and explicit history
resets produce `moving` or `settling` frames; stable frames report
`accumulating`. The same exported NV12/P010 pool and NVENC session remain in
use throughout. `renderer.accumulation_state` and
`renderer.accumulated_frames` expose the state without reading pixels back.
When `wavefront_interactive_render_scale` is set, moving and settling frames
render at that internal scale and reconstruct to the unchanged output extent;
stable frames return to `wavefront_render_scale`. Values from `0.25` through
`1.0` are accepted and must not exceed the full-quality render scale.

For automatic motion-only scaling, specify a frame-rate target instead of a
fixed interactive scale:

```python
config = ol.RendererConfig(
    wavefront_render_scale=1.0,
    wavefront_interactive_target_fps=60.0,
    wavefront_interactive_min_scale=0.5,
    stationary_delay_seconds=0.15,
)
```

The controller uses measured Vulkan GPU time to select the highest scale
expected to meet the target while the camera or scene is moving. It will not
go below `wavefront_interactive_min_scale`; once stationary, rendering returns
to `wavefront_render_scale`. A fixed `wavefront_interactive_render_scale` may
also be supplied as the automatic controller's maximum motion scale.

Simple scenes can spend otherwise unused motion-frame budget on additional
samples. The configured `samples_per_pixel` remains the ceiling, while the
motion controller begins at `wavefront_interactive_min_samples` and selects
the highest integer SPP predicted to meet the same frame-rate target:

```python
config = ol.RendererConfig(
    samples_per_pixel=8,
    wavefront_interactive_target_fps=60.0,
    wavefront_interactive_sample_scaling=True,
    wavefront_interactive_min_samples=1,
)
```

When combined with automatic motion resolution, resolution recovery has
priority: the renderer holds the minimum SPP until it can sustain the maximum
allowed interactive scale, then spends remaining headroom on samples. Stable
frames always use `samples_per_pixel`. The selected value is available as
`renderer.effective_samples_per_pixel`, the `wavefront_samples_per_pixel`
timing statistic, and `GpuFrame.attributes["samples_per_pixel"]` for streaming.
Fixed `interactive_samples_per_pixel` and automatic sample scaling are mutually
exclusive.

Use a smaller sample budget while the camera is moving and switch back after
it settles:

```python
config = ol.RendererConfig(
    samples_per_pixel=8,
    interactive_samples_per_pixel=1,
    stationary_delay_seconds=0.15,
    progressive_accumulation=True,
    stationary_accumulation=True,
)
```

The demo equivalents are `WAVE_RENDER_INTERACTIVE_SAMPLES` and
`WAVE_RENDER_STATIONARY_DELAY`. `presenter.effective_samples_per_pixel`
reports the budget used by the last frame.

`RendererConfig(temporal_history=True)` enables camera-motion reprojection in
the shader showcase and requires progressive accumulation. Ping-pong G-buffers
store an octahedrally encoded geometric normal, linear ray depth, and
primitive/material identity. Current primary hits are projected through the
previous camera and history is accepted only when its depth, normal, and
identity remain compatible; disoccluded pixels restart locally. Per-pixel
sample counts keep differently aged history statistically weighted. The demo
equivalent is `WAVE_RENDER_TEMPORAL_HISTORY=1`.

At 4K, the two RGBA32F accumulation images consume about 253 MiB. Wavefront
position history stores one R32F ray distance per pixel and reconstructs world
positions from the frame camera, avoiding RGBA16F world-position history and
saving about 63 MiB across two 4K frame slots. Normals use 15-bit octahedral
coordinates plus a 2-bit surface class in one R32_UINT image, saving another
63 MiB while retaining material identity in its own R32_UINT resource.

Reprojected history is clamped to a compatible 3×3 depth/normal/primitive
neighborhood before blending. `temporal_history_limit` caps the effective
per-pixel sample age (32 by default), preventing stale history from dominating
new lighting. The demo equivalent is `WAVE_RENDER_TEMPORAL_HISTORY_LIMIT`.
Set `temporal_neighborhood_clamping=False` (or
`WAVE_RENDER_TEMPORAL_CLAMPING=0`) when unbiased sharp reflections are more
important than suppressing temporal outliers.

Variance-guided sampling uses ping-pong R32F luminance second moments:

```python
config = ol.RendererConfig(
    samples_per_pixel=4,
    progressive_accumulation=True,
    temporal_history=True,
    adaptive_sampling=True,
    adaptive_min_samples=1,
    adaptive_variance_threshold=0.0025,
)
```

Pixels with at least two history samples and variance below the threshold use
the minimum budget; noisier pixels retain `samples_per_pixel`. Environment
variables are `WAVE_RENDER_ADAPTIVE_SAMPLING`,
`WAVE_RENDER_ADAPTIVE_MIN_SAMPLES`, and `WAVE_RENDER_ADAPTIVE_VARIANCE`.
Moment ping-pong adds about 63 MiB at 4K. The reported effective spp is the
frame's upper budget; adaptive selection occurs independently per pixel.

Point lights are explicitly sampled with shadow rays at diffuse interactions:

```python
scene.add_point_light(
    position=(2.0, 5.0, 1.0),
    color=(1.0, 0.8, 0.6),
    intensity=80.0,
)
```

This is next-event estimation for delta lights. Their MIS weight is one because
a continuous BSDF sampler has zero probability of selecting the exact point
light direction.

Mesh triangles with nonzero material emission are automatically sampled as
area lights. Their area PDF is converted to solid angle and combined with the
diffuse BSDF PDF using the power heuristic, both for explicit light samples
and when a BSDF path lands on an emitter. Triangle selection is proportional
to area times emission luminance, so large or powerful emitters receive more
samples without biasing the result. The dedicated showcase uses only emissive
geometry:

```bash
ordinarylight-workbench
```

Emission is one-sided by default and follows triangle winding. Set
`Material(emission_two_sided=True)` for thin geometry that should radiate from
both faces. Imported glTF materials preserve their `doubleSided` flag for this
purpose. Explicit light sampling and rays that hit emitters use the same
sidedness rule and MIS PDF.

Small, bright emitters may need more than one explicit sample per diffuse
interaction. Use `RendererConfig(area_light_samples=N)` (`1`–`16`) or set
`WAVE_RENDER_AREA_LIGHT_SAMPLES=N` in the GLFW demos. These samples are
stratified across the power-weighted light distribution and averaged; MIS uses
the corresponding sample count. The area-light showcase defaults to four.
Because those samples affect only direct diffuse lighting, the showcase also
uses eight full paths per stationary pixel and four while the camera moves;
this targets indirect, reflection, and refraction noise as well.

## Render pipeline composition

### Wavefront execution foundation

The monolithic path tracer remains the active execution strategy, while the
first renderer-neutral wavefront ABI is available in `ordinarylight.wavefront`.
`RAY_DTYPE`, `HIT_DTYPE`, and `PATH_STATE_DTYPE` exactly describe the std430
records that will connect ray generation, intersection, and shading dispatches.
Rays and hits carry persistent path indices; path state retains throughput,
radiance, RNG state, bounce metadata, and a 16-entry nested-medium stack.

`WavefrontQueueLayout` sizes counter-prefixed work queues and provides matching
host header/record views. `wavefront_glsl_structs()` emits the corresponding
GLSL declarations. `wavefront_generate.comp` generates jittered primary rays
for a bounded pixel tile and initializes persistent path state;
`wavefront_intersect.comp` consumes that queue with Vulkan ray queries and
produces hit/miss records in a second queue. `create_wavefront_pipeline()`
exposes their logical dependency graph. These shaders are compiled and ABI
tested, but are not selected by the Vulkan presenter until its queue allocation
and descriptor path is connected. The first Vulkan connection is available as
an experimental diagnostic on the GLFW presenter:

```python
result = presenter.trace_wavefront_tile(
    scene, camera, width, height, tile_extent=(512, 512)
)
print(result["ray_queue"], result["hit_queue"], result["continuation_queue"])
```

This lazily allocates bounded ray, hit, continuation, and path-state buffers,
binds distinct generate/intersection/shading descriptor sets and pipelines,
records their storage barriers, and ping-pongs the queues through the configured
maximum bounce count. The shading pass handles environment misses, emission, diffuse
scattering, perfect metallic reflection, ideal transmission, and nested IOR
state. It currently consumes the packed built-in material parameters; generated
Python material-program dispatch will be connected after the bounce loop.
Configure the
maximum resident paths with `RendererConfig(wavefront_tile_capacity=...)`;
the default is 131,072 paths, selected from 4K GPU benchmarks after command
buffer reuse removed the CPU cost of recording additional tiles.
Direct wavefront presentation uses this pipeline through shading, tone mapping,
and the swapchain blit.

Direct presentation specializes the first bounce with
`wavefront_primary.comp`: camera-ray generation, ray-query intersection, path
initialization, and primary shading execute in one compute dispatch. This
avoids writing and rereading primary ray and hit queues. Surviving reflection,
diffuse, and transmission paths enter the ordinary continuation queue at bounce
one, so the configured nested-glass depth remains unchanged. The separate
generate/intersect/shade stages remain available to diagnostic callers.

Run `python -m tools.diagnostics.wavefront` for a minimal 64×64 GPU diagnostic.
The returned `radiance` value is a reconstructed `height × width × 4`
linear-HDR NumPy tile. A dedicated resolve compute stage associates compacted
persistent paths with their original pixel indices, so correctness does not
depend on atomic queue ordering.
The companion `rgba8` tile is tone-mapped entirely on the GPU using exposure,
an ACES-style curve, and sRGB encoding. Configure its exposure with
`RendererConfig(wavefront_exposure=...)`.
Direct presentation resolves each completed path tile with
`wavefront_path_to_hdr.comp` into a frame-local RGBA16F image. The independent
`wavefront_reconstruct.comp` stage reads that internal image, performs bilinear
reconstruction, applies exposure and ACES-style tone mapping, converts to sRGB,
and writes the full-resolution RGBA8 output. Both stages have separate pipeline
layouts and descriptor sets, and recreated images are rebound automatically.
When the window surface exposes swapchain storage images, reconstruction writes
directly to them. RGBA swapchains use the typed output shader; BGRA swapchains
use a formatless-storage variant when the Vulkan device exposes
`shaderStorageImageWriteWithoutFormat`. Unsupported surfaces retain the
intermediate-image copy path.

The fused primary stage also writes a reconstruction G-buffer at the internal
extent: RGBA16F world position plus primary distance, and RGBA16F shading normal
plus surface class (diffuse, metallic, or transmission). Optional temporal
reconstruction reprojects full-resolution HDR history using the previous
camera, rejects depth/normal/material mismatches, clamps accepted history to
the current 3x3 HDR neighborhood, and uses reduced history on metal and glass.
It remains independent of path tracing and is disabled by default, so bilinear
raw output is always available.

Animated wavefront frames upload a new RNG sequence seed through the existing
camera buffer, so cached command buffers still produce independent path samples.
Temporal history becomes valid after the first completed frame and remains
active without per-frame command rerecording; depth, normal, and material checks
continue to reject incompatible reprojections.

The wavefront primary and continuation shading stages perform the same
next-event estimation used by the direct renderer. Point lights are sampled
with visibility rays, emissive triangles are selected from the power-weighted
light distribution, and diffuse paths use power-heuristic MIS between explicit
light samples and emitter hits. Set `WAVE_RENDER_AREA_LIGHT_SAMPLES=1..16` in
the wavefront demo when small bright emitters need additional direct samples.
Run the emissive-only test scene through the wavefront backend with:

```bash
WAVE_RENDER_SCENE=area_lights WAVE_RENDER_AREA_LIGHT_SAMPLES=4 \
python -m tools.wavefront_present
```

For live comparison of conventional direct-light sampling and temporal ReSTIR
DI, use the **Area lights** scene and toggle **ReSTIR DI** in the workbench.
Automated matched comparisons remain in the ReSTIR quality gates.

Experimental spatiotemporal neighbor reuse is a separate opt-in feature:

```bash
WAVE_RENDER_RESTIR_DI=1 WAVE_RENDER_RESTIR_SPATIAL=1 \
WAVE_RENDER_RESTIR_SPATIAL_NEIGHBORS=4 \
WAVE_RENDER_RESTIR_SPATIAL_RADIUS=4 python -m tools.wavefront_present
```

The equivalent library controls are `wavefront_restir_spatial_reuse`,
`wavefront_restir_spatial_neighbors`, and `wavefront_restir_spatial_radius`.
Neighbors are read from the completed previous frame and require compatible
surface plane, normal, and material class. The feature remains disabled by
default while correlation-aware reservoir weighting is validated. Each source
supplies at most one canonical representative in spatial mode, and the history
limit caps the combined fresh and reused sample count. Recursive source
multiplicity is deliberately not imported because overlapping reservoirs can
otherwise introduce measurable positive energy bias.

Spatial reuse now uses the pairwise balance heuristic by default. It evaluates
a reused light at both the source and destination surfaces, using the source
target density already stored in the reservoir. This supports textured and
custom materials without an additional evaluated-material history buffer.
Set `wavefront_restir_pairwise_mis=False`, or
`WAVE_RENDER_RESTIR_PAIRWISE_MIS=0` in the live demo, to request canonical
spatial merging explicitly for compatibility and A/B comparisons.

Generalized neighbor balancing is independently opt-in with
`wavefront_restir_generalized_mis=True` or
`WAVE_RENDER_RESTIR_GENERALIZED_MIS=1`. It evaluates each untextured spatial
candidate under every compatible active proposal and normalizes only across
proposals assigning nonzero density. Textured surfaces retain pairwise reuse.
Repeated reuse is protected by a configurable balance-factor bound of `2.0`;
set `wavefront_restir_generalized_balance_cap` or
`WAVE_RENDER_RESTIR_GENERALIZED_BALANCE_CAP` to tune it. This mode is intended
for quality-oriented configurations: its neighbor work is
quadratic, while pairwise reuse remains the lower-cost option.

Capture deterministic moving-camera HDR sequences and compare conventional and
ReSTIR output with a high-sample reference using:

```bash
WAVE_RENDER_GLFW_PLATFORM=wayland python -m tests.gates.restir_quality
```

Pass `--motion-radians 0` for the corresponding stationary-camera stability
test. The harness streams memory-mapped HDR `.npy` sequences to disk while it
renders, then writes per-frame quality and GPU/reuse CSV reports. This keeps 4K
memory use bounded and provides explicit progress through capture, metrics, and
summary phases. On an X11 session, omit the platform override.

Capture canonical, pairwise, and generalized strategies against one shared
reference and enforce the default regression tolerances with:

```bash
WAVE_RENDER_GLFW_PLATFORM=wayland python -m tests.gates.restir_quality \
  --gate --output restir_strategy_gate
```

The quality harness accepts `--scene` to select one of several deterministic
procedural fixtures. Run the complete stationary/moving-camera regression
matrix with:

```bash
WAVE_RENDER_GLFW_PLATFORM=wayland python -m tests.gates.restir_matrix
```

This runs the conventional, canonical, and pairwise estimators
against a higher-sample reference for diffuse, glossy/glass, textured,
small-emitter, and occlusion-heavy scenes (plus the main area-light showcase).
It writes each capture and a consolidated `restir_matrix/matrix_summary.json`,
continues after individual failures, and exits nonzero if any case fails. Use
`--scenes`, `--cases`, or the resolution/sample options for targeted runs. The
regular matrix intentionally defaults to a low resolution; the separate 4K
performance gate remains the target-resolution guard. GPU timings are retained
in every matrix report, but sub-millisecond low-resolution ratios are not used
as pass/fail criteria. Generalized MIS remains
experimental and is deliberately outside the release gate; pass
`--include-generalized` to run it as a stricter diagnostic matrix.
The diagnostic uses an explicit 2× canonical GPU budget because generalized
cross-proposal evaluation is quadratic; override it with
`--gate-max-generalized-gpu-ratio` when profiling a different target.

The command exits nonzero on failure and records aggregate quality, mean GPU
time, thresholds, status, and individual failures in
`restir_strategy_gate_summary.json`. Each threshold also has a corresponding
`--gate-max-*` command-line override.

The wavefront backend can trace and average multiple independent camera paths
per pixel entirely in the internal HDR image before reconstruction. Configure
this with `RendererConfig(samples_per_pixel=N)` or the demo environment:

```bash
WAVE_RENDER_SCENE=area_lights WAVE_RENDER_SAMPLES=4 \
WAVE_RENDER_AREA_LIGHT_SAMPLES=2 python -m tools.wavefront_present
```

Camera SPP scales the complete path workload approximately linearly. Area-light
samples scale only direct emissive-light visibility work at diffuse hits, so the
two controls can be tuned independently.

Diffuse wavefront interactions also support explicit cosine-weighted environment
sampling with matching power-heuristic MIS on paths that escape naturally.
`RendererConfig(wavefront_environment_samples=N)` and
`WAVE_RENDER_ENVIRONMENT_SAMPLES=N` accept zero through four samples; zero
restores BSDF-only environment discovery. One sample is the default quality
baseline and adds one visibility query per diffuse interaction.

Secondary diffuse interactions choose area or environment lighting through one
power-weighted mixture draw by default. The technique probability is included
in both the estimator and its BSDF MIS weight, so the mode remains unbiased and
composes with every execution strategy. Square-root domain-power allocation
avoids starving the dim environment while favoring dominant emissive geometry.
The 12-case matrix measured 1.9% lower mean RMSE than independent domain draws
(the worst individual change was +4.3%), while the 4K area-light gate improved
from about 50.5 to 55--56 FPS. Restore independent area and environment draws
with `RendererConfig(wavefront_unified_secondary_nee=False)` or
`WAVE_RENDER_UNIFIED_SECONDARY_NEE=0` for comparison or specialized tuning.

An experimental mixed area/environment primary ReSTIR reservoir is available
with `RendererConfig(wavefront_unified_primary_restir=True)` or
`WAVE_RENDER_UNIFIED_PRIMARY_RESTIR=1`. It is intentionally disabled by
default: the single mixed reservoir increased variance and reduced the 4K gate
to about 49 FPS in current testing. The retained experiment is useful for A/B
work, but production primary reuse should instead keep the lighting domains
stratified (or add environment-only reuse) so one domain cannot evict the
other.

A domain-stratified alternative is available with
`RendererConfig(wavefront_stratified_primary_restir=True)` or
`WAVE_RENDER_STRATIFIED_PRIMARY_RESTIR=1`. It preserves the area-light
reservoir and stores environment candidates in an independent packed history
plane, currently with temporal-center reuse only. This avoids cross-domain
eviction and passes all 12 ReSTIR quality cases. It remains opt-in because the
extra plane increases reservoir storage (about 187 to 312 MiB at 3840x2130).
The environment-specific plane uses an eight-byte encoding with a 24-bit
octahedral direction; this saved about 62 MiB over the original stratified
prototype with no measurable HDR-quality change. The 4K gate remains close to
its 50 FPS threshold, so the mode is not yet a default. Mixed and stratified
primary modes are mutually exclusive.

The current interactive quality baseline is:

```bash
WAVE_RENDER_SCALE=0.5 WAVE_RENDER_SAMPLES=1 \
WAVE_RENDER_AREA_LIGHT_SAMPLES=2 WAVE_RENDER_ENVIRONMENT_SAMPLES=1 \
WAVE_RENDER_TEMPORAL=1 WAVE_RENDER_TEMPORAL_WEIGHT=0.93 \
WAVE_RENDER_DIFFUSE_FILTER=1 WAVE_RENDER_DIFFUSE_FILTER_STRENGTH=0.35 \
python -m tools.wavefront_present
```

Run the first no-readback wavefront swapchain demo with:

```bash
python -m tools.wavefront_present
```

Set a fixed internal-resolution scale for performance and reconstruction tests:

```bash
WAVE_RENDER_WIDTH=3840 WAVE_RENDER_HEIGHT=2160 \
WAVE_RENDER_SCALE=0.75 python -m tools.wavefront_present
```

Enable temporal reconstruction and tune its history weight with:

```bash
WAVE_RENDER_SCALE=0.5 WAVE_RENDER_TEMPORAL=1 \
WAVE_RENDER_TEMPORAL_WEIGHT=0.85 python -m tools.wavefront_present
```

The equivalent library settings are
`RendererConfig(wavefront_temporal_reconstruction=True,
wavefront_temporal_weight=0.85)`. The weight must be at least 0 and less than 1.

Variance-guided temporal confidence can increase reuse in noisy diffuse
regions without applying the spatial diffuse filter to glass or highlights:

```bash
WAVE_RENDER_TEMPORAL=1 WAVE_RENDER_TEMPORAL_WEIGHT=0.85 \
WAVE_RENDER_TEMPORAL_VARIANCE=1 \
WAVE_RENDER_TEMPORAL_VARIANCE_STRENGTH=0.5 \
python -m tools.wavefront_present
```

The equivalent settings are
`wavefront_temporal_variance_confidence=True` and
`wavefront_temporal_variance_strength=0.5`. History reuse increases only when
local diffuse variance is high and current/history luminance remain
statistically compatible; specular and transmission surfaces retain the
existing conservative history weight.

For material-class temporal confidence, enable transmission-aware history:

```bash
WAVE_RENDER_TEMPORAL=1 \
WAVE_RENDER_TEMPORAL_MATERIAL_CONFIDENCE=1 \
WAVE_RENDER_TEMPORAL_TRANSMISSION_SCALE=0.5 \
python -m tools.wavefront_present
```

This uses the packed G-buffer surface class to keep diffuse, metallic, and
transmission policies independent without allocating another 4K history
image. Stable glass can reuse more history, while refracted appearance changes
rapidly reduce its history contribution. The equivalent settings are
`wavefront_temporal_material_confidence=True` and
`wavefront_temporal_transmission_history_scale=0.5`.

For moving-camera disocclusions, enable the fractional 2x2 reprojection
search:

```bash
WAVE_RENDER_TEMPORAL=1 \
WAVE_RENDER_TEMPORAL_REPROJECTION_SEARCH=1 \
python -m tools.wavefront_present
```

`wavefront_temporal_reprojection_search=True` selects the closest compatible
previous position, normal, and surface class around the projected coordinate.
It reduces false history rejection and silhouette flicker without allocating
motion vectors or another history image, but performs up to four previous
G-buffer candidate reads per temporally reconstructed pixel.

Transient fireflies can be suppressed temporally without clamping raw HDR
radiance:

```bash
WAVE_RENDER_TEMPORAL=1 \
WAVE_RENDER_TEMPORAL_OUTLIER_CONFIDENCE=1 \
WAVE_RENDER_TEMPORAL_OUTLIER_STRENGTH=0.75 \
python -m tools.wavefront_present
```

The equivalent settings are `wavefront_temporal_outlier_confidence=True` and
`wavefront_temporal_outlier_strength=0.75`. A current-frame luminance outlier
receives more compatible history only when the previous value agrees with its
eight neighbors. Stable emitters and recurring highlights therefore remain
unchanged, and frames without valid history retain the original HDR sample.

Indirect-light reuse is being introduced behind a separate reservoir ABI. Run
its memory/correctness planning probe with:

```bash
python -m tools.diagnostics.indirect_reuse
```

The contract stores reconnectable secondary-vertex candidates at half
resolution: two 24-byte reservoir planes plus two 4-byte current-sample seed
planes use about 110.7 MiB at 3840x2160. A full-resolution equivalent exceeds
the default budget and is rejected by
the default 128 MiB feature budget. The reference
`IndirectLightReservoir.merge_reconnected()` requires both target
reevaluation and a reconnection Jacobian; direct-light reservoirs are not
silently reused for indirect paths. The compact ABI has CPU round-trip tests
for camera-relative FP16 secondary position/proposal density, octahedral
normal, RGB9E5 incident radiance, FP16 weight/target, and the packed validity
and sample-count header.

The Vulkan storage phase can be exercised independently of sampling with:

```bash
WAVE_RENDER_INDIRECT_REUSE_STORAGE=1 \
WAVE_RENDER_INDIRECT_REUSE_SCALE=0.5 \
WAVE_RENDER_INDIRECT_REUSE_BUDGET_MIB=128 \
python -m tools.wavefront_present
```

This flag allocates and resize-owns the device-local reservoir and seed buffers and
the required full G-buffer, but deliberately does not alter lighting yet.
`RendererConfig.wavefront_indirect_reuse_storage` is named accordingly so the
allocation probe cannot be mistaken for completed indirect reuse. Budget
validation occurs before size-dependent Vulkan images are created; frame stats
report `wavefront_indirect_reservoir_bytes`,
`wavefront_indirect_seed_bytes`, and its reservoir extent. Each new
or resized reservoir plane is zero-initialized exactly once by an isolated
compute pass. Its descriptor and pipeline layouts do not modify the production
tracing layouts, and the plane remains invalid until candidate generation is
implemented.

Candidate generation can be enabled independently with:

```bash
WAVE_RENDER_INDIRECT_REUSE_STORAGE=1 \
WAVE_RENDER_INDIRECT_REUSE_CANDIDATES=1 \
python -m tools.wavefront_present
```

This records the first secondary hit separately from the hot 48-byte path
state and writes one representative indirect contribution per half-resolution
reservoir. Direct lighting is removed from the seed, and transmissive primary
surfaces are excluded. At exact 1x, 1/2x, and 1/4x scales, only pixels that can
seed a reservoir write the optional secondary state.
When wavefront profiling is enabled, the pass is reported separately as
`indirect_candidates` so its cost does not disappear into reconstruction.

Opt-in temporal candidate merging adds reprojection without affecting final
lighting:

```bash
WAVE_RENDER_INDIRECT_REUSE_STORAGE=1 \
WAVE_RENDER_INDIRECT_REUSE_CANDIDATES=1 \
WAVE_RENDER_INDIRECT_REUSE_TEMPORAL=1 \
python -m tools.wavefront_present
```

Reuse is accepted only when the reprojected primary surface agrees in world
position, normal, and material signature. The previous reservoir target is
reevaluated against the current candidate before weighted merging. The strict
same-surface constraint currently uses a unit reconnection Jacobian; the CPU
oracle retains the generalized Jacobian contract for a future relaxed mode.

Spatial bootstrap reuse can be composed on top without allocating another
reservoir plane:

```bash
WAVE_RENDER_INDIRECT_REUSE_STORAGE=1 \
WAVE_RENDER_INDIRECT_REUSE_CANDIDATES=1 \
WAVE_RENDER_INDIRECT_REUSE_TEMPORAL=1 \
WAVE_RENDER_INDIRECT_REUSE_SPATIAL=1 \
python -m tools.wavefront_present
```

Four rotated cross-neighbors are read from the immutable previous-frame plane,
which avoids in-place read/write races and preserves the existing memory
budget. Neighbor candidates require the same material signature, compatible
normal, a bounded world-space separation, and ray-query visibility before
weighted merging.

Apply the selected indirect estimate conservatively with:

```bash
WAVE_RENDER_INDIRECT_REUSE_STORAGE=1 \
WAVE_RENDER_INDIRECT_REUSE_CANDIDATES=1 \
WAVE_RENDER_INDIRECT_REUSE_TEMPORAL=1 \
WAVE_RENDER_INDIRECT_REUSE_SPATIAL=1 \
WAVE_RENDER_INDIRECT_REUSE_APPLY=1 \
WAVE_RENDER_INDIRECT_REUSE_APPLY_STRENGTH=0.35 \
python -m tools.wavefront_present
```

The apply pass adds only the difference between the reconstructed and current
indirect samples, leaving direct lighting and unreused pixels unchanged. It
requires more than one represented sample and verifies the full-resolution
material signature before applying the default correction.

Run the deterministic HDR A/B gate with:

```bash
python -m tests.gates.indirect_quality --gate --output indirect_quality
```

It captures a high-sample reference and matched one-sample conventional/reuse
sequences, then gates relative RMSE, bias, and temporal lag. It writes HDR NPZ
sequences plus JSON and CSV metrics.

Set `WAVE_RENDER_INDIRECT_REUSE_PROFILING=1` to collect sampled GPU counters.
One out of every 64 reservoir pixels contributes to generated/empty totals,
temporal and spatial acceptance, position/normal/material rejection, empty
history, represented sample count, and compact-ABI saturation. Counters are
read only after the corresponding frame fence signals and are included in the
demo output and benchmark JSON under `indirect_reuse`.
Derived temporal/spatial acceptance rates, average represented samples, and
saturation fractions are reported under `indirect_reuse_metrics`.

Reservoir history is bounded by
`wavefront_indirect_reuse_history_limit=32`, configurable with
`WAVE_RENDER_INDIRECT_REUSE_HISTORY_LIMIT`. When a merge exceeds the limit,
both represented sample count and weight sum are scaled by the same factor, so
reservoir normalization is preserved. Profiling reports the affected fraction
as `history_clamp_rate`; the compact 7-bit counter therefore cannot saturate.
During camera motion, the effective limit is reduced so represented history
stays within an approximately 16-pixel screen-space footprint. Configure that
footprint with `wavefront_indirect_reuse_history_motion_pixels` or
`WAVE_RENDER_INDIRECT_REUSE_HISTORY_MOTION_PIXELS`; smaller values reject
history more aggressively and reduce temporal shadows during fast motion.
ReSTIR DI uses the same motion-adaptive policy independently through
`wavefront_restir_history_motion_pixels` and
`WAVE_RENDER_RESTIR_HISTORY_MOTION_PIXELS` (also 16 pixels by default).

Reservoirs can be inspected directly with
`WAVE_RENDER_INDIRECT_REUSE_DEBUG_VIEW`, whose values are `radiance`,
`history`, `validity`, and `acceptance`. The debug pass replaces the displayed
HDR result only when selected; it never feeds production lighting. Acceptance
uses green for temporal reuse and blue for spatial reuse, while position,
normal, material, and empty-history rejection add red, yellow, magenta, and
orange respectively. Neutral gray identifies a valid reservoir with no reuse
event (including the first frame). Invalid storage uses a magenta checkerboard
so it cannot be confused
with solid-red position rejection. The acceptance view should therefore never
be uniformly black. With profiling enabled, interactive runs print counters on
the first three frames and at the report interval even without benchmark mode.
For example:

```bash
WAVE_RENDER_INDIRECT_REUSE_STORAGE=1 \
WAVE_RENDER_INDIRECT_REUSE_CANDIDATES=1 \
WAVE_RENDER_INDIRECT_REUSE_TEMPORAL=1 \
WAVE_RENDER_INDIRECT_REUSE_SPATIAL=1 \
WAVE_RENDER_INDIRECT_REUSE_DEBUG_VIEW=acceptance \
python -m tools.wavefront_present
```

For a conservative diffuse-only cleanup pass, enable the variance-guided
cross-bilateral reconstruction filter:

```bash
WAVE_RENDER_TEMPORAL=1 WAVE_RENDER_TEMPORAL_WEIGHT=0.93 \
WAVE_RENDER_DIFFUSE_FILTER=1 WAVE_RENDER_DIFFUSE_FILTER_STRENGTH=0.35 \
python -m tools.wavefront_present
```

The equivalent settings are `wavefront_diffuse_filter=True` and
`wavefront_diffuse_filter_strength=0.35`. The filter uses world position and
normal agreement, increases its blend only in locally noisy regions, and skips
metal and transmission surfaces. Temporal blending also scales its configured
maximum weight by reprojection position and normal confidence.

`RendererConfig(wavefront_render_scale=...)` accepts values from 0.25 through
1.0. Camera rays and wavefront queues use the scaled extent. Reconstruction is
a separate full-output compute pass, so the built-in temporal reconstruction or
a future third-party reconstruction implementation can be changed without
changing path tracing or swapchain presentation.

On Linux, the wavefront demos default to GLFW's X11/XWayland library variant
to avoid native Wayland `libdecor-gtk` plugin initialization failures. Override
this with `WAVE_RENDER_GLFW_PLATFORM=wayland` when testing native Wayland, or
`native` to restore GLFW's automatic selection.

`VulkanGlfwPresenter.present_wavefront(...)` divides the framebuffer into
bounded tiles, writes each tile directly into a frame-local Vulkan image, and
uses the existing GPU blit and swapchain presentation path. All tile passes and
the final blit are recorded in one command buffer and submitted once per frame.

Wavefront record sizes are an explicit CPU/SPIR-V ABI. In particular, ray
padding is represented as three scalar words so its std430 array stride remains
48 bytes; using an aligned `uvec3` would silently increase the shader stride to
64 bytes and overrun queue allocations at large tile sizes.

The Vulkan executor uses a 48-byte hot path record. Throughput and radiance
remain full-precision RGB values; their otherwise-unused `w` lanes store the
small numeric bounce counter and previous-BSDF PDF. The live RNG word remains
in an integer metadata lane, while the two unused RNG words have been removed.
This avoids floating-point NaN canonicalization and reduces hot path storage
and continuation load/store traffic by 25% without quantization. Its 64-byte nested-IOR stack remains in a
separate cold buffer and is only accessed when required. The packed layout is
exact across all execution strategies, passed the full 12-case ReSTIR matrix,
and measured 60.8 FPS with a 10.23 ms median GPU time at 3840x2130. Scene
descriptor sets are cached until the uploaded TLAS or scene buffers change
rather than being rewritten for every tile.

Hybrid pipelines specialize out generalized MIS, unified-primary ReSTIR, and
stratified-primary ReSTIR when those optional estimators are disabled. The
general pipeline remains available automatically whenever any of those modes
is selected, and `WAVE_RENDER_RESTIR_SPECIALIZATION=0` provides a diagnostic
A/B override. On the target RTX 4070 Laptop GPU, specialization reduced the
driver hybrid binary from 453,632 to 389,248 bytes and improved the matched 4K
gate from 62.76 to 64.91 FPS. It retained exact execution/tile HDR parity and
passed all 12 stationary/moving ReSTIR quality cases.

For opt-in driver pipeline diagnostics, run:

```bash
python -m tools.diagnostics.pipeline_statistics
```

This enables `VK_KHR_pipeline_executable_properties` only for the diagnostic
device and reports executable binary size, register count, shared memory, and
other driver statistics. Normal renderer pipelines do not request statistics
capture.

Animated presentation stores camera vectors in coherent frame-local buffers.
The invariant tiled generate/intersect/shade/tone sequence is recorded once per
frame slot in a reusable secondary command buffer; normal frames only update
64 bytes of camera data and record the swapchain blit. Benchmark output reports
whether this command cache was built or hit.

Before every bounce, the GPU indirect-preparation shader derives dispatch size
from the active input count. A zero active count produces zero-work intersection
and shading dispatches, while nested glass paths remain free to consume the full
configured bounce depth. Continuation headers use transfer fills; benchmarks on
the target NVIDIA GPU found these faster than resetting them in the compute
preparation shader.

For aggregated GPU stage profiling, override the benchmark resolution, queue
capacity, or bounce depth as needed:

```bash
WAVE_RENDER_BENCHMARK_FRAMES=10 WAVE_RENDER_WIDTH=3840 \
WAVE_RENDER_HEIGHT=2160 WAVE_RENDER_TILE_CAPACITY=1048576 \
python -m tools.wavefront_present
```

Benchmark mode holds the camera fixed during ten warm-up frames, then covers
one deterministic orbit over the measured frames. Override the coverage with
`WAVE_RENDER_BENCHMARK_ORBITS`; this makes performance traces comparable across
machines and renderer revisions regardless of their frame rate.

Run the target-machine 4K performance gate with:

```bash
WAVE_RENDER_GLFW_PLATFORM=wayland tests/gates/run_4k_performance.sh
```

The procedural room scenes use a front-facing camera arc for the release gate.
Older measurements used a full 360-degree orbit around rooms that are open on
one side; many frames therefore viewed little geometry and overstated sustained
throughput. `WAVE_RENDER_FULL_ORBIT=1` reproduces that legacy workload for
diagnosis only and must not be used for release qualification. On the RTX 4070
test system, the current renderer measured 67.33 FPS / 10.41 ms median GPU on
the legacy orbit but 39.06 FPS / 20.14 ms on the representative front arc.

After changing queue layouts, workgroup sizes, indirect dispatch, or path-state
storage, run the transmission-heavy tile quality gate as well:

```bash
python -m tests.gates.tile_quality
```

It renders the same deterministic nested-glass scene once as a single tile and
once with many production-style tiles. The gate requires exact HDR parity and
reports horizontal-band magnitude and anisotropy, catching skipped continuation
work or tile-boundary corruption that execution-strategy parity alone can miss.

The gate requires at least 50 sustained FPS and at least 98% of the requested
3840×2160 pixel count after compositor decorations. It exits nonzero on failure
and writes its trace and machine-readable result to
`/tmp/ordinarylight_4k_gate.csv` and `/tmp/ordinarylight_4k_gate.json` by default.
The scale-adjusted initial window avoids a transient 4800×2700 allocation on
the target 125%-scaled Wayland desktop, then maximizes after initial allocation.
The release-throughput gate disables detailed GPU work counters because their
global atomic increments materially perturb ReSTIR-heavy frames. Use the normal
benchmark command with `WAVE_RENDER_PROFILE=1` for stage/workload diagnosis;
that diagnostic result is not the release FPS gate.

A completed update may explicitly accept a regression only by supplying both
an override flag and a reason; the JSON records the exception:

```bash
WAVE_RENDER_PERFORMANCE_GATE_ALLOW_FAILURE=1 \
WAVE_RENDER_PERFORMANCE_GATE_OVERRIDE_REASON="reason for accepted regression" \
WAVE_RENDER_GLFW_PLATFORM=wayland tests/gates/run_4k_performance.sh
```

Changing `WAVE_RENDER_PERFORMANCE_GATE_MIN_FPS` is intended for separate
hardware baselines, not for bypassing a regression on the target GPU.

The output separates `generate`, `intersect`, `shade`, `resolve`, `tone`, and
`present` GPU timestamp durations across every tile and bounce. It also reports
CPU time for scene preparation, swapchain recreation, image acquisition,
command recording, submission, and presentation. Benchmark mode enables stage
profiling automatically; set `WAVE_RENDER_PROFILE=0` to disable the detailed
timestamps while retaining the CPU and whole-frame timings.
Profiling also reports primary-kernel work counters for path rays, shadow rays,
texture evaluations, surface hits, and environment misses. Their scope is
labelled `full_path` for the megakernel, `inline_prefix` for hybrid execution,
and `primary_only` for the compacting wavefront implementation.
Set `WAVE_RENDER_BENCHMARK_CSV=/path/to/trace.csv` to retain per-frame GPU,
cadence, wait, stage, and workload-counter values. The terminal summary reports
each counter's correlation with GPU time and its change between the fastest and
slowest ten percent of measured frames.

Intersection and shading use GPU-generated indirect dispatch arguments. Before
each bounce, `wavefront_prepare_indirect.comp` converts the compacted active-ray
count to 64-thread workgroup dimensions; later bounces therefore scale with
surviving paths instead of the configured queue capacity.

`RenderPipeline` schedules ordered `RenderStage` objects. Stages declare their
logical inputs and outputs, so missing resources and duplicate stage names are
rejected before commands are recorded:

```python
pipeline = ol.RenderPipeline(
    (
        ol.RenderStage("trace", reads={"scene"}, writes={"radiance"}),
        ol.RenderStage("denoise", reads={"radiance"}, writes={"filtered"}),
        ol.RenderStage("present", reads={"filtered"}, writes={"swapchain"}),
    ),
    initial_resources={"scene"},
)
```

The direct Vulkan backend now executes its commands through this scheduler.
Its stages are `trace_temporal`, optional `denoise`, `tone_map`, and `present`; query them through
`VulkanGlfwPresenter.pipeline_stages`. Tone mapping and the FPS overlay execute
in their own compute pass over the HDR history image. A denoiser can therefore
be inserted between `trace_temporal` and `tone_map` without changing tracing or
swapchain presentation. Temporal reconstruction remains fused with tracing and
is the next candidate for extraction.

### À-trous denoising

Enable the edge-aware HDR denoiser with:

```python
config = ol.RendererConfig(
    progressive_accumulation=True,
    temporal_history=True,
    denoiser_enabled=True,
    denoiser_iterations=3,
)
```

The GLFW equivalents are `WAVE_RENDER_DENOISER=1` and
`WAVE_RENDER_DENOISER_ITERATIONS=1..5`. Set the relative-variance activation
threshold with `RendererConfig(denoiser_variance_threshold=...)` or
`WAVE_RENDER_DENOISER_VARIANCE` (default `0.01`). Stable diffuse pixels remain
raw, moderately noisy pixels blend partially, and only high-variance pixels
receive the full filter. The filter uses temporal luminance
variance plus depth, normal, and primary-surface-class guidance between
temporal tracing and tone mapping. Mirror, transmissive, and emissive primary
surfaces bypass spatial filtering. It allocates two additional RGBA32F
ping-pong images (about 253 MiB total at 4K). The area-light showcase defaults
to two passes and the `D` key toggles raw/denoised presentation for immediate
quality and performance comparison. Separable kernels and per-stage timestamps
are planned optimizations.
Set `WAVE_RENDER_DENOISER_START_RAW=1` to launch with the denoiser configured
but raw HDR selected.

Select **Python-authored materials** in the workbench:

```bash
ordinarylight-workbench
```

It renders a cosine-sampled diffuse floor, a perfect mirror, and two
concentric stochastic Fresnel-glass shells. This exercises multiple Python
program dispatch, per-bounce random inputs, PDF weighting, total internal
reflection, the nested-medium stack, and progressive HDR accumulation in one
scene. The camera starts stationary so convergence is visible. Space toggles
orbiting, C clears accumulated history, and R resets the camera orbit.

### Cross-scene validation

`tests.gates.validation_matrix` composes deterministic HDR, ReSTIR,
execution-strategy, memory, and native-4K performance gates across area-light,
diffuse, glossy/glass, textured, small-emitter, occlusion, nested-glass, and
dense-geometry scenes:

```bash
python -m tests.gates.validation_matrix --output validation_matrix
```

Select a cheaper subset while developing with `--scenes` and `--stages`:

```bash
python -m tests.gates.validation_matrix \
  --scenes textured nested_glass dense \
  --stages indirect parity
```

Quality stages retain HDR sequences and per-frame CSV data through their
specialized capture tools. The top-level `report.json` records each status,
elapsed time, summary path, and aggregate metrics. By default, the 4K stage
uses an undecorated 3072x1728 Wayland window on a 125% desktop scale and
verifies that its framebuffer is within 98% of 3840x2160. Override the
logical-extent and platform arguments for other desktop configurations.
Procedural room scenes use a front-facing presentation arc because their -Z
side is intentionally open; imported and freestanding scenes retain full
turntable orbits. Execution-parity captures also fail if fewer than 1% of
their pixels contain visible radiance, preventing all-black comparisons from
passing vacuously.

Temporal reconstruction, ReSTIR DI, and indirect reuse reject history when
camera rotation would move the image by more than 64 pixels between frames.
Override this conservative limit with
`WAVE_RENDER_TEMPORAL_MOTION_LIMIT_PIXELS`; the measured displacement and
validity decision are exposed as `wavefront_temporal_motion_pixels` and
`wavefront_temporal_motion_valid`. The ReSTIR matrix includes a `fast` camera
case to keep this behavior under objective HDR regression coverage.
The primary sampler hashes the frame sequence exactly once. Earlier builds
also XORed the same value from camera metadata, cancelling all frame variation
and pinning a structured 1-spp noise field to the screen during camera motion.
Temporal-quality reports include `low_frequency_energy_ratio` to expose broad
spatial structure independently of total variance.

### Deferred noise-quality follow-up

Interactive noise remains an explicit follow-up after the Qt workbench and
general-purpose API work.  Revisit the moving-camera area-light and
glossy/glass scenes at native resolution, with particular attention to diffuse
wall noise, specular/transmission paths, and temporal stability at direction
reversals.  Candidate improvements must be evaluated against HDR reference
captures and the native-4K performance gate; reducing noise by blurring detail,
introducing ghosting, or silently lowering render resolution is not an
acceptable quality tradeoff.
