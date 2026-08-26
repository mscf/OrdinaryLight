# General-purpose renderer API direction

`ordinarylight` is a general-purpose rendering library. Applications decide what
their content means and how it should be presented. The public API therefore
avoids domain-specific data models, plotting conventions, UI frameworks, and
notebook dependencies. Scientific visualization is one downstream use case,
not an organizing abstraction in the renderer.

## Design contract

- **Array friendly.** Geometry, attributes, textures, lookup-table samples,
  and rendered HDR images accept or return documented NumPy shapes and dtypes.
- **Headless by default.** Rendering must not require GLFW, a swapchain, or an
  event loop. Window and notebook presentation are optional integrations.
- **One engine, multiple targets.** Offscreen arrays, native Vulkan viewports,
  and a future streamed Jupyter widget must consume the same scene and renderer
  state. A notebook integration must not become a second rendering backend.
- **Explicit color handling.** The renderer returns linear HDR radiance.
  Display transforms must be opt-in so quantitative color mappings are not
  silently altered for any color-critical or quantitative workflow.
- **Stable resources.** Callers need scene handles and explicit update methods
  for positions, attributes, transforms, materials, and time-varying fields;
  changing a camera must not rebuild geometry.
- **Composable capabilities.** Triangle meshes, instancing, glyphs, lines,
  points, structured/unstructured fields, and volumes should share scene,
  camera, lighting, material, and output abstractions rather than separate
  renderer classes.
- **Allocation control.** Render calls should support caller-owned arrays and,
  eventually, device-buffer interoperability without making it necessary for
  ordinary Python callers to manage Vulkan synchronization.
- **Determinism and introspection.** Explicit frame/sample indices, device
  information, timing data, and capability queries are part of the supported
  API so scientific results and performance measurements can be reproduced.
- **Inspectable outputs.** Color is only one renderer product. Depth, normals,
  object/material IDs, motion, variance, radiance components, and future
  domain-specific compute buffers need named, explicitly requested readback.

## Current stable shape

```python
with ordinarylight.Renderer(**options) as renderer:
    hdr = renderer.render(scene, camera, (width, height))
```

`hdr` has shape `(height, width, 4)` and dtype `float32`; RGB contains linear
HDR radiance. `out=` reuses a matching caller-owned array. A supplied backend
may implement the same protocol, keeping the façade independent of Vulkan.

Scenes currently contain validated triangle meshes and point or emissive-area
lights. Python-authored material programs compile to GPU shader behavior, and
glTF supplies an interchange route for conventional assets.

Meshes and point lights now have stable scene-local integer IDs. Validated
`Scene.update_mesh()` and `Scene.update_point_light()` calls preserve the
Python handle, advance separate geometry/shading revisions, and invalidate
resident Vulkan resources. Removal and clearing retain monotonic ID allocation.
Meshes retain object-space arrays and an immutable, composable affine
`Transform`; transform motion advances its own revision. Equal-layout material,
light, and attribute changes now use one batched device-buffer upload while
reusing BLAS/TLAS resources. The Vulkan backend builds one object-space BLAS
per mesh. A transform-only edit updates the corresponding TLAS instance data
and rebuilds the TLAS in place without rebuilding mesh geometry. Topology and
buffer-layout changes retain the conservative full-rebuild path. Equal-topology
position updates reuse object/world vertex allocations, refit only the affected
BLAS, and refresh the existing TLAS allocation for meshes declared with
`deformable=True`. Static meshes omit Vulkan's update capability so their
acceleration structures retain maximum tracing performance.

## Presentation boundary

The core renderer should support these modes without changing scene ownership:

```python
# Headless/offscreen: suitable for notebooks, tests, and parameter sweeps.
hdr = renderer.render(scene, camera, (1024, 1024))

# Native viewport: a separate integration owns its window and swapchain while
# Python remains the control surface.
viewport = renderer.open_viewport(scene, camera, size=(1280, 720))  # planned

# Embedded notebook: an optional package transports rendered frames and input
# events through a custom widget; it calls the same offscreen renderer.
widget = renderer.widget(scene, camera, size=(800, 600))            # planned
```

The first mode exists today. Native presentation currently uses the lower-level
`VulkanGlfwPresenter`; the high-level viewport object is intentionally deferred
until scene resources have stable identities. The widget belongs in an optional
integration module so importing `ordinarylight` never imports Jupyter.

Named output selection should eventually make diagnostics analyzable rather
than merely visible:

```python
frame = renderer.render(
    scene, camera, (1024, 1024),
    outputs=("color", "depth", "normal", "instance_id", "variance"),
)
np.asarray(frame["depth"])
```

The structured boundary now exists: explicitly requesting outputs returns an
immutable mapping-like `RenderFrame` with frame metadata and supports `color`
and per-call sample `variance`. The legacy call without `outputs=` continues to
return its HDR array directly. `renderer.available_outputs` permits capability
discovery. Native primary-hit `depth` and world-space geometric `normal` are
also exposed without deriving them from color. Background depth is `inf`, its
normal is zero, and sample zero defines primary-hit products for multisample
renders. Stable scene-local `instance_id` and `material_id` products use the
`uint32` background sentinel `0xffffffff`; `object_id` remains a compatibility
alias for `instance_id`. `motion` is available as current-
minus-previous screen displacement in output pixels for camera, rigid-transform, and
equal-topology deformation motion. History advances on calls requesting this
product and is cleared by `reset_sequence()`.

Each completed high-level render now exposes an immutable structured record as
`renderer.last_statistics`; named `RenderFrame` results also include it under
`frame.metadata["statistics"]`. Its normalized `total_ms` and `gpu_ms` fields
are accompanied by the complete backend timing map, while `as_dict()` returns
a flat row suitable for a pandas table or benchmark JSON. This avoids parsing
UI titles or console text while retaining backend-specific diagnostic detail.

## Capability sequence

The next renderer-level additions should be implemented in this order:

1. Extend the existing partial GPU uploads and object transforms to multi-BLAS
   instancing, TLAS-only transform updates, and acceleration-structure refits.
   **Implemented.** Hardware instancing that lets several objects share one
   mesh allocation is implemented by item 5.
2. Named render outputs and structured benchmark results, built on declared
   render-graph resources.
3. A high-level native viewport/controller that shares the headless renderer's
   scene and camera state. **Initial GLFW integration implemented** as
   `ordinarylight.integrations.glfw.NativeViewport`, with managed and stepped
   event loops, controller callbacks, resize stabilization, and structured
   per-frame statistics.
4. Per-vertex generic attributes, beginning with color and scalar channels,
   plus renderer-owned one-dimensional lookup textures for transfer functions.
   **Backend-neutral foundation implemented:** meshes accept immutable named
   float channels through `attributes=`, expose built-in and custom channels
   through `vertex_attribute()`, and support atomic revision-tracked updates.
   Immutable HDR `Texture1D` resources provide clamp/repeat/mirror addressing
   and linear or nearest lookup without attaching domain-specific range rules.
   `VertexAttributeLayout` selects only shader-requested channels and packs a
   stable aligned vec4 ABI, keeping the default vertex record unchanged.
   Python materials declare dependencies with
   `ctx.attribute("name", components=N)`; shader generation validates these
   against a layout and specializes symbolic channel names to stable slots.
   Vulkan scene resources now allocate and partially update the packed custom
   attribute buffer only when active materials declare such dependencies. The
   runtime-compiled ray-query path binds that buffer at an opt-in descriptor,
   interpolates the selected vec4 slots at the committed hit, and exposes the
   requested one-to-four components to Python-authored material expressions.
   Deterministic `MaterialEvaluation` programs now have equivalent dynamically
   compiled primary and continuation/shade pipelines in staged wavefront
   execution. Attribute scenes select this compatible staged strategy instead
   of opaque, untextured, fused, persistent, or SER specializations. Stochastic
   `SurfaceResponse` parity remains separate because it must preserve path RNG
   and medium-stack transitions exactly.
5. **Implemented.** General-purpose mesh resources and lightweight instances,
   including stable instance IDs, per-instance transforms/materials/visibility,
   atomic batched updates, shared BLAS construction, TLAS-only placement
   updates, named `instance_id` output, and dense-scene memory/performance
   regression gates. Repeated glTF nodes preserve that sharing, column-oriented
   creation/update APIs avoid per-object change dictionaries, and JSON-friendly
   scene snapshots expose stable identity and source metadata without copying
   geometry arrays.
6. **Implemented.** Backend-neutral finite world-space point, line, and
   arbitrary mesh-glyph batches lowered through shared mesh resources. They
   retain general materials, stable instance IDs, named outputs, snapshots,
   atomic animation updates, and the existing Vulkan BLAS/TLAS path without
   introducing domain-specific plotting semantics.
7. **Implemented.** Structured scalar volumes with immutable dense NumPy data,
   explicit normalization ranges, affine placement, stable identity, RGBA
   transfer functions, emission--absorption materials, CPU-reference sampling,
   native Vulkan buffer upload and shader ray marching, mixed mesh/volume
   composition, snapshots, mutation tracking, and standalone showcases/gates.
   Volume materials also expose opt-in point-light single scattering with
   scattering color/strength and normalized isotropic or Henyey--Greenstein
   phase functions. Native Vulkan extends the estimator to emissive triangles,
   environment lighting, source-path attenuation, and opaque visibility.
   Orders 2--8 optionally add a separately specialized, energy-bounded local
   multiple-scattering closure controlled by RGB albedo and optical depth. The
   default order 1 remains the validated direct estimator with no added shader
   cost, while zero scattering strength retains emission--absorption only and
   uses the smaller kernels.
8. Optional Jupyter frame/event transport layered over offscreen rendering.
9. Optional interoperability adapters for array ecosystems such as DLPack,
   without making them core dependencies.

Each addition should preserve the existing HDR quality and 4K performance
gates where applicable. Convenience plotting APIs and dataset-specific readers
belong in the consuming scientific project.
