# Public API organization

`ordinarylight` keeps its root namespace intentionally small and convenient while
semantic subpackages own extensible feature families.  Root-level names are
stable shortcuts, not alternate implementations.

## Namespace ownership

- `ordinarylight.renderer` owns the high-level `Renderer`, `RenderFrame`, and
  statistics contract.
- `ordinarylight.scene` owns scene resources and mutation semantics.
- `ordinarylight.cameras` owns camera models.
- `ordinarylight.lights` owns analytic light resources.
- `ordinarylight.animations` owns keyframe tracks, clips, and playback.
- `ordinarylight.loaders` owns external asset and scene ingestion. Each format
  module exposes `load()`; descriptive aliases such as `load_gltf()` may also
  be exported by the package.
- `ordinarylight.materials` owns material programs and their symbolic language.
- `ordinarylight.primitives` owns point, line, and glyph convenience resources.
- `ordinarylight.integrations` owns optional window-system and GUI adapters.
- `ordinarylight.outputs` owns display conversion and encoded headless sinks.
- `ordinarylight.backends` owns backend implementations and their portable
  configuration objects.
- Backend-specific controls remain in their backend module and must not leak
  into backend-neutral scene resources.

The canonical glTF API is:

```python
import ordinarylight as ol

scene = ol.loaders.gltf.load("scene.glb")
```

Applications that accept several formats can use semantic dispatch without
importing a format module:

```python
scene = ol.loaders.load("scene.glb")
print(ol.loaders.supported_formats())
```

`ol.loaders.load_gltf()`, `ol.load_gltf()`, and
`ordinarylight.gltf.load_gltf()` are compatibility aliases to that same function.

## Rules for new APIs

1. Put a feature in the namespace that owns its semantics, not the module that
   first needs it.
2. Keep one implementation and make compatibility paths aliases or thin
   forwarding modules.
3. Avoid importing optional GUI or window dependencies from the package root.
4. Represent backend support through capability discovery and validation,
   rather than backend checks in application code.
5. Preserve stable resource identity and explicit mutation through `Scene`.
6. Add namespace, compatibility, validation, and snapshot tests with every
   public resource family.

## Headless outputs

`Renderer.render()` returns linear HDR NumPy arrays and accepts caller-owned
`out=` arrays. Presentation and encoding remain separate output concerns:

```python
with ol.outputs.FFmpegVideoWriter("render.mp4", (width, height), fps=30) as video:
    for camera in cameras:
        hdr = renderer.render(scene, camera, (width, height))
        video.write(ol.outputs.to_sdr(hdr))
```

The writer also accepts a binary file-like destination and streams compressed
Matroska bytes into it. FFmpeg is optional and is only required when the writer
is instantiated; importing `ordinarylight` does not launch or require it.
The converter returns RGB by default because the renderer's fourth HDR channel
is reserved metadata rather than alpha; pass `alpha=True` for genuine RGBA
input from another source.

## Non-blocking rendering

`Renderer.render_async()` returns a `RenderJob` immediately. Jobs submitted to
one renderer are ordered and never call a mutable backend concurrently. This
makes the API safe for notebooks, GUI event loops, and producer/consumer output
pipelines without exposing Vulkan synchronization:

```python
job = renderer.render_async(scene, camera, (1920, 1080))
# Keep servicing the application while Ordinary Light renders.
hdr = job.result()

# RenderJob is also awaitable inside an asyncio application.
hdr = await renderer.render_async(scene, camera, (1920, 1080))
```

The current Vulkan headless implementation serializes GPU submission and
readback on the renderer worker. It removes caller-thread stalls, but it does
not yet overlap several headless readbacks on the GPU. Native presentation uses
frames in flight and avoids host pixel readback.

## Workbench showcases

`ordinarylight-workbench` is the sole interactive feature browser. Built-in
scenes and third-party extensions use the same declarative `Showcase` contract.
Put Python scripts in a directory listed by `ORDINARYLIGHT_SHOWCASE_PATH`, then
use **Reload scripts** in the workbench. Each script defines `SHOWCASE` or
`SHOWCASES`; scene construction is lazy and runs away from the Qt event thread.

## Backend portability

Backends implement the structural `ordinarylight.backends.RenderBackend`
contract: `render_frame(...)`, `close()`, and semantic capability metadata.
Backends with named products additionally implement `render_products(...)`.
The contract deliberately contains no wavefront, Vulkan, queue, or swapchain
terminology.

Applications can use the same high-level renderer with the Vulkan backend or
the deterministic CPU reference backend:

```python
backend = ol.backends.ReferenceBackend(ol.backends.ReferenceConfig(seed=7))
renderer = ol.Renderer(backend=backend)
print(renderer.capabilities.as_dict())
```

The reference backend prioritizes correctness and portability over throughput.
Use `renderer.capabilities.supports(...)` to select optional behavior rather
than testing backend class names.

Vulkan-specific construction is canonical under `ordinarylight.backends.vulkan`:

```python
config = ol.backends.vulkan.RendererConfig(max_bounces=8)
backend = ol.backends.vulkan.VulkanRayTracingBackend(config=config)
renderer = ol.Renderer(backend=backend)
```
