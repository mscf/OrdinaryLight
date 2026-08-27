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

## GPU-resident output

Linux Vulkan applications can opt into externally shareable output and avoid
NumPy readback entirely:

```python
config = ol.RendererConfig(external_image_interop=True)
with ol.Renderer(config=config) as renderer:
    frame = renderer.render_gpu(scene, camera, (1920, 1080),
                                pixel_format="rgba8")
    memory_fd = frame.export_memory_fd()
    ready_fd = frame.export_ready_semaphore_fd()
    # Import both opaque FDs into CUDA. Wait on ready_fd in the CUDA stream,
    # consume the image, synchronize that consumer, then release the slot.
    frame.close()
```

`render_gpu()` records and submits work but does not wait for a pixel copy. Its
current Vulkan product is a dedicated, optimal-tiled, tone-mapped
`VK_FORMAT_R8G8B8A8_UNORM` image in `VK_IMAGE_LAYOUT_GENERAL`. The frame
metadata includes the Vulkan handles, allocation size, physical-device UUID,
extent, format, layout, and completion fence. The opaque memory and binary
semaphore FDs are exported on demand.

FD ownership transfers to the caller. CUDA consumes an FD after a successful
external-memory or external-semaphore import; close it explicitly if import
fails. `GpuFrame.close()` returns the Ordinary Light frame slot and is
idempotent, but the caller must first ensure that the consumer has stopped
using the allocation. Two exported frames may be outstanding. Advanced
consumers can export the release semaphore, queue its signal after consuming a
frame, mark that signal with `frame.mark_external_release_scheduled()`, and
close the frame immediately. Vulkan then waits on the GPU without a CPU stall.

For H.264, request NV12 and use the optional NVENC sink:

```python
config = ol.RendererConfig(external_image_interop=True)
with ol.Renderer(config=config) as renderer, \
     ol.outputs.NvencVideoWriter(
         "render.h264", (1920, 1080), fps=30
     ) as video:
    for index, camera in enumerate(cameras):
        frame = renderer.render_gpu(
            scene, camera, (1920, 1080), frame_index=index,
            pixel_format="nv12",
        )
        video.write(frame)
```

Install this optional path with `pip install ordinarylight[video-gpu]`. The
final Vulkan compute pass performs BT.709 limited-range RGBA-to-NV12 conversion
into a dedicated pitch-linear external buffer. CUDA imports each of the two
stable allocations and semaphore pairs once; steady-state frames use only GPU
wait/signal operations and NVENC. There is no GPU-to-CPU readback, NumPy HDR
array, CPU tone mapping, CPU YUV conversion, or CPU-to-GPU upload. The writer
accepts either a path or a binary file-like stream and emits an H.264 elementary
stream.

For a 10-bit path, request the distinct P010 product and configure NVENC for
HEVC or AV1:

```python
with ol.outputs.NvencVideoWriter(
    "render.h265", (1920, 1080), fps=30,
    codec="hevc", pixel_format="p010",
) as video:
    frame = renderer.render_gpu(
        scene, camera, (1920, 1080), pixel_format="p010"
    )
    video.write(frame)
```

P010 metadata reports `bit_depth=10`, `storage_bits=16`, a byte pitch twice
the image width for tightly packed output, and MSB-aligned 10-bit samples. Its
conversion reads the linear HDR render target directly, avoiding an
intermediate RGBA8 quantization step.

### Resident scene and settings transitions

Do not create a new `Renderer` when a stream changes scenes. Replace resident
scene resources transactionally while retaining the Vulkan device, compiled
pipelines, swapchain/headless images, and exported NV12/P010 pool:

```python
renderer.replace_scene(next_scene)
frame = renderer.render_gpu(
    next_scene, next_camera, (1920, 1080), pixel_format="nv12"
)
```

Already-submitted asynchronous work completes first. If building `next_scene`
fails, the previous resident scene remains valid. The deterministic frame
sequence and temporal history reset by default; an encoder should request an
IDR frame at the same application-level transition.

Common shader parameters can also change without rebuilding Vulkan:

```python
renderer.reconfigure(
    samples_per_pixel=2,
    max_bounces=8,
    wavefront_exposure=1.1,
    wavefront_render_scale=0.75,
)
```

Structural settings deliberately raise an error saying renderer recreation is
required. This makes a device-loss/recreation fallback explicit and keeps the
normal transition path zero-copy.

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
