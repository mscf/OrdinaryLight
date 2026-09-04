# Using Ordinary Light

From the Ordinary Light checkout, install the sibling Ordinary Shade compiler
and the portable core for the CPU reference renderer, loaders, scene API, and
NumPy outputs. Python 3.10 or newer is required:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ../ordinaryshade
python -m pip install -e .
```

Install hardware rendering or an application integration explicitly from the
same checkout:

```bash
python -m pip install -e '.[vulkan]'
python -m pip install -e '.[webgpu]'
python -m pip install -e '.[qt]'
python -m pip install -e '.[window]'
```

Vulkan paths require Vulkan 1.2; GI additionally requires hardware ray-query
support. `Renderer(renderer_preference="auto")` selects GI when available and
Vulkan raster otherwise. Explicit `"gi"` and `"raster"` requests do not switch
renderer families silently.

## Headless stills and named products

`Renderer.render()` returns linear HDR `float32` pixels. It can reuse a
caller-owned array, making ownership and allocation explicit:

```python
import numpy as np
import ordinarylight as ol

scene = ol.Scene()
camera = ol.PerspectiveCamera((0, 0, -5), (0, 0, 0))
implementation = ol.renderers.reference.CpuReferenceRenderer(
    samples_per_pixel=4, seed=7,
)
with ol.Renderer(implementation=implementation) as renderer:
    destination = np.empty((720, 1280, 4), np.float32)
    renderer.render(scene, camera, (1280, 720), out=destination)
```

Renderers advertise optional products through `renderer.capabilities`. Within
an open renderer context, request only supported products:

```python
outputs = ("color", "depth", "object_id")
if all(renderer.capabilities.supports_output(name) for name in outputs):
    frame = renderer.render(
        scene, camera, (1280, 720), outputs=outputs
    )
```

## Asynchronous rendering

`render_async()` returns an ordered `RenderJob`. It supports polling,
callbacks, cancellation before execution, blocking `result()`, and `await`:

```python
job = renderer.render_async(scene, camera, (1280, 720))
job.add_done_callback(lambda completed: queue.put(completed.result()))

async def capture():
    return await renderer.render_async(scene, camera, (1280, 720))
```

## Video output

Encoding is separate from rendering. `FFmpegVideoWriter` accepts display-ready
RGB/RGBA arrays and can target a path or binary stream:

```python
with ol.outputs.FFmpegVideoWriter("result.mp4", size, fps=30) as writer:
    for camera in cameras:
        hdr = renderer.render(scene, camera, size)
        writer.write(ol.outputs.to_sdr(hdr))
```

## Materials and scene updates

Python material functions construct typed expressions that a capable backend
compiles; Python does not execute once per GPU hit. See
`examples/custom_material.py`. Meshes, instances, lights, textures, and volumes
retain stable identities and are updated through `Scene`; see
`examples/scene_updates.py`.

## Qt workbench extensions

The workbench is an optional feature browser, not part of the renderer core.
Point `ORDINARYLIGHT_SHOWCASE_PATH` at scripts declaring `SHOWCASE` or
`SHOWCASES`:

```bash
ORDINARYLIGHT_SHOWCASE_PATH=examples/workbench_showcase.py \
ordinarylight-workbench
```

The workbench builds scenes and renderers away from the Qt event thread and
owns native Vulkan presentation on its render worker. Startup and target
switches expose an indeterminate progress state; close requests wait
asynchronously for a safe GPU boundary. Application code should share the
semantic `Scene` and camera resources, not reach into the workbench internals.
