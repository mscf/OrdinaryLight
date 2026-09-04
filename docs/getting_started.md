# Using Ordinary Light

Install the portable core for the CPU reference backend, loaders, scene API,
NumPy outputs, and backend contracts:

```bash
python -m pip install ordinarylight
```

Install hardware rendering or an application integration explicitly:

```bash
python -m pip install 'ordinarylight[vulkan]'
python -m pip install 'ordinarylight[qt]'
python -m pip install 'ordinarylight[window]'
```

## Headless stills and named products

`Renderer.render()` returns linear HDR `float32` pixels. It can reuse a
caller-owned array, making ownership and allocation explicit:

```python
import numpy as np
import ordinarylight as ol

backend = ol.backends.ReferenceBackend(samples_per_pixel=4, seed=7)
with ol.Renderer(backend=backend) as renderer:
    destination = np.empty((720, 1280, 4), np.float32)
    renderer.render(scene, camera, (1280, 720), out=destination)
```

Backends advertise optional products through `renderer.capabilities`. Request
only supported products:

```python
if renderer.capabilities.supports_output("depth"):
    frame = renderer.render(
        scene, camera, (1280, 720), outputs=("color", "depth", "object_id")
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
