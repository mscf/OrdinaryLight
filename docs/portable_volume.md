# Portable vector-histogram volume milestone

The experimental `ordinarylight/portable-volume-v1` package prepares the RT
example without a GPU device and executes its exported WGSL compute and volume
passes in a browser. Scope is deliberately limited to scalar-parameter batches,
a selected floating vector output, its histogram, and a buffer-backed volume.

## Project boundaries

| Project | Entry point | Responsibility |
| --- | --- | --- |
| OrdinaryShade | `ordinaryshade.portable.export_shader` | Versioned WGSL, entry point, workgroup and binding reflection, source digest. |
| OrdinaryLattice | `ordinarylattice.portable.export_batched` | Ordered kernels, allocations, output identities, little-endian parameter data, and live/structural parameter classification. |
| OrdinaryScience | `VectorHistogramPreparation.prepare()` / `.export()` | Device-free compilation, selected vector output, histogram metadata and restoration state. `reprepare_volume` handles structural edits. |
| OrdinaryLight | `PortablePackage` / browser `PortableRuntime` | Package validation, native compute execution, browser execution and volume presentation on shared buffers. |
| LatticeModel | Existing `ProgramSpec` | Serialized model source; no new API needed for the supported path. |

`PreparedBatchedModel.prepare` creates host resource descriptions and compiled
steps. `BatchedModelSession` creates the native executor afterwards. The RT
controller exposes `prepare_volume()` and `export_portable()` through the same
preparation path. These evolving upstream APIs are not yet available as
coordinated published releases. Use the pinned wheel build below or compatible
development checkouts.

## Package and ownership contract

`PortablePackage.write(directory)` writes `manifest.json` and `payload.bin`.
The JSON contains producer versions, required GPU features/limits, shader
sources and reflection, ordered compute passes, named resources and outputs,
scientific metadata, initial parameters and presentation state. Payload and
shader digests detect corruption. Version mismatches and unsupported layouts
are rejected before allocating browser resources.

Resources are little-endian 32-bit scalar storage buffers with explicit shapes,
byte lengths and optional initial byte ranges. Compute bindings refer to the
same resource IDs used by rendering. Pass dependencies must refer to earlier
passes; the executor submits them in listed order. There are no serialized
Python objects, device pointers or native buffer handles. The allocation budget
is 512 MiB, with a 128 MiB maximum individual buffer, subject to device limits.

The browser owns allocated buffers, pipelines and render targets. It owns a
device only when it creates one; a supplied device remains host-owned. The
histogram density buffer is read directly by the volume shader on that device.
Normal compute-to-render operation performs no CPU readback. `read()` and
`readPixels()` are explicit diagnostics. Pixel captures always return RGBA even
when the preferred canvas format is BGRA.

The volume description fixes the density layout to `[z, y, x]`, nearest-cell
sampling, an RGBA transfer buffer, a fullscreen triangle and a 64-byte settings
uniform. Its offsets are viewport/opacity/maximum at 0, camera at 16, slice at
32, and dimensions at 48. General textures, arbitrary scene graphs and remote
rendering executors are outside this first format.

## Updates and restoration

- **Live parameters** are packed into existing parameter-buffer rows; compute
  reruns without replacing pipelines. Grid-axis values come from exported samples.
- **Structural parameters** affect shapes or code and require preparation.
  Grid size, histogram bins, selected output and axis selection also require a
  replacement. `PortableViewer` invokes the host's `onReprepare(snapshot, changes)`
  callback, builds the replacement on the existing device, and retains the old
  runtime if preparation fails.
- **Presentation values** change camera, opacity, normalization, slice or mode
  without rerunning compute.

`snapshot()` returns `ordinarylight/view-state-v1`, the package ID, parameter
values and presentation settings. `restore()` validates before applying changes
and reruns compute; snapshots restore in a fresh runtime loaded with the same
package. A host must retain/reload the matching package to restore a snapshot
from an older structural configuration. The demo keeps replacement packages
while its server is running; it is not a persistent package registry.

Scientific metadata specifies output name, optional units, vector-element
sample meaning, axis samples, histogram edges, excluded nonfinite/out-of-range
counts, f32 precision and the random-number contract. The last bin includes its
right edge. Floating point agreement across backends is measured with tolerance;
bitwise agreement of transcendental computations is not a portable guarantee.

## Run the example

### Clean wheel installation

From this checkout, with Python 3.12 or newer and Git access to the upstream
repositories, run:

```sh
python scripts/check_portable_install.py --output /tmp/ordinary-portable-install
```

Choose an output directory that does not already exist. The script builds this
OrdinaryLight checkout and the four exact revisions in
[`upstream-requirements.txt`](../examples/portable_volume/upstream-requirements.txt)
into wheels, downloads their Python dependencies, and installs them into a fresh
virtual environment using only that wheelhouse. It runs `pip check`, verifies
the packaged browser assets, and prepares the DMC example outside the checkout
with isolated Python imports. No `PYTHONPATH`, sibling checkout, `.staging`
directory, or native `wgpu` installation is needed. Building requires network
access; installation from the resulting wheelhouse does not.

Run the copied example with the resulting environment:

```sh
/tmp/ordinary-portable-install/venv/bin/python -I \
  /tmp/ordinary-portable-install/serve.py \
  --output /tmp/ordinary-portable-install/viewer --port 8765
```

Then open `http://127.0.0.1:8765/` in a WebGPU-capable browser. See
[Linux host diagnostics](webgpu_host_diagnostics.md) if no adapter is available.
This installation check validates preparation and packaging; browser execution
and numerical parity are separate checks described below.

The pins identify the experimental API combination because current package
version floors alone do not distinguish it from earlier source revisions.
LatticeModel is not available on the configured package index; access to its
Git repository is required. This is a reproducible upstream source selection,
not a complete lock of transitive dependencies. Preserve the generated
wheelhouse when an identical installation is needed. Coordinated release
versions and updated dependency floors remain a separate release step.

### Development checkouts

In the local development workspace, the launcher supplies the upstream source
paths and uses this checkout's `.venv`:

```sh
bash examples/portable_volume/run-local.sh --output /tmp/portable-volume
```

It looks for sibling OrdinaryScience, OrdinaryLattice, ordinaryshade and
LatticeModel checkouts, falling back to `.staging/LatticeModel` for the model
package. This avoids requiring manual `PYTHONPATH` setup.

For a separately configured environment:

Install the source checkouts for OrdinaryShade, OrdinaryLattice, OrdinaryScience
and LatticeModel into the same environment as `ordinarylight[webgpu]`, or expose
their source roots through `PYTHONPATH`. From the OrdinaryLight checkout:

```sh
python examples/portable_volume/serve.py --port 8765 --output /tmp/portable-volume
```

Open `http://127.0.0.1:8765/` in a WebGPU browser. Model and presentation controls
come from the package. Save/restore uses JSON view-state files. Structural edits
call the local `/prepare` endpoint.

For a static export, with no server or GPU allocation:

```sh
python examples/portable_volume/serve.py --prepare-only --output /tmp/portable-volume
```

A static host can execute and update live/presentation values; structural edits
need a preparation callback. Browser WebGPU requires a secure context (localhost
is suitable). The demo binds only to the loopback interface.

## Verification

Run the Python and Node contract gates with the upstream source checkouts:

```sh
python -m pytest -q tests/test_portable_volume.py tests/test_portable_preparation.py
```

`test_portable_preparation.py` is an integration gate and skips when the upstream
preparation modules are unavailable. The GPU browser probe requires a dedicated
Chrome instance exposing its debugging port. For a reproducible Linux headless
software run, the tested Chrome configuration is:

```sh
google-chrome --headless=new --no-sandbox --disable-extensions \
  --enable-unsafe-webgpu --enable-unsafe-swiftshader \
  --use-angle=swiftshader --use-vulkan=swiftshader \
  --enable-features=Vulkan --disable-vulkan-surface \
  --remote-debugging-port=9226 --user-data-dir=/tmp/portable-chrome-test about:blank
node --experimental-websocket examples/portable_volume/browser_probe.mjs \
  http://127.0.0.1:9226 http://127.0.0.1:8765/ /tmp/portable-volume/browser.json
python examples/portable_volume/verify_results.py /tmp/portable-volume
```

The Node flag is needed for Node 20; Node 22+ provides WebSocket directly.
These Chrome flags are a test configuration. Successful testing with them does
not establish support in default browser configurations. On this Linux workspace,
the user reports that ordinary Chrome and the VS Code preview cannot acquire an
adapter, while Chrome launched with `--enable-unsafe-webgpu --use-vulkan` can.
VS Code's embedded host must be checked independently of external Chrome. The
page cannot enable a GPU backend that its host does not expose; `chrome://gpu`
provides Chrome's diagnostic information. See Chrome's
[WebGPU troubleshooting guidance](https://developer.chrome.com/docs/web-platform/webgpu/troubleshooting-tips).

This milestone validates the export/execution contract in an enabled host. It
does not yet provide a viewer that works across default hosts: there is no
automatic native/server rendering or CPU fallback when WebGPU is unavailable.
An incompatible headless compositor can lose the WebGPU device on canvas
presentation even when offscreen compute/render works.

On September 5, 2026, the combined contract/compute/upstream regression run passed
41 tests; 5 existing SPIR-V tests skipped because their compiler was unavailable.
Chrome SwiftShader passed nine browser checks with no GPU validation errors,
including exact numerical and pixel restoration in both existing and fresh
runtimes, live updates, same-device structural replacement and failed-replacement
recovery. Initial, seed-edited and structurally replaced RT packages matched
native NVIDIA RTX 4070 execution exactly for this sample, including finite masks,
histogram and exclusion counts. A native llvmpipe comparison had a maximum RT
absolute difference of `1.1920928955078125e-7`; histogram/exclusion counts were
exact. Native histograms also matched the NumPy reference and conserved sample
counts. This validates the small example, not all models, browsers or GPUs.

The verification program emits `verification.json` and `browser-volume.png`.
The wheel includes the browser JS/HTML and WGSL assets.

The checked sample's [verification report](../artifacts/portable-volume/verification.json)
and [rendered volume](../artifacts/portable-volume/browser-volume.png) are retained
with this milestone.

A subsequent [local host investigation](webgpu_host_diagnostics.md) reproduced
the default-adapter failure and verified an explicit NVIDIA/Vulkan configuration
without the unsafe-WebGPU override.
