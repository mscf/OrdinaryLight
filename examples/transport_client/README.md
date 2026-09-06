# External transport client

This separately installable package depends only on OrdinaryLight's public APIs.
It renders three rows of application-indexed samples: an exact two-diffuse-bounce
cavity result, a refracting/absorbing SDF sphere, and a triangle glass box.
The sphere reads a declared application storage buffer. Two surface samples per
diffuse output are reduced explicitly using SampleReduction, and their GPU
input allocation is updated between frames without recreating integrators.
The application owns sample identities and display layout. There are no voxel
reconstruction or artistic averaging algorithms here.

Install the current OrdinaryLight source (the new APIs are unreleased), then this
package into the same environment:

```bash
python -m pip install -e /path/to/wave-render
python -m pip install -e /path/to/wave-render/examples/transport_client
ordinarylight-transport-demo --output /tmp/transport.png
# Optional GLFW native presentation:
ordinarylight-transport-demo --present --frames 16 --output /tmp/transport.png
```

Custom intersection/integrator compilation needs `glslangValidator` or `glslc`.
Native presentation requires GLFW, available with this package's `present` extra.
The final PNG/JSON/NPZ export intentionally reads the GPU results; live native
presentation uses the resident HDR image. The diffuse row deliberately stops
after two bounces, so its truncation counter is expected and its value is checked
against the finite cavity series. Nonzero invalid-path status raises an error.

Version 0.3.0 composes transport, reduction, HDR resolve, tone mapping and optional
presentation into an application graph. It keeps a persistent tone-map target
and uses a two-frame ring. CPU sample updates in the cavity demo intentionally
remain synchronized uploads.

The second entry point exercises GPU animation in a fixed sparse grid:

```bash
ordinarylight-animated-grid --frames 8 --output /tmp/grid.png
ordinarylight-animated-grid --present --frames 120 --output /tmp/grid.png
# From the source package:
python -m ordinarylight_transport_demo.animated_grid --frames 8
```

OrdinaryShade generates occupancy and exports access reflection. OrdinaryLight
orders the producer, custom-geometry updates, transport, reset/resolve, tone
mapping and presentation. Chunk capacity grows once; slots are activated and
removed without moving grid cells. This is a small execution fixture, not a
voxel authoring or reconstruction engine. Final RGB values are checked against
an exact emission/environment reference, and only final verification/export
reads back GPU results. Build from the current unreleased OrdinaryLight source.

## Material graph workload (0.4.0)

```bash
ordinarylight-material-graph --samples 64 --output /tmp/material-graph.png
```

This additional public-API client renders graph-defined rough metal and emission,
rough glass, analytic lighting, environment and emissive-sphere MIS,
GPU-discovered sphere slots and
acceleration refits, unequal coverage weights, persistent HDR and tone mapping.
The final PNG and JSON report are explicit diagnostic readbacks. See
[material graph contracts](../../docs/material_graph_milestone.md).

## GPU-authored reduction workload

    python -m ordinarylight_transport_demo.gpu_reduction --frames 6 --output /tmp/gpu-reduction.json

A GPU producer generates sample records, changes active counts, and builds ordered
weighted reduction groups. The same compiled graph validates the maps, dispatches
transport indirectly, and accumulates into three persistent IDs. No sample/count/
map readbacks or CPU grouping occur; only final means and diagnostics are read for
verification. Allocation capacities remain explicit and fixed.

The GPU reduction client starts with two sample slots and grows to four and six
using `grow_capacity`. It rebinds producers after each migration while preserving
weighted output history. The JSON report includes the capacity used each frame.
This illustrates lifecycle correctness; production clients can cache producer
kernels and schedules between growth events.

## Grouped procedural geometry reference

The `bulk_updates` client exercises general bulk incremental geometry updates
using analytic SDF spheres: it moves all spheres, changes application identities,
then disables alternating slots and grows scene capacity. Queries follow each edit in the graph and check
distances, identities and validity. It uses only public APIs and no window.

```bash
# Runs offscreen GPU validation.
python -m ordinarylight_transport_demo.bulk_updates --count 128
```

The staging clients own immutable upload snapshots and are closed after their
operations complete. CPU command/packing tests and the offscreen GPU regression pass, including
identical query results across capacity growth.

```bash
python -m ordinarylight_transport_demo.chunk_probe --samples 64 --output /tmp/chunk-probe.json
```

This uses public `BoxBatch` geometry to compare eight independent custom primitives
with two groups of four boxes. It checks hit distances, normals, material and
application identities, medium boundaries, and multi-bounce radiance across three
activation/emission-resource states. Materials include diffuse, graph-driven
metal, ideal glass and resource-backed graph emission. The JSON contains errors
and elapsed submission/wait times; those small, unordered timings are diagnostic,
not a throughput benchmark. Emission is reached by BSDF paths; this reference has
no custom area sampler and does not evaluate emissive NEE performance.

The fixture follows the resident-resource pattern used by vxl8r, but imposes no
grid layout, voxel identity scheme, reconstruction, or face-averaging policy.
It scans bounded record groups linearly. It is a correctness reference for a
future spatial traversal implementation, not a production sparse-grid renderer.

The `chunk_probe` additionally compares indexed spatial groups produced by
`BoxBatch.partition`, including graph emission and activation edits.

```bash
python -m ordinarylight_transport_demo.partition_probe --boxes 8192 --rays 32768 --repeats 11
```

This larger opaque-emission workload reports raw warm measurement samples for
linear, spatially partitioned and per-box geometry, with a visibility parity
check. It uses arbitrary box positions/sizes. It does not measure scene updates,
material evaluation complexity, dielectric transport, or an entire voxel renderer.
Choose group sizes against the intended workload; fewer primitives alone do not
guarantee faster tracing. See `BoxPartition.refit` for stable-slot motion updates.

## GPU visibility timing breakdown

```bash
python -m ordinarylight_transport_demo.visibility_probe --output /tmp/visibility-probe.json
```

Uses the same seeded 8,192-box / 32,768-ray workload as `partition_probe`, with
persistent visibility and zero-bounce transport pipelines. Each layout is tested
with a reused graph; a transport variant recompiles only the graph each iteration
(not shaders or pipelines). The report includes the GPU/driver, raw samples and
medians from 15 randomized rounds after two warmups.

GPU timestamps bracket intersection, transport and reduction passes, and the full
GPU interval. Timestamp support and valid-bit wrapping are handled. Host timings
separate execute/record/submit from completion wait. Query-result and hit readback,
scene construction, partition building and uploads are excluded. Timings include
profiling instrumentation, and GPU intervals may include scheduling/dependency
stalls; they are not instruction-level shader profiles. The compiled schedule
still records/submits command buffers on each execution.

The probe checks full hit identity/distance agreement and matches transport's
emissive output against the visibility mask. It deliberately omits presentation,
color lookup, multiple bounces and animated scene updates. See
[the measurement notes](../../docs/visibility_measurements.md) for the initial run.

## Image-resolution fused color comparison

```bash
python -m ordinarylight_transport_demo.color_probe --output /tmp/color-probe-720p.json
```

At 1280×720, compare a persistent full-hit query followed by a GPU color lookup
with a fused `VulkanRayQuery(colors=palette)` dispatch. Both retain color and basic
hit diagnostics in the same compact output layout. The probe uses 8,192 boxes,
nine randomized timing rounds after two warmups, exact output comparisons, and
GPU timestamps plus host submission/wait measurements. It reports buffer memory
flags and exports an sRGB preview alongside the JSON after timing finishes.

Ray generation, uploads, readback, PNG encoding, tone mapping and presentation
are outside the timed interval. Colors are precomputed application data; there
is no lighting evaluation. The default runtime buffers are host-visible/coherent;
the report records whether the selected memory type is also device-local. These
allocation choices must be considered when interpreting image-resolution results.

The color probe now defaults to explicit device-local working buffers. Compare
both placement policies with:

```bash
python -m ordinarylight_transport_demo.color_probe --memory host --output /tmp/color-host.json
python -m ordinarylight_transport_demo.color_probe --memory device --output /tmp/color-device.json
```

This changes placement for box records, indices, palettes, rays and outputs
together. Upload/readback staging is outside the timed interval. It therefore
measures a resident workload, not streaming data from the CPU every frame.

## Resident HDR output loop

```bash
python -m ordinarylight_transport_demo.hdr_viewer --output /tmp/hdr-viewer.png
# Optional native-window path (not part of the offscreen validation):
python -m ordinarylight_transport_demo.hdr_viewer --present --frames 600
```

The default offscreen client runs twelve frames at 1280×720. A GPU producer changes
the stored-color palette; fused visibility writes HDR directly, and the persistent
tone-map target produces RGBA8. The graph performs no CPU image/color readback
inside the frame loop. Only after the loop does it export a PNG and JSON report.
The first two frames are omitted from timestamp medians. The `--present` path
uses the public single-use presentation operation, bounded acquisition and
cancellation handling; its native-window behavior has not been newly validated.

Render resolution and ray inputs stay fixed; framebuffer resizing affects the
presentation blit. Box geometry is static, and palette animation demonstrates
GPU resource updates rather than recomputing illumination. The compact hit buffer
remains available for diagnostics in addition to direct HDR output. Install the
client's `present` extra for the optional window path.

For box-count scaling probes, use `--boxes 32768 --frames 11` with
`--layout fixed_size` (the default), `fixed_coverage` (approximately constant
projected coverage), or `overlap` (large overlapping boxes). See
[visibility measurements](../../docs/visibility_measurements.md#box-count-scaling-probe)
for measured results and limitations.
JSON reports also include `setup_seconds`, separating box generation, uploads and
targets, geometry declarations, scene construction, and query/output preparation.
Scene construction includes acceleration setup; these are host wall-clock times.
The viewer now passes `BoxBatch.geometries()` directly to the scene, using the
general array-backed `CustomGeometryBatch` path. This preserves one acceleration
primitive per box while avoiding a Python object per box. CPU packing equivalence
and offscreen GPU output are validated. Custom scene metadata now stays in
device-local memory during creation and growth; see the visibility measurements
for the million-box follow-up.
