# Visibility timing investigation

The earlier 5–6 ms box-probe numbers measured host submission through completion,
using the general transport integrator. They were not isolated visibility costs.

## Reproduction

```bash
PYTHONPATH=examples/transport_client/src:$PYTHONPATH python -m ordinarylight_transport_demo.visibility_probe --output /tmp/visibility-probe.json
```

Measured on an NVIDIA GeForce RTX 4070 Laptop GPU, Vulkan driver version integer
2434253120. Workload: 8,192 arbitrary boxes, 32,768 rays, eight boxes per spatial
group (1,024 groups), fixed seed 251. Two warmups and 15 measured rounds, randomized
across layouts and modes. All hit comparisons passed; 22,915 rays hit geometry.

## Initial measured medians (milliseconds)

| Layout | Visibility GPU | Visibility wall | Transport GPU | Transport wall | Transport pass GPU | Reduction interval GPU |
|---|---:|---:|---:|---:|---:|---:|
| Full linear scan | 23.235 | 24.747 | 23.522 | 27.050 | 22.925 | 0.598 |
| Spatial groups | 0.523 | 1.514 | 1.336 | 2.722 | 0.709 | 0.618 |
| Per-box primitives | 0.509 | 1.867 | 1.294 | 2.441 | 0.649 | 0.634 |

Visibility writes the full hit contract without material evaluation or reduction.
Transport uses first-hit emission, zero bounces, and persistent accumulation.
GPU timestamps bracket each pass and the overall operation; wall timing covers
host graph execution/command recording/submission through fence completion.
Query results and output readback occur after the wall timer stops.

Graph recompilation did not show a consistent wall-time improvement or regression:
spatial-group transport measured 2.675 ms with recompilation versus 2.722 ms with
reuse; per-box transport measured 2.889 versus 2.441 ms. Noise and independent
medians prohibit treating those differences as a precise compilation cost.
Each variant retains shader pipelines and buffers. Graph recompilation does not
include constructing a fresh transport operation.

## Interpretation and limits

- The grouped/per-box visibility workload takes about half a millisecond on the
  GPU here, substantially below the earlier submission-to-completion figure.
- A dedicated query avoids the general transport path and its reduction work.
  Reduction is unnecessary for camera nearest-hit queries. It remains necessary
  for the general many-samples-to-one-output transport contract.
- Roughly 0.5–0.8 ms of host execute/record/submit time remains for these modes.
  The rest of wall-minus-GPU time can include queue scheduling, driver work and
  fence wakeup; it is not all Python execution and is not all removable overhead.
- Per-box and spatially grouped GPU visibility remain close in this workload.
  The grouping tradeoff still needs update/memory measurements before choosing a
  production policy.

GPU intervals include instrumentation and may include dependency or scheduling
stalls. Medians of individual stages do not necessarily sum to the median total.
The frame count is small enough for device load, clocks and scheduling to matter;
these are local measurements, not portable performance guarantees. Timestamp
semantics follow the [Vulkan timestamp query specification](https://docs.vulkan.org/spec/latest/chapters/queries.html#queries-timestamps).

This is not a full rendered frame: no camera-ray generation, per-hit stored-color
lookup, tone mapping, presentation, scene updates or multi-bounce lighting is
included. It also writes the full 80-byte hit record. A fused application visibility
pass could reduce traffic; the benchmark does not establish its cost. Do not
extrapolate these 32,768 rays directly to a full-resolution frame.

The next priorities supported by these measurements are to use the dedicated
visibility path for camera queries, measure a fused color lookup at the intended
resolution, and investigate command recording/submission overhead separately.
Changing the intersection algorithm alone cannot explain or remove all of the
previous wall-clock latency.

## 720p fused color and memory-placement follow-up

A second probe traces an orthographic 1280×720 image (921,600 rays) against the
same 8,192-box distribution and fetches stored linear RGBA by application ID.
Compare full-hit output plus a separate lookup pass with fused intersection/lookup
writing the compact 32-byte color/identity/status record. Both layouts produced
exactly identical output, with 645,656 hits and no invalid lookup/intersection
statuses. Nine randomized rounds followed two warmups on the same RTX 4070 Laptop
GPU. Readback and PNG export occur after timing finishes.

| Memory policy | Layout | Separate lookup GPU ms | Fused GPU ms | Fused wall ms |
|---|---|---:|---:|---:|
| Host-visible/coherent | Spatial groups | 31.749 | 11.368 | 12.577 |
| Host-visible/coherent | Per-box primitives | 33.052 | 11.297 | 12.232 |
| Device-local | Spatial groups | 1.680 | 1.031 | 2.183 |
| Device-local | Per-box primitives | 0.810 | **0.259** | **1.423** |

The selected host memory type had flags 6 (host-visible/coherent, not device-local);
the device-local type had flags 1. Placement changed for box records, indices,
palette, ray input and query output buffers together. This experiment does not
isolate the contribution of each allocation. All data is uploaded before timing;
it does not include per-frame CPU streaming costs.

This changes the optimization priority. Memory placement and avoiding large
intermediate hit buffers have much larger effects here than median spatial
grouping. With device-local data, per-box primitives clearly outperform the
initial spatial partition in this fixture. Keep that path as the practical
baseline; measure update/acceleration costs before adopting groups downstream.

The fused path writes 29.5 MB of compact output per 720p frame instead of writing
and rereading the 73.7 MB full-hit intermediate and then writing compact output.
It still exports a GPU buffer, not an image. Camera-ray generation, stored-color
production, tone mapping, presentation and animated scene updates remain outside
the measurement. Device-local staging is explicitly synchronized; this test does
not claim those boundary operations are free. Host query/buffer defaults remain
unchanged, with device-local storage available as an opt-in public API.

## Direct HDR and tone-mapping composition

The `hdr_viewer` public client adds a GPU palette producer and direct HDR image
writes to fused visibility, followed by persistent ACES/sRGB tone mapping. The
720p / 8,192-box offscreen run on the same RTX 4070 Laptop GPU used twelve frames,
with the first two omitted from the medians:

| Interval | Median ms |
|---|---:|
| GPU palette producer | 0.0046 |
| GPU fused visibility and HDR write | 0.2918 |
| GPU tone mapping | 0.0467 |
| Total GPU interval | **0.3471** |
| Host submission through completion | **1.6008** |

No readback occurs in the frame loop. The final PNG and hit diagnostics are read
only after all measured frames complete; all hit statuses were valid. Independent
regressions compare tone-mapped pixels with the compact color output, including
misses, changing palettes and non-square images. This demonstrates the offscreen
output composition, not native presentation performance. The optional window path
is wired to bounded acquisition but has not been newly exercised. Geometry/rays
remain fixed and colors are produced procedurally on the GPU; multi-bounce lighting,
camera updates and animated geometry are still outside this measurement.

### Box-count scaling probe

A brief offscreen sweep on the same GPU held resolution at 1280×720 and used
one acceleration primitive per box. Each case ran eleven frames, omitting two
warmups; case order was randomized. Median total GPU intervals include the palette
producer, fused visibility/HDR output and tone mapping:

| Layout | 1,024 boxes | 8,192 boxes | 32,768 boxes |
|---|---:|---:|---:|
| Fixed box sizes | 0.317 ms | 0.345 ms | 0.394 ms |
| Approximately fixed projected coverage | 0.327 ms | 0.343 ms | 0.453 ms |
| Large overlapping boxes | 0.526 ms | 0.648 ms | 0.641 ms |

Fixed-size hit coverage increased from 17% to 85%. The fixed-coverage layout scales
XY extents by `sqrt(8192 / boxes)` and maintained approximately 70% hit coverage.
Thus 32 times as many boxes increased total GPU time by about 24% and 38%,
respectively, rather than linearly. Host submission through completion ranged
from 1.30 to 1.81 ms. All nine cases returned valid hit statuses.

Large overlapping boxes increase traversal cost here, but are not a guaranteed
worst case: opaque closest-hit traversal can prune occluded geometry. Their
8,192-to-32,768 plateau should not be interpreted as a general scaling guarantee.
This short probe excludes scene construction, uploads, acceleration builds/refits,
final image export, native presentation and GI. It does not establish scene-editing
cost or worst-case complexity.

Reproduce a case with `python -m ordinarylight_transport_demo.hdr_viewer
--boxes 32768 --frames 11 --layout fixed_coverage --output /tmp/scaling.png`.
Supported layouts are `fixed_size`, `fixed_coverage`, and `overlap`; each JSON
sidecar includes raw frame timings and hit count.

### Million-box capacity follow-up

The fixed-coverage case was extended to 131,072, 524,288 and 1,048,576 boxes
at 720p on the same GPU. Each case used a fresh process and eleven frames,
omitting two warmups. Hit coverage remained 69.6–69.8%; all cases returned zero
invalid hit statuses. A subsequent 32,768-box control is included below.

| Boxes | Median GPU output chain | Total instrumented setup | Geometry declarations | Scene construction | Acceleration setup, included in scene construction |
|---|---:|---:|---:|---:|---:|
| 32,768 control | 0.451 ms | 0.94 s | 0.46 s | 0.15 s | 5.5 ms |
| 131,072 | 6.208 ms | 2.80 s | 1.84 s | 0.56 s | 12.3 ms |
| 524,288 | 14.286 ms | 10.29 s | 7.57 s | 2.19 s | 20.2 ms |
| 1,048,576 | 21.794 ms | 20.95 s | 15.49 s | 4.48 s | 39.2 ms |

Setup covers box generation, record upload/targets, Python geometry declarations,
scene construction and query/output preparation, excluding runtime initialization
and teardown. Acceleration setup was measured by a temporary wall-clock wrapper
around `build_acceleration`: it includes bounds upload, allocation and completed
BLAS/TLAS builds, and is **not** an isolated GPU build timestamp. GPU output-chain
timings exclude all setup, final readback/export, GI and native presentation.

| Boxes | Process peak RSS | Sampled device memory increase over pre-run baseline |
|---|---:|---:|
| 32,768 control | 738 MiB | 155 MiB |
| 131,072 | 828 MiB | 171 MiB |
| 524,288 | 1,182 MiB | 253 MiB |
| 1,048,576 | 1,632 MiB | 358 MiB |

Host peaks use Linux `ru_maxrss`. Device memory is whole-device `nvidia-smi`
usage sampled approximately every 0.2 seconds, minus its pre-run baseline; it is
an approximate increase, not an exact application allocation peak. Other GPU
applications were active and were left running. Short-lived peaks can be missed.

Concurrent GPU activity limits performance conclusions. The million-box GPU
interval ranged from 16.06 to 28.14 ms across the nine measured frames, with a
26.57 ms median host submission-through-completion interval. The 32K control
nevertheless matched the earlier 0.453 ms result, so contention alone cannot be
assumed to explain the higher-count slowdown. The earlier modest growth does
not extrapolate to a million boxes. This establishes capacity for the tested
static scene, not an interactive GI performance guarantee. CPU declaration and
scene-packing work dominate initial setup; larger-scene traversal needs separate
profiling under an isolated GPU workload.

The client now records `setup_seconds` in its report. For example:

```bash
python -m ordinarylight_transport_demo.hdr_viewer --boxes 1048576 --frames 11 \
  --layout fixed_coverage --output /tmp/million-box.png
```

Raw timing reports, profiled reports and images from this run are in
`/tmp/million-box-probe/`; the temporary memory/build instrumentation driver is
`/tmp/million_box_probe.py`. These temporary artifacts are not repository fixtures.

### CPU bulk-declaration follow-up

With further GPU profiling deferred, a CPU-only comparison used the same generated
box records for the object and array-backed declaration paths. Timings cover
declarations plus scene record/bounds packing, excluding box generation, GPU
uploads/builds, runtime initialization and rendering:

| Boxes | Object declarations + packing | Bulk declarations + packing |
|---|---:|---:|
| 131,072 | 2.263 s | 0.021 s |
| 1,048,576 | 18.016 s | 0.252 s |

At one million boxes, bulk declarations took 0.059 s and packing took 0.193 s.
The complete packed custom records and acceleration bounds had identical SHA-256
digests between paths at both sizes. This single CPU trial demonstrates removal
of the per-object bottleneck, not a new end-to-end setup or GPU timing result.
The object path ran first; no repeated-trial confidence interval is claimed.

The viewer now uses `BoxBatch.geometries()`, backed by the general public
`CustomGeometryBatch`. Earlier GPU tables describe the preceding object-based
setup. GPU validation was deferred for this CPU trial; the subsequent validation
is recorded below. CPU probe results and its temporary
driver are `/tmp/bulk-geometry-cpu-results.json` and
`/tmp/bulk_geometry_cpu_probe.py`.

### GPU validation and custom-metadata placement

The subsequent GPU pass validated bulk construction, slot updates/removal,
capacity growth and resident HDR output. With the production fix below, 79
focused tests passed (native presentation deliberately deselected), followed by
five multi-bounce, dielectric, graph-emission and resource-backed transport tests.
An earlier 80-test run also exercised the display-enabled presentation regression;
this does not constitute a fresh four-panel Qt viewer smoke test.

The scientific DMC acceptance script passed baseline, non-basis parameter edits
and basis changes on WebGPU raster, Vulkan raster and Vulkan GI. All nine cases
matched their CPU histogram reference exactly, and all three target count volumes
agreed exactly. Fourteen additional scientific histogram, likelihood-data/display
and presentation-adapter tests passed. Outputs are in
`/tmp/scientific-gpu-validation/rt/`.

The first bulk GPU sweep still used host-visible custom scene metadata. At 720p
with approximately constant coverage, its total GPU interval was 0.455 ms at
32K boxes, 1.074 ms at 64K, 2.216 ms at 128K, 10.784 ms at 512K and 14.715 ms
at 1,048,576. At one million boxes, visibility accounted for 14.623 ms, compared
with 0.021 ms for palette generation and 0.068 ms for tone mapping. This localized
the expensive stage but did not alone distinguish traversal from metadata access.

A targeted probe changed only custom scene metadata allocation to device-local
memory. The million-box interval fell to 1.684 ms; the 128K interval fell to
0.547 ms. This identifies metadata memory placement as a major cause of the
observed slowdown, without claiming a hardware-counter measurement of bandwidth.
The production implementation now allocates the custom metadata table in
device-local memory during both construction and capacity growth. Its record
format, host/GPU update paths and application identities are unchanged.

The final production sweep measured:

| Boxes | Median GPU output chain | Visibility/HDR stage | Instrumented setup |
|---|---:|---:|---:|
| 32,768 | 0.522 ms | 0.436 ms | 0.309 s |
| 131,072 | 0.699 ms | 0.603 ms | 0.346 s |
| 524,288 | 1.180 ms | 1.098 ms | 0.439 s |
| 1,048,576 | 1.685 ms | 1.592 ms | 0.585 s |

The million-box final run used 0.057 s for declarations, 0.190 s for scene
construction (including 24.3 ms acceleration setup), and 0.201 s for query/output
setup. Its host submission-through-completion median was 2.887 ms. Process peak
RSS was 903 MiB; sampled whole-device memory rose about 421 MiB over baseline.
The additional device memory compared with the earlier 358 MiB increase is
consistent with relocating the 64 MiB custom record table. Memory sampling has
the same approximate whole-device limitations described above.

All four final PNGs are byte-identical to the corresponding host-metadata sweep,
and all hit statuses were valid. The bulk sphere example additionally verifies
identical complete query records after growing a scene containing moved and
disabled slots.

Each case used a fresh process, 31 frames and two omitted warmups (29 measured
frames). Resolution was 1280×720 with one acceleration primitive per box;
coverage stayed approximately 70%. Desktop graphics and a small resident Python
GPU process remained, so these are not exclusive-device benchmarks. Timing
variation at smaller counts should not be interpreted as a precise regression.
The GPU intervals exclude setup, final export/readback, native presentation and
GI. Instrumented setup excludes runtime initialization and teardown. The results
support static million-box visibility, not a universal frame-rate or GI guarantee.

Raw reports/images are under `/tmp/bulk-gpu-validation/` (host metadata),
`/tmp/bulk-device-metadata/` (targeted probe), and `/tmp/bulk-gpu-final/`
(production fix). Temporary drivers are `/tmp/bulk_gpu_probe.py`,
`/tmp/bulk_device_metadata_probe.py` and `/tmp/bulk_gpu_final_probe.py`.

### Ten-million-box allocation, build and growth

A subsequent capacity test used exactly 10,000,000 active boxes with the same
fixed-coverage distribution, 1280×720 output, 31 frames and two omitted warmups.
Hit coverage was 69.51% (640,628 rays). The RTX 4070 Laptop GPU had 8,188 MiB total
memory and approximately 440 MiB occupied before each process started.

| Measurement | 10M active boxes / 10M slots | 10M active boxes / 12M slots after growth |
|---|---:|---:|
| Median total GPU output interval | 3.490 ms | 3.426 ms |
| Median visibility/HDR interval | 2.782 ms | 2.730 ms |
| Median palette update | 0.644 ms | 0.633 ms |
| Median tone mapping | 0.058 ms | 0.057 ms |
| Median host submission through completion | 4.930 ms | 4.700 ms |
| Instrumented setup, including growth when applicable | 5.438 s | 7.788 s |
| Process peak RSS | 3,317 MiB | 4,843 MiB |
| Sampled whole-device memory peak | 3,192 MiB | 5,523 MiB |
| Peak increase over pre-run device usage | 2,752 MiB | 5,082 MiB |

Initial acceleration setup took 235 ms in the first run. In the growth run,
initial acceleration setup took 261 ms, followed by 319 ms for the replacement
acceleration allocation/build. Explicit `reserve_custom_geometry(12000000)` took
3.315 s in total, including host packing and allocation migration. These are
wall-clock measurements, not isolated GPU build timestamps.

The two million added slots remained inactive. Growth preserved all ten million
active boxes, their application records and palette. Both runs returned zero
invalid statuses and exactly the same hit count; the final PNGs are byte-identical.
Thus the second column validates additional capacity, not twelve million active
boxes or a new distribution. The small rendering-time difference is not a claimed
growth optimization.

Memory monitoring sampled `nvidia-smi` approximately every 0.2 seconds and also
sampled immediately after acceleration construction. In particular, the growth
sample captured the old and replacement allocations simultaneously resident at
5,523 MiB (5.39 GiB) whole-device usage. After retiring the old allocations, usage
was 3,529 MiB before query/output pipeline preparation. Exact transient peaks may
still be missed; these are observed peaks, not a strict worst-case memory bound.

This confirms that the tested ten-million-box workload fits with approximately
2.69 GiB additional peak device usage, and the tested growth path fits on this
8 GiB-class GPU as well. It does not establish capacity for arbitrarily large
textures, history buffers or material resources, nor performance for heavily
overlapping geometry or multi-bounce GI. The fixed-coverage distribution shrinks
projected box sizes as count increases. Frame timings exclude construction,
readback/export and presentation; setup excludes runtime initialization/teardown.

Raw reports and PNGs are in `/tmp/ten-million-boxes/` and
`/tmp/ten-million-growth/`. Temporary instrumentation drivers are
`/tmp/ten_million_box_probe.py` and `/tmp/ten_million_growth_probe.py`.
