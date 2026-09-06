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
