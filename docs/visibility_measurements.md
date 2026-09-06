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
