# Primary-stage cumulative diagnostics

`scripts/profile_primary_prefixes.py` generates temporary typed OrdinaryShade
variants of the fused primary shader. It does not change packaged shaders or
production rendering. It requires the sibling vxl8r checkout on `PYTHONPATH`.

From the Ordinary repository root:

```sh
PYTHONPATH=../vxl8r/src .venv/bin/python components/OrdinaryLight/scripts/profile_primary_prefixes.py --stage trace
PYTHONPATH=../vxl8r/src .venv/bin/python components/OrdinaryLight/scripts/profile_primary_prefixes.py --stage material
PYTHONPATH=../vxl8r/src .venv/bin/python components/OrdinaryLight/scripts/profile_primary_prefixes.py --stage lighting
```

Run these sequentially, without another GPU workload. Each compares the selected
prefix with a full renderer, alternating execution order at identical animation
times. There are 12 warmup frames and 40 measured samples per renderer. Native
stage timestamps report previous frame-slot use; warmup exceeds that delay.
Readback and waits belong to this diagnostic harness, not production rendering.

## Animated perspective results

RTX 5090 Laptop GPU, 3840×2160 GI and output, 512×512 voxel scene, perspective
close-up (zoom 3.8), GPU animation/layout, tight dynamic bounds, 128 chunk pool,
four bounces, one sample per pixel, denoising and 524,288 path capacity.

| Cumulative workload | Primary median | Paired full median | Prefix p95 | Full p95 |
| --- | ---: | ---: | ---: | ---: |
| Camera, initialization, intersection, hit/history exports | 6.056 ms | 6.641 ms | 7.007 ms | 7.948 ms |
| Above plus material/surface setup and guides | 5.983 ms | 6.646 ms | 6.938 ms | 7.959 ms |
| Above plus emission and direct/environment lighting | 6.094 ms | 6.928 ms | 7.064 ms | 8.082 ms |

These original measurements include queue reset, bulk secondary-buffer clearing
and synchronization before the shader dispatch. The extended profiler now
reports that interval separately as `setup_median_ms`; `primary_median_ms` in
new captures measures the subsequent primary graph/dispatch interval. Do not
compare the old and new primary columns without accounting for this boundary.

These are cumulative shader workloads, **not additive in-shader timings**.
Removing later work changes compiler optimization, register pressure, memory
traffic and subsequent rendering. The trace prefix also writes invalid guides
for safe downstream execution; the material prefix sinks evaluated material
values into its diagnostic radiance to retain that work. The lighting prefix
stops before the opaque BSDF continuation sampling; earlier optical/transmission
branches remain. Results describe this scene, not arbitrary materials or lights.

The early workload already costs approximately 6 ms. Material evaluation does
not show a measurable positive marginal cost in these captures. Full primary
costs another approximately 0.6–0.8 ms relative to its paired prefix. This does
not prove that intersection alone costs 6 ms, or that lighting is free. The
early workload includes secondary-record initialization, camera sampling,
custom traversal, primary-hit exports and custom history exports. The next
useful investigation is separating that buffer traffic from traversal, followed
by continuation sampling/record storage and queue emission. Further material
arithmetic micro-optimization has weak evidence here.

## Validation and limitations

All three generated variants compiled and ran on the GPU. Each run checks exact
equality against full rendering for the final sampled primary identity, ray
origin/direction, position/distance and geometric normal. Shading-normal parity
is deliberately excluded because the trace prefix stops before material setup.
This is one final-frame hit check per run, not a test of all scenes or frames.
Prefix HDR is intentionally invalid and must not be consumed as a lighting
result; downstream timings and images are not parity or performance evidence.
The harness temporarily replaces the source-expansion function in its own
process and must not run concurrently with other shader compilation threads.

These headless, serialized diagnostic measurements exclude presentation and
are not frame-rate results. Raw logs, including parity assertions, are stored
under `artifacts/primary-prefixes/`. No production optimization is enabled by
this experiment.

## Follow-up: setup, traversal and export traffic

The profiler now inserts `primary_setup` immediately before recording the native
primary graph. This separates preceding queue resets, transfer clears and
synchronization from the primary graph/dispatch interval. Instrumentation uses
the existing timestamp callback and is confined to the diagnostic process.

`--stage synthetic` replaces traversal with a constant-distance custom hit while
preserving camera-dependent hit/history exports. It asserts the synthetic
identity/distance and exact camera-ray equality. At 4K its primary interval was
5.417 ms, versus 5.874 ms for full rendering; setup was 0.699/0.708 ms. Therefore
the earlier approximately 6 ms prefix result is not evidence that intersection
itself costs 6 ms. Initialization and output traffic warrant attention. This is
an inference from different shader workloads, not a bandwidth-counter result.

Two complete-render candidates were tested, also authored in OrdinaryShade:

| Candidate | Baseline primary | Candidate primary | Baseline total GPU | Candidate total GPU |
| --- | ---: | ---: | ---: | ---: |
| Remove shader secondary-record clear; retain bulk transfer clear | 5.903 ms | 7.721 ms | 22.743 ms | 24.682 ms |
| Store one complete primary-hit record per branch | 5.926 ms | 5.876 ms | 23.092 ms | 23.005 ms |

Neither is enabled in production. Removing redundant stores is not necessarily
faster: the clear-removal regression is measured, but its cache/compiler cause
has not been established. The single-record rewrite saved only 0.087 ms total
in one capture, insufficient to establish a reliable benefit.
Both candidates preserved raw and denoised HDR exactly; face-averaged maximum
differences were 1.19e-7 and 1.79e-7 respectively. Final-frame sampled primary-hit
checks passed. These scene-specific checks do not replace broader conformance
testing if either candidate is revisited for production.

Replay with `--stage full-no-clear` or `--stage full-single-export`. The former
is only safe in this harness's denoiser-enabled configuration, whose host path
clears all secondary records before primary execution. It must not be copied
unconditionally into production shader code.

`--stage trace-no-export --baseline trace` compares matching traversal prefixes
with and without the 96-byte primary-hit export. The variant retains a
distance/normal sink and custom-history exports. Primary-hit contents and
downstream face averaging are invalid, so no hit parity or end-to-end speedup
is claimed. Its downstream total GPU time must not be interpreted. This
experiment is solely a lead for investigating output layout/traffic, not
permission to remove application-visible hit information.

In the matching-prefix capture, primary time fell from 5.182 to 3.837 ms
(1.345 ms); setup measured 0.661 versus 0.696 ms. Each 4K hit-output buffer is
3840 × 2160 × 96 = 796,262,400 bytes (759.375 MiB) per sample. Custom history
exports add another 253.125 MiB per frame. A public compact or better-coalesced
hit-output representation is a stronger next candidate than small traversal
arithmetic changes, provided it preserves sampled-ray identity and all required
attributes. This experiment does not establish how much of that 1.345 ms such
a representation can recover. Disabled exports also change downstream work
and cache behavior; a production candidate requires full-output validation and
interleaved frame timing.
