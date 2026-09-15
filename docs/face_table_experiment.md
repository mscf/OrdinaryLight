# Smaller face-averaging hash tables

Status: rejected as an unconditional default after crowded-table GPU profiling.
Production retains four buckets per input pixel and the validated 32-stripe default.

Reprofiling production at animated native 4K gives approximately 1.53 ms linking,
0.55 ms reduction, 0.30 ms clearing, 0.17 ms finish and 0.16 ms resolve. Individual
pass instrumentation bypasses cached averaging commands and may affect overlap.

The `full-face-table` diagnostic allocates the next power of two at least twice
the input pixel count, instead of four times. All averaging shaders, sample
contributions and stripe counts remain unchanged. At 3840×2160 this halves the
bucket count from 33554432 to 16777216, saving 448 MiB across keys, heads,
stripe links and partial sums (28 bytes per bucket). This is an allocation-size
calculation, not a device-wide VRAM measurement. Other buffers are unchanged in
the tested scene; face-index storage can also depend on table capacity.

## Animated 4K measurements

RTX 5090 Laptop GPU, perspective, four bounces, one sample, 524288 paths,
GPU animation/layout, compact hits and paired spatial denoising. Twelve warmups,
forty measured frames with alternating comparison order at matching animation
times. Normal averaging command caching; headless GPU timings exclude CPU and
presentation.

| View | Production averaging | Smaller table | Production total | Smaller-table total |
| --- | ---: | ---: | ---: | ---: |
| Close-up | 2.694 ms | 2.493 ms | 18.957 ms | 18.700 ms |
| Overview | 3.056 ms | 2.780 ms | 13.142 ms | 12.867 ms |

Raw and denoised HDR and compact identities match exactly. Face output is within
the existing tolerance (overview matches exactly). The existing GPU averaging
suite passes all 17 tests, covering HDR formats, full/compact hits, mixed/coherent
faces, all misses, indexing modes, output scaling and resource reuse.

A separate 512×256 unique-face case creates 258048 entries in 262144 buckets,
98.4375% occupancy. It passes output, unique worklist entry and indirect dispatch
checks. The archived fixture derives from vxl8r's existing 520×260 test, changing
the extent and allowing a single dispatch row. Test wall time is not a GPU lookup
measurement. The two-bucket policy permits much higher occupancy than the current
at-most-half-full bound, so crowded-table performance remains to be measured
before considering production use. No general speedup or robust worst-case
lookup latency is established by these two scene measurements.

## Crowded-table GPU follow-up

`profile_face_table_occupancy.py` constructs 512×256 compact hits with a different
face for every valid pixel. Valid fractions sweep 25%, 50%, 75% and 100%; remaining
pixels are misses. Each comparison uses six warmups and 24 measured executions,
alternating table order on the same runtime and inputs. The input HDR is fixed;
this isolates averaging and is not a full animated renderer benchmark.
Cached command timings bracket the entire operation. A separate uncached run
adds per-pass bottom-of-pipe timestamps to identify the slow stage.

| Smaller-table occupancy | Production cached GPU time | Smaller cached GPU time |
| --- | ---: | ---: |
| 24.61% | 0.222 ms | 0.222 ms |
| 49.22% | 0.231 ms | 0.233 ms |
| 73.83% | 0.241 ms | 0.287 ms |
| 98.44% | 0.253 ms | 2.121 ms |

The 98.44% case is **8.4× slower**; p95 rises from 0.259 to 2.698 ms. Split timings
locate the regression in linking/hash insertion: 0.205 → 2.185 ms. Reduction and
resolve remain approximately unchanged. This is consistent with increased linear
probing at high occupancy; probe counts themselves were not instrumented.
Both runs validate final HDR against the expected per-pixel values, exact entry
counts, and uniqueness of every used worklist bucket at all four occupancies.

Keep production capacity unchanged. The modest savings in the earlier camera
views do not justify this performance cliff. A follow-up can investigate removing
face-anchor hash entries through direct face indexing, with a fallback for large
identity domains, rather than merely accepting a higher table load. That design
has not been implemented or benchmarked here.

```sh
PYTHONPATH=../vxl8r/src .venv/bin/python components/OrdinaryLight/scripts/profile_face_table_occupancy.py
PYTHONPATH=../vxl8r/src .venv/bin/python components/OrdinaryLight/scripts/profile_face_table_occupancy.py --split
```

## Earlier replay

```sh
VXL8R_TEST_VULKAN=1 PYTHONPATH=../vxl8r/src .venv/bin/python components/OrdinaryLight/scripts/validate_face_averaging_experiment.py --variant table
PYTHONPATH=../vxl8r/src .venv/bin/python components/OrdinaryLight/scripts/profile_primary_prefixes.py --stage full-face-table
PYTHONPATH=../vxl8r/src .venv/bin/python components/OrdinaryLight/scripts/profile_primary_prefixes.py --stage full-face-table --overview
VXL8R_TEST_VULKAN=1 PYTHONPATH=../vxl8r/src .venv/bin/python components/OrdinaryLight/scripts/validate_face_averaging_experiment.py --variant table --tests components/OrdinaryLight/artifacts/face-table/test_face_crowded.py::test_sparse_average_compact_queue_supports_unique_faces_and_2d_dispatch
```

Logs and crowded fixture: `artifacts/face-table/`. Python compilation checks pass.
