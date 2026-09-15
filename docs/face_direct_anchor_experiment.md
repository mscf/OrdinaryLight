# Direct face anchors with bounded stripe hashing

Status: promoted in vxl8r, retaining 32 stripes and the large-domain hash fallback.
No upstream production renderer changes were required. Restart the viewer to use
the new allocation policy; no flag is needed.

The temporary OrdinaryShade variant stores anchors at `stripe_capacity + face_id`
in extended key/head/partial arrays, outside the hash domain. The winning atomic
compare/exchange publishes each direct anchor once to the existing worklist.
Nonzero stripe keys retain hash insertion and their per-face chain. Reduction,
finish, resolve, contributions, and record bindings remain unchanged.

For P pixels, the stripe table uses the next power of two at least 2P. At most
one stripe key is inserted per pixel, so its load stays at most one half. Direct
storage is enabled only when the full face domain fits within that stripe-table
capacity. Otherwise allocation and anchor hashing retain the production path.
Buffer length selects the shader branch, without a new public ABI or capability.
This avoids the rejected smaller-table design's near-full-table behavior.

Memory savings depend on the face domain: direct arrays add 24 bytes per possible
face (key/head/partial), while reducing hash allocations. This is not a claim
of the earlier prototype's fixed 448 MiB saving; that prototype had no direct
anchor storage.

## GPU validation

The existing averaging suite passes **17 tests**, including full/compact hits,
HDR formats, mixed/coherent faces, misses, scaling, large-domain fallback and
cached reuse. The standalone occupancy benchmark additionally packs six unique
face IDs per slot to exercise direct indexing with a distinct face at every
pixel. Every occupancy checks output values, entry counts and bucket uniqueness.

At 512×256 with all pixels valid:

| Identity domain | Production GPU | Candidate GPU | Candidate stripe occupancy |
| --- | ---: | ---: | ---: |
| Packed faces, direct indexing | 0.244 ms | 0.232 ms | 48.44% |
| Large domain, fallback | 0.255 ms | 0.256 ms | 49.22% |

Packed-face p95 is 0.250 → 0.237 ms. This workload has the same number of entries
as the earlier crowded test, but different key values and a separate anchor
domain; it is not an identical hash-key distribution. Six warmups, 24 measured
cached operations per occupancy, alternating order on the same runtime.

## Animated native 4K

RTX 5090 Laptop GPU, perspective, four bounces, one sample, 524288 paths,
GPU animation/layout, compact hits and paired spatial denoising. Twelve warmups,
40 measured frames at matching animation times, alternating renderer order.
Normal cached averaging; headless GPU timings exclude CPU/presentation.

| View | Production averaging | Candidate averaging | Production total | Candidate total |
| --- | ---: | ---: | ---: | ---: |
| Close-up | 2.686 ms | 2.557 ms | 18.782 ms | 18.648 ms |
| Overview | 3.068 ms | 2.666 ms | 13.597 ms | 13.044 ms |

Final raw and denoised HDR and compact hits match exactly. Face output remains
within the established comparison tolerance. These are single comparisons for
each view, not repeated evidence for a fixed total speedup. Other stages vary,
especially in the overview run. Repeat timings and cover both sides of the
allocation threshold before promoting the new layout.

## Promotion checks

A repeated animated 4K close-up comparison gives averaging **2.694 → 2.552 ms**,
total GPU **18.887 → 18.599 ms**. Together with the first run, averaging saves
0.130–0.142 ms; total savings vary from 0.134–0.288 ms. Final raw and denoised HDR
and compact hits match exactly. Total timing variation includes unmodified stages.

At 512×256, adjacent domain sizes straddle the direct-allocation boundary:
43690 slots use 262140 direct anchors with 262144 stripe buckets; 43691 slots
use the legacy 524288-bucket layout. Both pass output, entry-count and uniqueness
checks at all occupancy levels. Full-input GPU medians are 0.234 and 0.246 ms
respectively. These pre-promotion runs used low-numbered identities.

The production averaging suite was expanded with direct/fallback boundary cases
that use the highest valid slots at 520×260, and now passes **19 tests**. This
includes cached reuse, both hit formats, HDR, misses, and unchanged dense averaging
controls. A post-integration animated per-frame comparison is also retained in
the artifacts. The diagnostic installer reconstructs the previous implementation
for baseline comparisons; the candidate uses the maintained production module.

## Diagnostic replay

```sh
VXL8R_TEST_VULKAN=1 PYTHONPATH=../vxl8r/src .venv/bin/python components/OrdinaryLight/scripts/validate_face_averaging_experiment.py --variant direct
PYTHONPATH=../vxl8r/src .venv/bin/python components/OrdinaryLight/scripts/profile_face_table_occupancy.py --variant direct --packed-faces
PYTHONPATH=../vxl8r/src .venv/bin/python components/OrdinaryLight/scripts/profile_face_table_occupancy.py --variant direct
PYTHONPATH=../vxl8r/src .venv/bin/python components/OrdinaryLight/scripts/profile_primary_prefixes.py --stage full-face-direct
PYTHONPATH=../vxl8r/src .venv/bin/python components/OrdinaryLight/scripts/profile_primary_prefixes.py --stage full-face-direct --overview
```

Logs: `artifacts/face-direct/`. Python compilation and whitespace checks pass.
