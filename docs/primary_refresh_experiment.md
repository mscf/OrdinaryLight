# Primary workload refresh after paired spatial filtering

Status: diagnostics only; no production renderer or vxl8r changes promoted.
The 2048-slot acceleration grouping candidate is rejected for both performance
and HDR parity.

## Configuration and boundaries

RTX 5090 Laptop GPU, animated perspective close-up, native 3840×2160,
512×512 voxel domain, GPU animation/layout, tight bounds, four bounces,
one sample, 524288 path capacity, compact (20-byte) primary-hit exports,
and the current production paired spatial filter. Every comparison alternates
baseline/candidate order for 40 measured frames after 12 warmups.

Primary timestamps exclude the separately measured queue/setup interval
(about 0.02 ms). Prefix shaders intentionally stop before completing lighting;
their HDR and downstream GPU totals are not valid rendering or speedup results.
Removing workload changes compiler/register/cache behavior. Differences are
not additive in-shader timings and are not exact isolated traversal costs.

## Results

| Comparison | Baseline primary | Candidate primary |
| --- | ---: | ---: |
| Full rendering → trace/init/export prefix | 4.752 ms | 3.688 ms |
| Trace prefix → synthetic-hit prefix, face averaging disabled | 3.683 ms | 3.682 ms |
| Trace prefix → compact exports suppressed, face averaging disabled | 4.227 ms | 3.827 ms |

The first prefix retains camera sampling, secondary initialization, intersection,
compact hit and custom-history exports. Its compact identity/validity matches the
full renderer exactly in the final frame. Compact output does not include ray
vectors, so these runs do not assert sampled camera-ray parity.

The synthetic variant retains camera-dependent work but uses a fixed-distance
custom hit. Its identity/validity matches the intended synthetic values. This
comparison offers no measurable benefit from removing traversal in this prefix;
it does not establish that traversal is free in the complete shader. The no-export
variant saves 0.400 ms in its matched comparison, but deliberately removes a
required public output. It is not a production candidate. Between-run baseline
variation means the three rows must not be combined into a cost budget.

An initial synthetic run with face averaging enabled caused artificial contention:
every synthetic hit had the same identity, pushing face output to 29.77 ms.
The profiler now disables face averaging for both sides of prefix experiments;
complete-render experiments retain it. The corrected synthetic comparison removes
that confounder. The initial log is retained but is not used for conclusions.

## Smaller acceleration groups

A diagnostic subclass changed each instance's BLAS grouping from 4096 to 2048
slots while retaining the original class-level spatial brick ordering, resident
slot identities, geometry callback, and shading. Both use the existing public
acceleration resources. This is a vxl8r-side diagnostic, not an upstream patch.

Primary measured 4.848 → 4.923 ms, first secondary 4.052 → 4.081 ms, and total
GPU 19.019 → 19.363 ms (about 1.8% slower). Final raw HDR parity failed in six
color components across two pixels, maximum difference 0.22265625. The harness
stopped at that failure; denoised/hit parity was not subsequently asserted.
Changing traversal order at coincident boundaries is a possible cause, but it
has not been established. No mismatch is dismissed as harmless rounding, and
no production grouping changes are retained.

## Direction

These measurements weaken the case for additional group-size tuning or claiming
that primary time is predominantly traversal. Initialization and output traffic
remain stronger candidates for investigation. At 4K, a complete 128-byte
secondary-record initialization writes 1012.5 MiB, compact primary-hit output
158.2 MiB, and the 32-byte custom-history output 253.1 MiB. These are layout-based
byte counts, not measured hardware transactions or proven bandwidth bottlenecks.
Any optimization must preserve their consumers and sampled-ray/history contracts;
removing required outputs is not an acceptable solution.

## Replay

From Ordinary, run sequentially:

```sh
PYTHONPATH=../vxl8r/src .venv/bin/python components/OrdinaryLight/scripts/profile_primary_prefixes.py --stage trace --primary-hit-format identity
PYTHONPATH=../vxl8r/src .venv/bin/python components/OrdinaryLight/scripts/profile_primary_prefixes.py --stage synthetic --baseline trace --primary-hit-format identity
PYTHONPATH=../vxl8r/src .venv/bin/python components/OrdinaryLight/scripts/profile_primary_prefixes.py --stage trace-no-export --baseline trace --primary-hit-format identity
PYTHONPATH=../vxl8r/src .venv/bin/python components/OrdinaryLight/scripts/profile_primary_prefixes.py --stage full-as-2048
```

The final command is expected to fail HDR parity on the recorded workload.
Logs are in `artifacts/primary-refresh/`. The first archived trace capture predates
the face-averaging suppression; current replay disables it for prefix comparisons.
All captures exclude swapchain/compositor presentation.
