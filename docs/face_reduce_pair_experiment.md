# Two-node face reduction loop

Status: diagnostic only. Production remains unchanged.

Current production profiling with four-pixel grouping gives approximately
1.426 ms linking, 0.497 ms reduction, 0.164 ms clearing, 0.078 ms finish and
0.163 ms resolve. These per-pass timings disable command caching and insert
bottom-of-pipe markers; full-operation comparisons use normal cached commands.

The typed OrdinaryShade experiment processes up to two linked-list nodes per
loop iteration. Each next-pointer load precedes that node's HDR load. It retains
the original sequential addition order, sentinel checks, empty-list behavior,
and odd-tail handling. No buffers, dispatch dimensions, layouts or capabilities
change. Source-level scheduling does not itself prove more outstanding GPU reads.

The candidate passes **19 GPU averaging tests**, covering both identity formats,
HDR, mixed/coherent faces, misses, scaling, cached reuse and allocation boundaries.
Final raw/denoised HDR and compact hit outputs match exactly in the animated 4K
comparison; face output differs only within the existing tolerance.

RTX 5090 Laptop GPU, native 3840×2160 perspective close-up, four bounces,
one sample, 524288 paths, GPU animation/layout, compact hits and paired spatial
denoising. Twelve warmups, forty measured frames, alternating renderer order at
matching animation times. These are headless GPU timings, excluding CPU and
presentation.

| Run | Production averaging | Candidate averaging | Production total | Candidate total |
| --- | ---: | ---: | ---: | ---: |
| Initial | 2.345 ms | 2.254 ms | 18.507 ms | 18.479 ms |
| Repeat | 2.323 ms | 2.259 ms | 18.585 ms | 18.514 ms |

Averaging saves 0.064–0.091 ms, but total GPU savings are only 0.028–0.071 ms
(0.15–0.38%). Keep the maintained loop for now; these results do not justify
promoting another shader variant without broader workload evidence. A larger
opportunity may be tuning pixel grouping, which previously produced a materially
larger improvement while preserving all contributions. Larger groups are not
tested by this experiment.

## Replay

```sh
VXL8R_TEST_VULKAN=1 PYTHONPATH=../vxl8r/src .venv/bin/python components/OrdinaryLight/scripts/validate_face_averaging_experiment.py --variant reduce-pair
PYTHONPATH=../vxl8r/src .venv/bin/python components/OrdinaryLight/scripts/profile_primary_prefixes.py --stage full-face-reduce-pair
```

Logs: `artifacts/face-reduce-pair/`. Python compilation and whitespace checks pass.
