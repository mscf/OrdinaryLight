# Four-pixel stripe grouping

Status: promoted in vxl8r. Restart the viewer to use it; no new flag or upstream
production renderer change is required.

Stripe selection changes from `pixel_index % stripes` to
`(pixel_index / 4) % stripes`, using integer division. Four adjacent linear input
pixels share a stripe number. Face identity still separates their lists, and
every valid pixel contributes exactly once. This changes reduction order and
the number of occupied stripe keys, not transport, denoising, sample count or
the arithmetic mean being estimated. Table allocation and the at-most-half-full
hash bound remain unchanged.

The intent is better list locality; neighboring insertion contention can work
against it. The measurements below establish an overall improvement in these
views, not a separate measurement of cache misses or atomic contention.

## Animated native 4K

RTX 5090 Laptop GPU, 3840×2160, four bounces, one sample, 524288 paths,
GPU animation/layout, compact hits, paired spatial denoising, production direct
anchors and 32 stripes. Twelve warmups and 40 measured frames, alternating
renderer order at matching animation times. Cached averaging operations,
headless GPU timings; no CPU/presentation FPS claim.

| Perspective view | Previous averaging | Grouped averaging | Previous total | Grouped total |
| --- | ---: | ---: | ---: | ---: |
| Close-up | 2.648 ms | 2.213 ms | 18.679 ms | 18.299 ms |
| Close-up repeat | 2.646 ms | 2.224 ms | 18.693 ms | 18.284 ms |
| Overview | 2.655 ms | 1.714 ms | 12.684 ms | 11.757 ms |

The close-up total improves by 2.0–2.2%; the overview by 7.3%. Raw and denoised
HDR and compact hits match exactly. Final face-output error stays within 1.20e-7
in the close-up and is zero in the overview.

The post-integration orthographic comparison is essentially neutral for averaging:
1.999 → 1.995 ms. Total GPU time is 18.636 → 18.303 ms, but that difference should
not be attributed to grouping because its own operation barely changed. Final
HDR and hit comparisons pass. The benefit is view-dependent.

## Validation

The prototype and integrated production implementation each pass all **19 GPU
averaging tests**. These cover HDR, mixed/coherent faces, full/compact hits,
misses, scaling, cached reuse and both sides of the direct-index cutoff.
The unique-face worklist count assertion now counts pixels assigned stripe zero
under four-pixel grouping; expected image averages are unchanged. The diagnostic
runner originally changed that assertion in a temporary fixture, and production
tests now maintain the new cardinality expectation.

The diagnostic installer reconstructs the old stripe mapping for baseline
comparisons after promotion. No separate shader capability is introduced.

## Replay

```sh
VXL8R_TEST_VULKAN=1 PYTHONPATH=../vxl8r/src .venv/bin/python -m pytest ../vxl8r/tests/test_face_average.py -q
PYTHONPATH=../vxl8r/src .venv/bin/python components/OrdinaryLight/scripts/profile_primary_prefixes.py --stage full-face-locality
PYTHONPATH=../vxl8r/src .venv/bin/python components/OrdinaryLight/scripts/profile_primary_prefixes.py --stage full-face-locality --overview
PYTHONPATH=../vxl8r/src .venv/bin/python components/OrdinaryLight/scripts/profile_primary_prefixes.py --stage full-face-locality --projection orthographic
```

Logs: `artifacts/face-locality/`. Python compilation and whitespace checks pass.
