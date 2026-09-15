# Shared primary continuation evaluation

Status: diagnostic only; no production shader or viewer default changes.

The temporary typed OrdinaryShade `samplePbr` override shares half-vector,
Fresnel, GGX distribution and diffuse evaluation between the PDF, BSDF and
denoiser lobe-fraction calculations. Sampling and random draws are unchanged.
The separate PDF view-half clamp and invalid-hemisphere evaluation remain.
Only fused primary uses the override; secondary shading remains unchanged.

## Animated native 4K comparison

RTX 5090 Laptop GPU, perspective close-up, 512×512 voxel scene, GPU animation
and layout, tight dynamic bounds, four bounces, one sample, 524288 path capacity,
compact hit outputs and production paired spatial filtering. Forty measured
frames after twelve warmups, alternating baseline/candidate order at matching
animation times. Default content-reset history policy. Headless GPU measurements
exclude CPU and presentation costs.

| Run | Existing primary | Candidate primary | Existing total | Candidate total |
| --- | ---: | ---: | ---: | ---: |
| Initial | 4.922 ms | 4.889 ms | 19.184 ms | 19.107 ms |
| Repeat | 4.859 ms | 4.825 ms | 19.002 ms | 18.921 ms |

Primary improves by about 0.033 ms in both runs. Total improves by 0.077–0.080 ms,
approximately 0.4%. Unmodified secondary timings also improve by 0.048–0.056 ms;
the full total difference cannot be attributed to the edited primary arithmetic.
This does not establish whether the compiler previously duplicated computations.
The small primary gain does not justify promoting a duplicate implementation of
the shared PBR contract. A production version would need a maintained common
helper and broader material/transport coverage.

Final raw and denoised HDR match exactly in both 4K comparisons. Face-averaged
output differs by at most 1.50e-7; compact identity/validity match exactly.
A 321×181 animated check also passes per-frame HDR comparisons for three warmup
and eight measured frames. No separate raw continuation-record comparison,
camera-ray comparison, optical-material matrix or valid-history matrix is claimed.
Python compilation and diff whitespace checks pass.

## Replay

```sh
PYTHONPATH=../vxl8r/src .venv/bin/python components/OrdinaryLight/scripts/profile_primary_prefixes.py --stage full-primary-shared-pbr --width 321 --height 181 --warmup 3 --frames 8 --check-each-frame
PYTHONPATH=../vxl8r/src .venv/bin/python components/OrdinaryLight/scripts/profile_primary_prefixes.py --stage full-primary-shared-pbr --warmup 12 --frames 40
```

Logs: `artifacts/primary-shared-pbr/`. The next larger measured target is vxl8r
face averaging (approximately 3.03 ms in these runs), where separate timings for
linking, reduction and resolve can guide a bounded optimization.
