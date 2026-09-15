# Compact selected-diffuse continuations

Status: stopped at the user's request; retained for reference only. Not enabled
in the viewer or native presenter. The maintained native GI path is the baseline.

The experiment classifies the eligible standard PBR branch of selected-diffuse
primary replay. Rejected diffuse draws finish in the classifier. Potential
specular draws and selected diffuse draws append a tile-local pixel index to a
GPU list, then return. A one-thread OrdinaryShade kernel prepares indirect
arguments; a 64x1 replay kernel finishes the listed pixels. Fallback materials
retain the original classifier path. All shader behavior is typed OrdinaryShade.

The list and control buffer persist per native pipeline/slot. At the default
524,288-pixel tile capacity, they require about 2 MiB. Control is reset before
every tile, including edge tiles; the native secondary queue is not reset
between classify and resume. The graph orders transfer reset, classification,
argument preparation and indirect reads without CPU readback or waits between
stages. Resource declarations combine compute and indirect access to control.

This is not a minimal continuation kernel: listed pixels repeat primary setup,
material evaluation and direct-light work. Extension evaluation must therefore
be repeatable. Profiling counters can count repeated evaluation. It is not an
appropriate default for arbitrary side-effectful extensions. A future experiment
would need to avoid this repeated setup without adding excessive state traffic.

## Animated 4K result

Paired alternating order; perspective camera, animated 512x512 voxel scene,
3840x2160, 4 bounces, 524,288 tile capacity, 12 warmup + 40 measured frames.
Both paths use selected diffuse, planar visibility, one visibility dispatch,
quad links, 32x2 classifier/replay, identity hits, environment samples 0.

| GPU median | Existing selected replay | Compact continuation |
| --- | ---: | ---: |
| Primary | 8.712 ms | 11.400 ms |
| Total | 21.521 ms | 23.697 ms |

The candidate regressed total time by 10.1%. This comparison is against the
selected-diffuse experiment, not the maintained viewer path. It does not measure
presentation or prove CPU throughput. Normal stochastic face selections differ
between the two runs; use fixed-mask tests below for transport parity. No
quality equivalence claim follows from the timing run alone.

Reproduce with `PYTHONPATH=../vxl8r/src .venv/bin/python
components/OrdinaryLight/scripts/profile_primary_prefixes.py` and:

```text
--stage full-selected-diffuse --selected-baseline --compact-continuations
--planar-visibility --single-visibility-dispatch --quad-visible-links
--replay-width 32 --environment-samples 0 --primary-hit-format identity
--width 3840 --height 2160 --warmup 12 --frames 40
```

## Validation

- 70 primary ABI/compiler/operation tests pass, including invalid continuation
  variants and insufficient control/list capacities.
- 8 headless GPU argument/reset tests cover counts 0, 1, 63, 64, 65, 262144,
  262145 and 524288.
- Fixed masks `none` and `all`: animated 161x97, local history and resets every
  three frames, exact raw/upstream/output HDR, guide and sampled-hit parity.
- All-selected 1025x513 spans multiple tiles: exact raw and upstream HDR,
  guides and sampled hits; final face-averaged output maximum error 2.98e-8.
- Environment-sample fallback with the refractive fixture: no eligible compact
  continuations, exact per-frame HDR, guide and sampled-hit parity through
  animation and history resets.
- Generated primary source check passes. All 52 default primary SPIR-V files
  recompiled byte-identically to packaged artifacts; defaults are unchanged.

Logs: `artifacts/selected-diffuse/compact-{none,all,args,multitile,fallback,4k}.log`.
