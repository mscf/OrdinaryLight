# Denoiser preparation read experiments

Status: diagnostic only; neither candidate is promoted. Production renderer,
shader artifacts, public API and vxl8r defaults remain unchanged.

After secondary field-write optimization, preparation still costs roughly
2.5 ms at native 4K. Two typed OrdinaryShade variants were tested:

- `full-prepare-fields`: remove the local whole-record secondary load and read
  fields directly at each use site. Preserve the full secondary ABI and all
  calculations, including history and optional guide paths.
- `full-prepare-single`: retain the existing record load but constant-fold the
  known diagnostic configuration: sampled indirect, one sample, sample index
  zero, and planar mirror guides disabled. This is NOT a general replacement
  for preparation with other sample/guide configurations.

The harness compiles temporary typed source, selects the variant only for the
candidate runtime, and restores the compiler hook on exit. No handwritten
shader logic or production renderer patches are introduced.

## Results

RTX 5090 Laptop GPU; animated perspective close-up, native 3840×2160,
512×512 voxel scene, GPU animation/layout, tight bounds, four bounces,
one sample, compact identity exports, 524288 paths. Each comparison alternates
baseline/candidate order for 40 measured frames after 12 warmup frames at
matching animation times. GPU diagnostics are headless and exclude presentation.

| Experiment | Baseline prepare | Candidate prepare | Baseline total | Candidate total |
| --- | ---: | ---: | ---: | ---: |
| Direct field reads | 2.536 ms | 2.743 ms | 20.249 ms | 20.264 ms |
| Single-sample specialization | 2.498 ms | 2.459 ms | 19.986 ms | 19.890 ms |

Direct field reads regress the preparation stage and offer no total improvement.
Single-sample specialization saves only 0.039 ms in preparation; total shifts
0.096 ms (about 0.5%). These results do not justify an additional production
variant or a new runtime selection interface. Differences in other stages
also contribute to total changes; stage medians are not additive.

Both runs passed exact final-frame raw and denoised HDR parity and compact hit
identity/validity parity. Face-averaged output differed by at most 1.20e-7.
Logs are archived under `artifacts/prepare-read-experiments/`.

## Replay

From the Ordinary checkout, with the local vxl8r dependency available:

```sh
PYTHONPATH=../vxl8r/src .venv/bin/python components/OrdinaryLight/scripts/profile_primary_prefixes.py --stage full-prepare-fields
PYTHONPATH=../vxl8r/src .venv/bin/python components/OrdinaryLight/scripts/profile_primary_prefixes.py --stage full-prepare-single
```

The profiler now records preparation time alongside primary, first secondary,
face output and GPU total. Earlier workgroup experiments (64/128/256) were also
neutral; see `temporal_reset_optimization.md`.

A more substantial follow-up should evaluate avoiding intermediate signal
traffic between preparation and temporal filtering. This needs an explicit
upstream design and confirmation: multi-sample accumulation, sampled pixel
mapping, custom motion/identity, diagnostic outputs, and standalone graph
operations must retain their current contracts. These measurements alone do
not establish the performance benefit of such a change.
