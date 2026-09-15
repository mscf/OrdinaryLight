# Selective primary-hit export experiment

The experiment below has now been promoted to a public `identity` output option.
See [the API contract](native_gi_composition.md#selecting-primary-hit-exports).
The native-GI vxl8r viewer selects it by default; `--primary-hit-format full`
restores full diagnostics. The historical prototype measurements below used
oversized allocations; the production option allocates 20 bytes per record.

Production validation (`--stage full-public-identity`) measured 23.240 → 21.984 ms
GPU total in animated native 4K close-up, approximately 5.4% faster, with exact
raw/denoised HDR parity and face-average maximum difference 1.04e-7. Compact
identity/validity and exact 20-byte allocation checks passed. The three material
fixtures passed again through the real API, including the deterministic emissive
fixture. Upstream two-sample glass/resize/lifetime checks and 24 vxl8r GPU tests
also passed. Logs are in `artifacts/public-primary-hits/`.

The old compiler-override prototypes require their pre-promotion source version;
use `--stage full-public-identity` to reproduce current comparisons without any
shader or consumer override.

The diagnostic `--stage full-compact-hits` writes five uint32 words per sampled
pixel: the four application identity components, followed by explicit hit
validity (0 or 1). Identity and validity remain independent; no identity value
is reserved to encode a miss. Pixel indexing is unchanged, including the sample
offset. All shader logic is authored in OrdinaryShade.

This reduces exported data from 96 to 20 bytes per sample/pixel. At native 4K
that is 158.203 MiB rather than 759.375 MiB written per sample. **The prototype
retains the original allocation size** to isolate shader/consumer effects and
reuse the existing diagnostic readback machinery. It does not yet save resident
VRAM. Unwritten tail bytes are never consumed by the candidate shaders.

The matching temporary face-averaging consumer reads only validity and identity.
Primary and secondary transport, material evaluation, normals, optical state,
denoiser guides and custom history exports remain unchanged. No new dispatch,
allocation, readback or CPU wait is introduced between production frame stages;
the comparison harness itself uses serialized GPU timings and final readbacks.

## Animated native 4K results

RTX 5090 Laptop GPU, milliseconds (median):

| View/run | Full primary | Compact primary | Full total | Compact total | Total reduction |
| --- | ---: | ---: | ---: | ---: | ---: |
| Close-up | 5.908 | 5.100 | 23.088 | 21.864 | 5.3% |
| Close-up repeat | 5.976 | 5.190 | 23.353 | 22.319 | 4.4% |
| Overview | 5.411 | 4.420 | 16.431 | 14.932 | 9.1% |

The primary interval excludes setup, which remained approximately 0.73–0.80 ms.
Face output improved by approximately 0.27 ms in close-up and 0.49 ms in the
overview. Medians from separate stages are not precisely additive. Both
close-up runs and the overview preserved raw and denoised HDR exactly. Final
face-averaged maximum differences were 2.09e-7, 1.19e-7 and zero respectively;
all compact identity and validity comparisons passed.

This is a stronger result than rearranging full records. It supports pursuing
an opt-in public selection API, but the normal viewer and public output ABI are
unchanged by this diagnostic. No resident-memory or presentation-FPS improvement
is claimed from these measurements.

## Validation

The normal 129×97 smoke test passed exact raw, denoised and face-averaged HDR
comparison, plus exact identity and validity comparison against full exports.
Compact exports intentionally do not provide normals, positions or camera rays;
the diagnostics do not claim those missing fields are available in this format.

vxl8r's current public voxel materials provide albedo and emission with a rough
dielectric surface. They do not expose reflective/refractive material authoring.
The diagnostic therefore supplies temporary typed material callbacks to both
baseline and candidate for the reflection/refraction checks. Reflection uses
zero roughness and metallic=1; refraction uses zero roughness, transmission=1
and IOR=1.5. The emissive fixture uses real resident voxel emission (3, 1.5, 0.5)
and the normal emitter distribution. These are controlled regression fixtures,
not new application material features or exhaustive optical-boundary tests.

Reflective and refractive fixtures passed exact raw, denoised and face-averaged
HDR comparisons and exact compact identity/validity comparisons at 257×145.
They ran four warmup frames followed by six animated frames each. These checks
compare the final output frame, not every intermediate frame.
The final run of each material fixture also explicitly rejected non-finite
values in both renderers' raw, denoised and face-averaged HDR; all passed.

The first emissive comparison failed (maximum raw HDR difference 0.578125).
An unchanged-versus-unchanged `--stage full-control` run reproduced the same
maximum error, so this is not evidence of an export-induced transport change.
The normal emitter list uses atomic appends, making order-dependent seeded
sample comparisons unsuitable for strict parity. `--stable-emitter-fixture`
temporarily enumerates active exposed emissive faces in slot/face order within
the selected chunk; it retains the count, uniform sampling distribution and
PDF. With that reference sampler, raw, denoised and face-averaged HDR and compact
identity/validity matched exactly. Its raw maximum was 4.25, so this was not an
empty-lighting comparison. This slow reference is diagnostic only; no emitter
sampling implementation was changed in production.

## Reproduction

From the Ordinary checkout, with no other GPU workload:

```sh
PYTHONPATH=../vxl8r/src .venv/bin/python components/OrdinaryLight/scripts/profile_primary_prefixes.py --stage full-compact-hits
PYTHONPATH=../vxl8r/src .venv/bin/python components/OrdinaryLight/scripts/profile_primary_prefixes.py --stage full-compact-hits --overview
PYTHONPATH=../vxl8r/src .venv/bin/python components/OrdinaryLight/scripts/profile_primary_prefixes.py --stage full-compact-hits --material-fixture refractive --width 257 --height 145 --warmup 4 --frames 6
```

The material fixture also accepts `emissive` and `reflective`. The 4K benchmark
uses the default diffuse scene: perspective camera, animation, four bounces,
1 spp, GPU layout, tight bounds, 524,288 paths and a 512×512 voxel floor. Each
4K run alternates baseline/candidate order, 12 warmup frames and 40 samples.
These are headless GPU measurements, excluding actual presentation, not FPS.

For strict emissive parity use `--material-fixture emissive
--stable-emitter-fixture`. Ordinary stochastic emitter comparisons and their
failed control are retained alongside successful tests in the artifact logs.

The experimental ABI and temporary compiler/consumer selection must not be
used as a production integration. Adoption requires a public output-selection
contract, correct allocation sizes, metadata and readers, retention of the
existing full-output default, and coverage for resize, samples and consumers.
Logs are archived in `artifacts/compact-primary-hits/`, including both the
original emissive mismatch and the unchanged-renderer control.
