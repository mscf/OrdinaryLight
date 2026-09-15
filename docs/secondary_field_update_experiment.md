# Secondary hit capture: field update experiment

Status: promoted to production with user approval. The typed OrdinaryShade
source and generated GLSL now write only the two changed fields; all 48
manifest-owned secondary shader variants have been rebuilt. No vxl8r adapter
changes or settings are required.

Previously, the custom/native secondary shader captured the first secondary hit
by loading an entire 128-byte SecondaryPathState, replacing position_valid and
normal_pdf, and storing the entire record. Production now writes those two vec4 fields
explicitly, preserving normal_pdf.w and every other field. It changes neither
record layout nor allocation size, transport, bounce counts, or history rules.
The measured gain does not prove a specific driver memory transaction count.

The diagnostic harness reconstructs the old whole-record write in a temporary
OrdinaryShade source variant for the baseline renderer. The candidate uses the
installed production shader. It changes no installed renderer
code or generated production shaders. Private compiler interception is confined
to the diagnostic. Both renderers use compact public primary-hit exports.

## Measurements

RTX 5090 Laptop GPU; perspective camera, native 3840×2160, animated 512×512
scene, GPU animation/layout, tight bounds, four bounces, one sample per pixel,
524288 paths. Forty measured frames after twelve warmup frames, alternating
baseline/candidate order at matching animation times. Headless GPU timings
exclude swapchain/compositor presentation and do not predict displayed FPS.

| View / GPU stage | Baseline | Candidate |
| --- | ---: | ---: |
| Close-up, first secondary | 5.153 ms | 4.122 ms |
| Close-up, total | 20.812 ms | 19.838 ms |
| Overview, first secondary | 0.529 ms | 0.519 ms |
| Overview, total | 13.736 ms | 13.716 ms |

Close-up total improves about 4.7%; overview is effectively unchanged. Camera
composition matters: the overview performs substantially less secondary work.

Final-frame raw and denoised HDR match exactly in both views. Close-up face
averaging differs by at most 1.79e-7; overview matches exactly. Compact hit
identity and validity match exactly. A 320×180 smoke run also passed all output
checks. Material fixtures are separate correctness tests, not performance data.

## Replay

From the Ordinary checkout:

```sh
PYTHONPATH=../vxl8r/src .venv/bin/python components/OrdinaryLight/scripts/profile_primary_prefixes.py --stage full-secondary-fields
PYTHONPATH=../vxl8r/src .venv/bin/python components/OrdinaryLight/scripts/profile_primary_prefixes.py --stage full-secondary-fields --overview
```

Use `--material-fixture reflective`, `refractive`, or `emissive` for material
checks; emissive parity needs `--stable-emitter-fixture` to remove independent
atomic emitter-list ordering as a sampling confounder. These fixtures exercise
native shader material behavior; they do not add optical material support to
vxl8r's public voxel format.

All three 640×360 material fixtures passed: reflective, refractive, and emissive
raw HDR, denoised HDR, face-averaged HDR, and compact identity/validity matched
exactly after three warmup and four measured animated frames. Logs are archived
alongside the proposed source patch. Production validation is recorded below.

## Production verification

The promoted source passed generation equality checks and compiled all 48
secondary variants. Source/ABI/manifest tests: 32 passed, 9 skipped, 12 subtests
passed. The headless Vulkan regression set passed all 31 tests, covering
secondary tile initialization, one/two samples, sparse capture, hit/miss,
resize, public hit formats, graph lifetime, and history controls.

A fresh 40-frame animated native 4K close-up comparison used the installed
production shader for the candidate and reconstructed the old whole-record
write only for the baseline. First secondary shading measured 5.174 → 4.101 ms;
total GPU measured 21.073 → 19.830 ms (about 5.9%). Raw and denoised HDR matched
exactly; face output differed by at most 1.19e-7; hit identity/validity matched
exactly. These headless results exclude presentation. The two comparisons measured a 4.7–5.9% total improvement; this is not a
guarantee of the same gain in every scene.

The existing local vxl8r integration picks up the change on restart, without
new flags, API changes, or renderer overrides. The release has not been
committed or published as part of this change.

Production material replays also passed exact raw, denoised, face-output, and
hit parity for reflective, refractive, and deterministic emissive fixtures.
See `production-*.jsonl` in the artifact directory.
