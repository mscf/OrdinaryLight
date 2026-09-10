# OrdinaryShade shader authoring

All OrdinaryLight-authored GPU algorithms now pass through typed OrdinaryShade
Python. GLSL, WGSL, and SPIR-V are generated artifacts. This includes the fused
primary and secondary stages used by `raster_feature_viewer`, texture and light
sampling, ReSTIR and indirect reuse, volume transport, public transport kernels,
custom material/geometry dispatch, material graph resource reads, diagnostic
queries, and portable WebGPU volume rendering/upload.

The rule is recorded in `../AGENTS.md` and `../CONTRIBUTING.md`. The inventory in
`ordinarylight/shaders/authorship.json` identifies each generated artifact's
source. `scripts/check_shader_authorship.py` rejects unclassified shader files
and executable shader bodies embedded in production Python strings.

Existing upstream FSR remains the explicitly authorized temporary exception.
FSR2 is planned for removal; FSR1 must be replaced or rewritten in OrdinaryShade.
The FSR2 preparation pass is already OrdinaryShade. Upstream native FSR2 sources
and `fsr1_easu.glsl` are not claimed as migrated.

## Source organization

* `ordinarylight/shaders/*_programs.py` and the existing typed generator/library
  modules own shader algorithms.
* `*_layout.py` contains only descriptor/structure ABI, constants, includes,
  backend directives, and insertion markers. These declarations preserve the
  existing Vulkan ABI; they must not contain algorithm bodies.
* `materials/shade.py`, `shaders/scene_dispatch.py`, and `shaders/dynamic.py`
  specialize typed Python functions for scene-dependent dispatch and expression
  graphs before compiling them with OrdinaryShade. They do not emit GLSL bodies
  directly. External function signatures link actual typed implementations or
  caller-provided geometry programs.
* Public material authoring and viewer controls retain their existing interfaces.
  Historical shader language-selection defines no longer select handwritten
  implementations. Rendering feature variants remain supported.

## Regeneration and validation

OrdinaryLight 0.4.0 requires OrdinaryShade 0.1.0a5 or newer. CI and the portable
installation checker pin the validated compiler commit
`98d12db5dfef408b32134bf5aa0c4a2798eab247`. The temporary compiler patch is no
longer required.

```
python scripts/check_shader_authorship.py
python scripts/check_generated_shaders.py
python scripts/compile_shaders.py
python -m pytest tests -q
ORDINARYLIGHT_TEST_VULKAN_TRANSPORT=1 python -m pytest tests/test_transport_gpu.py -q
```

Each `scripts/generate_*.py` supports `--check`. Run the relevant generator
without that flag after editing its typed source, then rebuild SPIR-V. The
raster/denoiser artifact generators additionally require glslang and Naga; their
binary and WGSL validation remains in the packaging CI job. The source archive
includes the generators, typed modules, and this documentation.

Changes to estimators or shader control flow also require GPU image checks and
representative timing measurements; passing a source inventory is not proof of
rendering equivalence.

## Migration verification record

The migration was checked on the local RTX 5090 GPU:

* All 76 generated shader artifacts pass source checks. The packaged SPIR-V
  build completes for every manifest variant.
* Rough reflections at 2052 × 1764, shared primary, 2 spp and two reservoirs:
  the migrated frame measured 30.80 ms versus 31.04 ms for a repeated original
  reference. Primary measured 11.42 ms versus 11.31 ms. These are comparable
  timings, not evidence of an optimization. HDR error against the first
  reference was the same 0.277% relative RMSE as the repeated original.
* The custom-attribute GI comparison had maximum absolute error 1.2e-7.
* Portable volume raymarch/slice rendering matched the original WGSL on WebGPU;
  3D volume upload also passed a GPU readback check.
* Both secondary implementations matched their own pre-migration outputs
  exactly in the volume multiple-scattering/custom-material scene. The
  cross-implementation comparison still differs (28.4% relative RMSE at
  128 × 72, four samples). That discrepancy also exists in the original
  shaders; this migration preserves their behavior rather than changing the
  volume estimator.

The compiler regression suite additionally covers output parameters, static
specialization, loop scoping, 3D images, runtime-array lengths, bitcasts, and
Vulkan atomic operations, including rejection of unsupported WGSL operations.
