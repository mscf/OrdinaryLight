# Native resolve scratch aliases live secondary state

Confirmed while testing compact local-only state and now fixed upstream.

Native resolve enables reservoir/seed writes only when indirect reuse is enabled.
The public `VulkanPathResolve.operation()` and shared `path_resolve_operation()`
accept `seed_reservoirs=False` to disable seeding independently of signal capture.
Its default remains True for existing standalone callers.

For sampled-indirect denoising without reuse, resolve is HDR-only and neither reads
nor writes secondary state. Legacy non-sampled denoising still writes resolved
signals, while leaving seed/reservoir placeholders untouched. Real indirect reuse
continues seeding. The graph declares only the resources used in each mode, and
native cached dispatches include the seeding policy in their key. Active seeding
rejects aliases with path/secondary/camera storage before recording.

The implementation changes typed OrdinaryShade source and regenerates the packaged
resolve shader/SPIR-V. It adds no per-frame allocation, wait or readback. The
compact-state profiler no longer supplies separate scratch: it exercises the
native dormant placeholders with the fixed production policy.

## Original failure and evidence

`targets/vulkan/path_resolve_graph.py` binds `secondary_path_buffer` as both
indirect reservoir and seed storage when the corresponding frame resources are
absent. The comment describes dormant placeholders, but `record_path_resolve`
enables capture when either indirect reuse or denoiser signals are active.
With sampled-indirect denoising and reuse disabled, the shader can therefore
write reservoir/seed data into the first secondary record. The public
`VulkanPathResolve` constructor rejects such aliases; the native adapter does not.

The generated `wavefront_path_to_hdr` writes the representative seed and an empty
reservoir even when there is no secondary contribution. This can overwrite the
first record's position/validity and leading normal/PDF words. In the old layout,
primary position and identity live later, which hides some of the damage. In the
compact experiment, the overwritten validity word also served as a layout marker,
so preparation could interpret stale trailing fields as current guides.

GPU evidence at odd 161×97 extent with animation/local history:
- Native bindings 2, 3 and 5 were confirmed to share a handle on both renderers.
- A depth-guide mismatch appeared at pixel (0,0): 26.707579 versus 26.735296.
  It later produced a small denoised HDR mismatch while raw HDR still matched.
- Giving both diagnostic renderers distinct persistent 24-byte reservoir and
  4-byte seed scratch buffers eliminated the mismatch. All normal/roughness,
  motion, depth, diffuse and specular channels then matched exactly across seven
  checked frames, as did HDR and sampled hits/rays. Active-ray and refraction
  checks passed with these separate buffers too.

The original workaround belonged only to `--selected-local-packed` in the headless profiler.
Allocation occurred at kernel construction; kernels closed before their retained
scratch allocations. No allocation or readback is inserted between render stages.
It assumes the diagnostic no-reuse 1×1 reservoir extent; generalize using the
actual extent before adopting any allocation-based solution upstream.

The implemented fix suppresses seed/reservoir generation when
sampled-indirect preparation is the only consumer and indirect reuse is disabled.
The sampled prepare shader already produces denoiser signals. Legacy non-sampled
signal resolve and real indirect-reuse behavior are retained. Merely broadening
barriers cannot repair writes to the wrong allocation.

Evidence: `artifacts/selected-diffuse/local-packed-guides.log` (failure),
`local-packed-none-valid.log`, `local-packed-all-valid.log`, and
`local-packed-refractive-valid.log` (correct separate-scratch runs).


## Fix validation

- GPU tests cover 24 resolve combinations: no capture, legacy signal capture,
  sampled preservation, seeding enabled/disabled, one/two reservoir pixels and
  valid/missing secondary hits. Each accumulates two samples. Real seeds and
  reservoirs still update, while disabled seeding leaves their sentinels intact.
- Two additional GPU regressions bind native-style reservoir/seed placeholders to
  the secondary buffer. Sampled mode preserves every record byte; legacy mode
  changes only its two output signal fields, including at the first record.
- CPU checks cover all eight native reuse/denoiser/sampled-policy combinations,
  actual resource access masks, active-alias rejection and cached dispatch reuse
  when reservoir seeding is toggled.
- The native packed-layout reproduction now uses dormant aliases without separate
  scratch. Every channel of normal/roughness, motion, depth, diffuse and specular
  matches exactly across seven checked animated frames, including valid/reset
  history; raw/denoised/final HDR and sampled hits/rays also match exactly.
- Native triangle rendering with real indirect-reuse storage/candidates passes
  capture/replay comparison across two samples, history and resize.
- Maintained native 4K rendering passes an animated headless control check.
  Raw and denoised HDR match exactly; final face-averaged output differs by at most
  1.49e-7. GPU medians were 17.68 and 17.53 ms for the two identical controls;
  this is a post-fix smoke check, not a before/after speedup measurement.

Evidence for the implemented fix: `artifacts/selected-diffuse/resolve-alias-*.log`.
Generated OrdinaryShade resolve source is current, the packaged SPIR-V was rebuilt,
and whitespace checks pass. No GUI/presentation performance claim is made.
