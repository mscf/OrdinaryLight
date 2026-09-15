# Face-budgeted diffuse transport prototype

Status: **Stopped at the user's request.** Return to the maintained native GI
renderer as the working and performance baseline. Selected-diffuse grouping,
visibility replay and compact-continuation variants remain disabled diagnostic
experiments. Their results are retained for reference, not as the next work queue.
Do not resume this optimization direction without a new user request.

The prototype reduced diffuse continuations but did not improve total GPU time.
The maintained path remains the normal viewer implementation; no viewer rollback
is required. Keep independent correctness fixes, including resolve scratch alias
handling. Earlier sections below record the experiments rather than current plans.

## Working GPU sample planner

`scripts/face_lighting_plan_experiment.py` consumes the resident visible-face
lists from a completed vxl8r reference render. One GPU invocation per visible
face selects at most K actual camera-hit pixel indices. This avoids replacing
the camera's sampled hit by a fabricated face-center intersection.

Each face's linked pixels are partitioned into K disjoint rank intervals. A
seeded pseudorandom draw chooses one rank in each interval; integer rejection
avoids modulo mapping bias. The output stores pixel index, group index, the
interval's population fraction and total face population. Unequal interval
sizes receive unequal weights, summing to one. This is a screen-visible sample
population, not an area-uniform sample of the complete world-space voxel face.
Selection order follows the GPU-built lists and can change between frames.

Outputs are a compact sample array, per-group ranges and counters. Allocation
occurs at construction; clearing and selection form one reusable GPU operation.
No CPU readback or wait occurs between those passes. Explicit diagnostic reads
occur only after completion. Borrowed grouping resources must remain alive and
unchanged until the plan completes. The prototype requires no more than 2**24
pixels so the existing float population counts remain exact, and budgets 1..32.

Six GPU tests cover direct and hashed identity domains at budgets 1, 8 and 32,
including exact membership, unique selected pixels, budget counts, positive
weights summing to one, empty-frame resets and resource reuse. These tests do
not constitute validation of a new lighting estimator or temporal reconstruction.

## Measured opportunity at native 4K

RTX 5090 Laptop GPU, current four-bounce reference renderer, 3840×2160,
animated perspective scenes. Planning measurements use the final frame after
40 measured reference frames and 12 warmups. Eight repeated plan timings follow
two warmups; they reuse the same grouping with different seeds. This is not an
animated end-to-end lighting benchmark.

| Final-frame population | Close-up | Overview |
| --- | ---: | ---: |
| Hit pixels in face lists | 8,294,400 | 1,445,406 |
| Visible faces | 14,876 | 262,138 |
| Selected pixels, budget 8 | 118,561 | 1,313,040 |
| Selected fraction | 1.43% | 90.84% |
| Planner GPU median | 0.632 ms | 0.508 ms |
| Median pixels per face | 117 | 5 |

All final-frame plans pass exact membership and count checks and per-face weight
normalization. Reference raw/denoised HDR and compact hits still match because
the plan does not alter transport. Plan timing excludes constructing visible-face
lists, primary visibility, tracing, reconstruction and readback. In particular,
**98.57% fewer selected close-up pixels is not a renderer speedup claim**.

The overview leaves little opportunity at this budget while still paying for
planning. A production policy needs a small-face/per-pixel fallback and must
account for grouping overhead. A fixed eight-sample budget is not universally
beneficial. Cache/history reuse and dirty-face lighting propagation are not
implemented by this prototype.

## Required upstream transport work

Inspection confirms that `processPrimaryPixel` currently samples a PBR mixture:
each continuation's evaluated contribution contains both diffuse and specular
terms. `transportLastSpecularFraction` splits diagnostic signals; it does not
provide independently scheduled ray streams. Skipping unselected continuations
would therefore suppress reflections as well as diffuse lighting.

The maintained diffuse term also includes Fresnel dependence; it must not be
silently replaced by a purely Lambertian estimator. Sample/PDF changes require
energy and convergence tests, not bitwise comparisons with the old noise pattern.

The next implementation steps are:

1. Expose a reusable primary-visibility boundary before lighting. Preserve the
   same camera ray and complete native hit payload; public compact hit identity
   alone cannot reconstruct every custom material or optical boundary. Reuse
   existing native intersection contracts and retain primary ReSTIR behavior.
2. Let an application insert GPU grouping/selection operations at that boundary.
   OrdinaryLight consumes generic sample indices/weights; voxel slot/face identity
   and sample-budget policy remain in vxl8r. The reference planner currently runs
   after lighting because that boundary does not yet exist.
3. Separate diffuse continuation sampling from view-dependent specular transport,
   with explicit PDFs and contribution weights. Keep emission/direct-light terms
   accounted once. Preserve optical/custom-scattering behavior through a fallback.
   Simply adding a full per-pixel specular path plus face diffuse paths could
   increase ray work; sample allocation must be measured and justified.
4. Reconstruct the shared diffuse signal before native denoising, retaining
   per-pixel guides and specular signals. Both direct and face-averaged output
   should consume the same transport/history. Existing optional final face
   averaging can still operate on the composed image.
5. Validate shadows, reflective/optical fixtures, animation/history invalidation,
   resize, identity generation reuse and total GPU cost. Sampling a face's
   lighting is an approximation to its spatial variation; this must be assessed
   with image quality/convergence tests before enabling the mode in the viewer.

No CPU face selection, hidden upstream descriptor overrides, or handwritten
shaders are needed. All implemented planning logic is typed OrdinaryShade.

## Replay

```sh
PYTHONPATH=../vxl8r/src .venv/bin/python -m pytest components/OrdinaryLight/scripts/test_face_lighting_plan.py -q
PYTHONPATH=../vxl8r/src .venv/bin/python components/OrdinaryLight/scripts/profile_primary_prefixes.py --stage full-control --primary-hit-format identity --face-lighting-budget 8
PYTHONPATH=../vxl8r/src .venv/bin/python components/OrdinaryLight/scripts/profile_primary_prefixes.py --stage full-control --primary-hit-format identity --face-lighting-budget 8 --overview
```

Logs: `artifacts/face-lighting-plan/`. No release or commit was made.

## Primary capture/replay implementation

The upstream shader compiler now accepts `primary_visibility="capture"` or
`"replay"` for `wavefront_primary.comp` (non-inline compute), with `"fused"`
remaining the default. Both are authored in `fused_primary_programs.py` using
OrdinaryShade. `primary_bindings(primary_visibility=True)` adds set 0, binding 33.
`primary_operation()` accepts that binding and checks its sample-plane capacity.
`VulkanKernel(geometry_resources=...)` now supports retained native set-2 resources,
including the empty set-1 layout when there is no material bundle. No private
geometry descriptor replacement is needed by an application-owned kernel.

`PRIMARY_VISIBILITY_DTYPE` exposes the complete 112-byte NativeIntersection ABI:
position/distance, geometric/shading normals, identity, native address, texture
coordinates and previous position. A miss has `address.w == 0`; triangle payloads
retain native barycentric/address data for later evaluation. These are native
intersection payloads, not final normal-mapped denoiser guides. Records use
`(sample * render_height + y) * render_width + x`, regardless of tile order.
Coordinates refer to the GI render extent, not an upscaled display pixel.

Capture computes the existing jittered camera ray and intersects once. It writes
only this visibility cache; it does not clear reservoirs, initialize paths or
media, export guides, enqueue continuations, or shade. Replay regenerates the
same camera ray/RNG state, loads the intersection, then runs the original material,
optical, direct-light/ReSTIR and continuation logic. It does not retrace the primary
ray. All subsequent rays and lighting remain unchanged.

Callers must keep camera data, frame/sample indices, projection/jitter, scene/AS,
geometry, and shading resources unchanged between capture and replay. The cache
is not reusable across animation, camera changes or history generations. Capture
must precede every replayed pixel/sample, with a shader-write to shader-read
barrier. Explicitly order the graph nodes (e.g. `after=("capture",)`), since the
primary contract conservatively declares other shared outputs as read/write.
Allocate device-local cache storage before recording, retain kernels/resources
through completion, and retire or replace them only after their consumers finish.
There is no allocation, readback, submission or CPU wait between the two GPU
passes. The full payload costs 885.94 MiB per 4K sample plane; this is a baseline
correctness representation, not a final memory optimization.

The headless `primary_visibility_experiment.py` harness invokes these upstream
operations around each existing native tile. Its recorder interception is confined
to the diagnostic script. It does **not** yet expose an all-tiles visibility
boundary through `VulkanWavefrontPipeline.prepare()`, move face planning before
lighting, separate PBR lobes, reduce sample counts, or change viewer defaults.
The next step is full-frame scheduling with the application planner between
visibility and lighting; then independent diffuse transport and reconstruction.

Validation so far:

- 36 ABI/compilation/operation and related tests passed (one opt-in GPU test
  skipped); a deterministic regression covers public hit-view caching when Vulkan reuses a buffer handle.
- A headless triangle test passes with two samples/pixel, multiple tiles, repeated
  history, odd extents, resize and revisit. Full hit records and HDR match exactly.
- Animated voxel reflection, refraction and deterministic-emitter comparisons preserved raw and denoised
  HDR exactly; full-hit reflection checks also preserved sampled camera rays and
  identities. Refraction exercised local history across eight frames.
- Recompiled 184 packaged primary/hybrid/megakernel variants; all existing SPIR-V
  files were byte-identical. The normal fused path is unaffected by the optional
  specializations.
- Native 3840×2160 animation, 12 warmup + 40 measured frames: fused total median
  18.412 ms vs capture/replay 21.376 ms; primary 4.926 vs 7.917 ms. Final raw and
  denoised HDR were exact; final face-average differences were at most 8.95e-8.
  This is overhead, not a speedup. Face-budgeted work reduction must recover it.
  A preceding run measured 18.643 vs 21.458 ms total, with the same conclusion.
  The first diagnostic mistakenly used host memory; those earlier short parity
  runs validate correctness only and their timings must not be used.

Resize testing also exposed a public hit-view cache bug: numeric Vulkan buffer
handles can be reused after retirement. `native_gi_buffers()` now keys reuse on
allocation identity and swapchain generation, so a new frame cannot inherit an
invalid old view solely because its handle matches.

```sh
PYTHONPATH=components/OrdinaryLight .venv/bin/python -m pytest components/OrdinaryLight/scripts/test_primary_visibility_gpu.py -q
PYTHONPATH=../vxl8r/src .venv/bin/python components/OrdinaryLight/scripts/profile_primary_prefixes.py --stage full-primary-visibility --primary-hit-format identity
```

Logs are archived in `artifacts/primary-visibility/`.

## Full-frame planning before lighting

The `full-visibility-face-plan` diagnostic now records every primary capture tile
before the first lighting tile. It inserts vxl8r's new `VisibleFaceGroups`
operation, then the sample planner, then primary replay and the normal transport.
The capture cache uses global pixel indexing and is shared across those tiles;
compiled diagnostic graphs and device allocations are reused. There are no CPU
reads, queue submissions, or CPU waits between capture, grouping, selection and
lighting. End-of-run counter/HDR readbacks remain explicit diagnostics.

`vxl8r_render.backends.visible_faces` contains typed OrdinaryShade grouping and
population-counting kernels. They consume full NativeIntersection records and
have **no HDR or image dependencies**. They retain the existing striped direct
or hashed face indexing, so the planner can consume their lists unchanged.
Validity follows vxl8r's slot/face identity contract. Population now counts valid
camera hits, independent of whether a later lighting result is finite.

`primary_operation(..., visibility="capture"|"replay")` declares the cache as
write-only or read-only to the graph, matching the compiled variant. This avoids
an artificial cycle when an application reads visibility between the two stages.
Other bindings retain conservative access declarations.

Validation:

- Twenty-two CPU contract/compilation tests passed. Eight headless GPU tests
  cover direct/hash domains; budgets 1, 8, 32; empty and
  invalid identity planes; the highest valid slot; sample-plane bounds; membership,
  weights and allocation-free operations. Triangle parity tests cover both
  adjacent and full-frame capture, two samples/pixel, history, odd extents, multiple
  tiles and resize/revisit.
- A 960×576 multi-tile voxel comparison preserves sampled camera rays, full hits,
  raw HDR and denoised HDR. Full-frame replay would fail image parity if later
  tiles had not been captured before replay. Animated custom refraction with
  local history also preserves raw/denoised HDR exactly.
- Native 3840×2160 animation, 12 warmup + 40 measured frames: total GPU median
  **18.448 ms fused vs 23.810 ms capture/group/select/replay**, with all lighting
  still evaluated. Primary-stage aggregate: 4.936 vs 10.424 ms. This measures
  setup overhead, not a transport speedup.
- Both buffered plan snapshots covered all 8,294,400 pixels. They contained
  14,876 / 14,869 faces and selected 118,561 / 118,512 samples at budget eight:
  about **1.43%** of the original camera-hit population. Final raw and denoised HDR
  were identical; final face-average error was at most 8.95e-8.

This supersedes the earlier restriction that planning could only follow a
completed lighting render. It is still a diagnostic integration, not a viewer
mode or a new public prepared-frame hook. The selection seed is currently fixed
in this diagnostic; temporal rotation and adaptive fallback remain pending.
Nothing yet consumes the selected indices to reduce lighting. The next transport
change must independently schedule diffuse contributions, preserve the correct
PBR PDFs/weights and per-pixel specular/optical behavior, and reconstruct diffuse
signals before denoising. Simply suppressing unselected mixture rays is incorrect.

```sh
PYTHONPATH=../vxl8r/src:components/OrdinaryLight .venv/bin/python -m pytest components/OrdinaryLight/scripts/test_visible_face_groups.py components/OrdinaryLight/scripts/test_primary_visibility_gpu.py -q
PYTHONPATH=../vxl8r/src .venv/bin/python components/OrdinaryLight/scripts/profile_primary_prefixes.py --stage full-visibility-face-plan --primary-hit-format identity
```

Logs: `artifacts/visibility-face-plan/`.

## Separable continuation estimator and GPU selection consumer

`ordinarylight/shaders/pbr_lobe_programs.py` now implements an experimental
isotropic PBR continuation estimator in OrdinaryShade. It evaluates the same
base diffuse and specular BRDF terms as the maintained `samplePbr` helper,
including diffuse Fresnel dependence and its occlusion policy. It does not claim
to implement advanced material lobes or optical boundary events.

The estimator chooses a specular event with the maintained material-dependent
probability `p`, or a diffuse event with probability `1-p`. A specular event
samples a GGX half vector and weights only the specular BRDF by
`cosine / (p * q_GGX)`. A selected diffuse event samples the cosine hemisphere
and weights only the diffuse BRDF by `cosine / ((1-p) * q_cosine)`. An unselected
diffuse event is null. Specular choices do not consult the face-selection mask.
A below-surface GGX reflection is also null, rather than replaced with a cosine
sample under a mismatched PDF. There is at most one continuation per pixel.

Sampling density and BRDF regularization are separate: the former matches the
actual normalized half-vector sampling distribution, while the latter preserves
the maintained BRDF's denominator floors. `pbrLobeMisWeights` provides stable,
complementary power-heuristic weights for a **single** lobe. Light sampling and
continuation must use matching densities, including their event probabilities
and sample counts; this helper is not yet wired into native area-light/ReSTIR
transport. Existing renderer behavior is unchanged.

The face planner and selection policy now live in vxl8r's
`vxl8r_render.backends.face_lighting`. The old OrdinaryLight diagnostic module
re-exports the planner for existing experiments. `SelectedDiffuseSamples` expands
its compact selected records into a persistent per-pixel `(weight, population)`
buffer using an indirect GPU dispatch. Unselected pixels contain zero. Grouping,
selection, mask construction and estimator execution can share one GPU graph,
without CPU readback or waits between them.

A face mean uses the stored stratum weights directly. Null diffuse draws still
count toward that estimator; renormalizing over surviving rays would bias it.
Multiplying selected values by `weight * population` is appropriate for a
population-weighted image-mean diagnostic, not a substitute for reconstructing
the dense diffuse signal before denoising. Direct lighting must remain accounted
once; specular and optical signals must not be pooled into the diffuse mean.

Validation:

- 18 GPU estimator tests, each using 1,048,576 draws, cover diffuse, mixed and
  metallic materials; normal and grazing views; occlusion; and direct-light MIS
  counts 0, 1 and 4. Means agree with independent double-precision hemisphere
  quadrature within six measured standard errors plus 3e-4 absolute tolerance.
  The separated BRDF sum is also checked against the maintained GPU evaluator.
- With identical random inputs, disabling diffuse leaves all specular samples
  bit-for-bit unchanged **within the new estimator**. This is not a bitwise
  comparison against the old mixture sampler.
- Three additional GPU tests feed actual face plans into the estimator at budgets
  1, 8 and 32. They verify mask weights/populations, empty-frame clearing, unchanged
  specular samples and integrated energy with spatially varying incident light.
- At budget eight, the synthetic 512×512 fixture has 2,065 faces and 16,520 selected
  pixels. The estimator produces 36,612 nonzero continuations (13.97% of pixels),
  including per-pixel specular events. **These tests evaluate analytic incident
  lighting; they do not trace scene rays or establish a frame-time improvement.**
- The 12 existing grouping/planner GPU tests still pass after moving the policy
  into vxl8r, for 15 tests in the selection/regression run.

Still pending: native ray-queue integration, matching direct/emissive/ReSTIR MIS,
weighted diffuse reconstruction before denoising, and render-quality/convergence
checks with animation and optical fallback. The viewer remains on its original
transport. No reduced-ray renderer mode has been enabled.

```sh
PYTHONPATH=components/OrdinaryLight/scripts .venv/bin/python -m pytest components/OrdinaryLight/scripts/test_pbr_lobe_estimator.py -q
PYTHONPATH=../vxl8r/src:components/OrdinaryLight/scripts .venv/bin/python -m pytest components/OrdinaryLight/scripts/test_selected_pbr_lobes.py components/OrdinaryLight/scripts/test_visible_face_groups.py components/OrdinaryLight/scripts/test_face_lighting_plan.py -q
```

Logs: `artifacts/pbr-lobe-estimator/`.


## Actual selected-diffuse rendering (2026-09-13)

The opt-in compiler flag `primary_lobe_selection=True` requires surface-only
primary replay, denoiser signal capture, and non-inline continuation.
`primary_bindings(diffuse_selection=True, primary_visibility=True)` adds:

- Binding 34: per-pixel vec2 selection weight/population (read-only). Primary
  tracing uses the positive-weight mask; reconstruction consumes the weights.
- Binding 35: per-pixel vec4 local direct diffuse RGB and eligibility/continuation
  flag (0 = fallback, 1 = eligible null draw, 2 = eligible nonzero continuation).

The public primary operation validates single-sample full-image buffer capacity
and replay mode. Unselected diffuse events are deactivated after continuation
flags are formed, before enqueue. Specular sampling remains independent of the
selection mask. This is one mixture event per pixel, not a full specular ray plus
an additional diffuse ray.

The vxl8r `SelectedDiffuseReconstruction` GPU operation computes the weighted face
mean of indirect diffuse, then adds each pixel's original direct diffuse and
specular. A null or specular draw contributes zero diffuse with its original
weight; surviving diffuse samples are not renormalized. Reconstruction runs
before HDR snapshot/denoising through the public GI pipeline builder. Images,
selection, and scratch buffers persist; the operation allocates nothing and
performs no readbacks or waits between stages.

Restrictions: basic opaque PBR, no sampled environment lighting, no area/native
emitters, one sample/pixel, full-resolution RGBA16F signals. Advanced materials,
optical boundaries, transmission, and custom scattering retain the original
sampler. A face must have homogeneous eligibility; the diagnostic checks rejected
pixel counts against face population and fails if eligibility is mixed. This is
a diagnostic contract, not a generic mixed-material fallback solution. The
native production scheduler remains unchanged; recorder interception is confined
to the experiment harness.

### Animated native 4K result

RTX 5090 Laptop GPU, 3840×2160, perspective, 512×512 horizontal domain, four
bounces, budget 8, 12 warmups and 40 alternating measured frames. Both paths use
environment_samples=0; this is not a comparison against the viewer's default
lighting settings. GPU times include the existing final face averaging.

| GPU median | Baseline | Selected diffuse |
| --- | ---: | ---: |
| Primary (candidate includes capture/group/plan/replay) | 4.75 ms | 9.90 ms |
| First secondary stage | 3.57 ms | 2.52 ms |
| Total | 17.43 ms | 21.89 ms |

On the final frame, all 8,294,400 pixels are eligible. 118,561 pixels are selected;
521,114 primary continuations have nonzero throughput and pass the enqueue
active-path check (6.28% of pixels, including per-pixel specular draws).
This is a shader diagnostic count, not an independent hardware ray counter.

The total is about 26% slower despite reduced continuation work. The first
secondary stage improves about 29%, while capture/group/plan/replay adds about
5.15 ms. Reducing this overhead and profiling remaining fixed per-pixel secondary
work are the next performance tasks. These measurements do not establish
presentation FPS or CPU overhead.

All outputs are finite and compact hit validity/identity matches. Raw HDR mean
absolute difference is 0.0508; final face-averaged output difference is 0.0162.
These are differences, not quality scores: the sampling pattern and spatial
estimator change. Convergence, spatial lighting bias, optical fixtures, temporal
history, and resize validation remain necessary before viewer integration.

A separate 160×96 fallback run with environment_samples=1 has zero eligible
pixels and exactly matching raw, denoised, and final HDR. GPU reconstruction tests
cover weighted null draws, local direct/specular preservation, misses, unsupported
faces and reuse after eligibility changes. The selected-estimator plus
reconstruction run passes four GPU tests; compiler/primary-operation checks pass
32 tests. All 52 packaged primary SPIR-V variants compile and remain byte-identical
to the existing packaged artifacts when the experiment is disabled.

```sh
PYTHONPATH=../vxl8r/src .venv/bin/python components/OrdinaryLight/scripts/profile_primary_prefixes.py --stage full-selected-diffuse --environment-samples 0 --primary-hit-format identity --width 3840 --height 2160 --warmup 12 --frames 40
PYTHONPATH=../vxl8r/src .venv/bin/python -m pytest -q components/OrdinaryLight/scripts/test_selected_pbr_lobes.py components/OrdinaryLight/scripts/test_face_reconstruction.py
```

Logs: `artifacts/selected-diffuse/`. Direct/emissive/ReSTIR MIS integration and
production scheduling are unfinished. No viewer defaults changed.


## Preparation profile and adjacent-pixel linking (2026-09-13)

The opt-in `--profile-selected-diffuse` diagnostic adds persistent timestamp
queries around capture/replay tiles and grouping, planning, mask and
reconstruction passes. Query reads happen after GPU completion. BOTTOM_OF_PIPE
markers serialize these measurements; per-operation values below are final-frame
samples from two frame slots, not whole-run medians.

| Preparation component | Measured time |
| --- | ---: |
| Capture, all tiles | 1.70–1.78 ms |
| Grouping, including linking/counting/clearing | 2.15 ms |
| Sample planning | 0.65 ms |
| Selection mask | 0.08 ms |
| Replay, all tiles | 6.12–6.18 ms |
| Reconstruction | 0.70 ms |

Within grouping, linking costs about 1.58 ms and clearing only 0.21 ms. The
secondary dispatch already uses `min(queue.count, queue.capacity)` through GPU
indirect arguments; it does not launch a full pixel population after ray
reduction. Its native timing interval also includes queue preparation and
synchronization. The reduced path changes the surviving rays and their memory
access/coherence; the observed ray-count ratio alone cannot predict its timing.
A separate ray/coherence breakdown remains to be measured.

`VisibleFaceGroups(..., quad_links=True)` now provides an alternative typed
OrdinaryShade linker. Each invocation handles up to four consecutive pixels
(which share the existing stripe assignment), links equal-face runs locally, and
publishes each run with one atomic exchange. It reuses the established direct
and hashed insertion rules. Misses, face boundaries and the final partial quad
remain supported. This changes linked-list ordering and therefore selected
samples, not face membership or stratum weights. It is opt-in.

The detailed comparison reduces grouping to 1.90–1.92 ms. A separate direct
comparison without these internal markers alternates the original and quad
linkers over 40 animated native-4K frames after 12 warmups. Both sides use the
selected-diffuse experiment with environment_samples=0:

| GPU median | Original linker | Quad linker |
| --- | ---: | ---: |
| Primary aggregate | 9.75 ms | 9.46 ms |
| First secondary stage | 2.46 ms | 2.49 ms |
| Total | 21.73 ms | 21.48 ms |

This is a modest 0.25 ms (1.2%) total improvement in this run, not enough to beat
the maintained full-transport path. No production default changes. Replay is
still the largest preparation cost; clear-buffer tuning is a smaller opportunity.

GPU tests cover both linkers, coherent and alternating identities, direct and
hashed domains, budgets 1/8/32, odd extents, invalid hits, empty frames and reuse:
24 cases pass. Selection-energy/specular and reconstruction checks pass seven
GPU cases. Existing capture/replay multi-sample, history and resize regression
checks pass two more, for 33 GPU tests total. Render outputs remain finite and compact identities match; sample-order
changes mean images are not expected to match bitwise. This is not completed
render-convergence or temporal-quality validation.

```sh
PYTHONPATH=../vxl8r/src .venv/bin/python components/OrdinaryLight/scripts/profile_primary_prefixes.py --stage full-selected-diffuse --selected-baseline --quad-visible-links --environment-samples 0 --primary-hit-format identity --width 3840 --height 2160 --warmup 12 --frames 40
```

With `--selected-baseline`, the log's `full` label denotes the original
selected-diffuse linker, **not** maintained full transport. Metadata records this
distinction. Add `--profile-selected-diffuse` only for detailed diagnostic timings.
Logs are in `artifacts/selected-diffuse/diffuse-*-profile.log` and
`diffuse-quad-paired.log`.


## Replay workgroup experiment (2026-09-13)

The compiler accepts `primary_workgroup=(x,y)` for 64-thread power-of-two
compute shapes, defaulting to (8,8). Non-default shapes require non-inline
primary compute. Callers must dispatch the corresponding ceiling-divided tile
extent; the native presenter's dispatch policy is unchanged.

The headless harness exposes `--replay-width 16` (16×4) and
`--replay-width 32` (32×2). Capture remains 8×8. This changes work distribution,
not sample count, BRDF, or reconstruction weights. Wider groups are a memory
locality experiment; these results do not independently measure cache misses,
bandwidth, or register occupancy.

Both comparisons alternate the existing 8×8 selected-diffuse pipeline with the
candidate over 40 animated 3840×2160 frames after 12 warmups. Both sides use the
original linker and environment_samples=0. No internal operation timestamp
markers are enabled.

| Paired run | Primary aggregate, baseline → candidate | Total GPU, baseline → candidate |
| --- | ---: | ---: |
| 16×4 | 10.00 → 9.90 ms | 22.30 → 22.17 ms |
| 32×2 | 9.90 → 9.72 ms | 22.30 → 21.99 ms |

The measured savings are about 0.6% and 1.4%, respectively. They are modest,
workload-specific results and have not been combined with quad linking in a
validated performance run. No viewer or renderer defaults change. Neither
result closes the gap to maintained full transport.

Six GPU regression cases verify exact HDR and complete hit-record parity at
8×8, 16×4 and 32×2, with adjacent/full-frame capture, two samples, history and
odd-sized resize cycles. A separate alternating 160×96 mixed-shape fallback run
checks raw/denoised/final HDR every frame and matches exactly. Compiler tests pass
24 cases and generated primary source remains current.

The first mixed-shape benchmark exposed a diagnostic cache ownership defect:
a newly recorded frame slot could use the other renderer's last-compiled shader
with its own dispatch shape. That run failed eligibility validation and was
discarded. The harness now retains binaries and dispatch shape per executor/
pipeline, then borrows that same bundle for subsequent frame-slot resources.
Only corrected runs (`replay*-paired-valid.log`) are reported above.

```sh
PYTHONPATH=../vxl8r/src .venv/bin/python components/OrdinaryLight/scripts/profile_primary_prefixes.py --stage full-selected-diffuse --selected-baseline --environment-samples 0 --primary-hit-format identity --replay-width 32 --width 3840 --height 2160 --warmup 12 --frames 40
```

Logs and validation results: `artifacts/selected-diffuse/replay*.log`.
Replay still shades and writes per-pixel state; wider dispatch alone does not
remove that work. Further work must measure and reduce those costs rather than
extrapolate frame time from the continuation-ray reduction.


## Direct-light cost investigation (2026-09-13)

The proposed next step of sharing primary direct diffuse lighting was checked
before implementing it. It does not currently show a useful saving in this scene.

The native vxl8r renderer imports `Scene(environment=environment)` with no analytic
lights. The existing selected-diffuse experiment also disables primary environment
sampling. Consequently, analytic-light sharing cannot explain or fix its replay
cost. An analytic-count-zero ablation confirmed no measurable saving (21.93 ms
baseline versus 22.08 ms ablated total); this is an empty-workload control, not an
estimate for scenes containing analytic lights.

A separate `full-environment-cost` diagnostic uses environment_samples=1 and
the maintained continuation sampler. Both sides use the same capture/replay
schedule without face planning. The candidate omits the primary environment
lighting evaluation through a temporary typed OrdinaryShade variant, while
retaining the two random draws per environment sample. Secondary environment
sampling and the continuation algorithm remain enabled. Shader resources and
production artifacts are unchanged.

Animated 3840×2160, 12 warmups, 40 alternating measured frames:

| GPU median | Environment evaluation | Evaluation omitted |
| --- | ---: | ---: |
| Primary aggregate | 7.96 ms | 7.94 ms |
| First secondary stage | 4.12 ms | 4.16 ms |
| Total | 21.31 ms | 21.37 ms |

There is no measurable net saving here. Fused-shader ablation is not an additive
timer or hardware bandwidth/occupancy measurement, and removing work can affect
overlap and compilation. It does not prove environment lighting is intrinsically
free or predict other scenes. It does show that the current evidence does not
justify adding face-shared direct lighting and its shadow-edge approximation.

An initial push-constant environment ablation changed downstream random draws;
it was superseded by the random-sequence-preserving OrdinaryShade variant. The
reported table uses only `primary-environment-preserved-rng.log`.

Ablated HDR is deliberately incorrect (direct lighting is missing), not a usable
render mode or quality-preserving optimization. Outputs are checked finite and
primary identity/validity matches, but HDR parity is disabled and output metadata
marks lighting invalid. Primary environment MIS correctness in the selected-lobe
transport remains unfinished; this diagnostic does not implement that integration.

```sh
PYTHONPATH=../vxl8r/src .venv/bin/python components/OrdinaryLight/scripts/profile_primary_prefixes.py --stage full-environment-cost --environment-samples 1 --primary-hit-format identity --width 3840 --height 2160 --warmup 12 --frames 40
```

No viewer or production lighting behavior changes. The next investigation should
target duplicated per-pixel state and capture/replay memory traffic. Logs:
`artifacts/selected-diffuse/primary-environment-preserved-rng.log` and
`primary-analytic-ablation.log`.

The six existing GPU capture/replay regression cases pass after these diagnostic
changes (exact HDR/hit parity, multiple samples, history and resize).


## Single-dispatch visibility capture (2026-09-13)

Capture previously inherited the lighting queue capacity check despite never
enqueueing a lighting ray. The OrdinaryShade source now applies that check only
outside capture. A full visibility sample plane can therefore be captured in one
dispatch, independently of the lighting tile capacity. Replay retains its queue
bound and tile scheduling. The 112-byte lossless hit record and sampled camera
ray remain unchanged.

The diagnostic `--single-visibility-dispatch` uses this capability for the
selected-diffuse path. At 3840×2160 and a 524,288-entry lighting queue, it replaces
16 capture dispatches and their graph boundaries with one. Both comparison
paths keep the original linker and 8×8 replay shape; no previous optimization
gains are assumed to add together.

40 alternating animated frames after 12 warmups, environment_samples=0:

| GPU median | Tiled capture | Single capture |
| --- | ---: | ---: |
| Primary aggregate | 11.11 ms | 10.80 ms |
| Total | 24.60 ms | 23.89 ms |

The total reduction is 0.71 ms (about 2.9%) in this paired run. Absolute times
differ from earlier sessions; use the within-run comparison. This is not a
hardware bandwidth measurement. The visibility allocation and write volume
remain 928,972,800 bytes (about 886 MiB) per 4K sample plane. This change removes
dispatch/synchronization overhead rather than compressing or eliminating that
payload.

Nine GPU cases verify exact complete-hit and HDR parity across adjacent, tiled
full-frame and single-dispatch capture, at three replay shapes. They include two
samples, history and odd-sized resize cycles, with images larger than the lighting
queue. Animated custom-voxel output is finite and primary identity/validity
matches. All 52 packaged primary shader variants compile byte-identically to the
existing artifacts when capture is disabled.

```sh
PYTHONPATH=../vxl8r/src .venv/bin/python components/OrdinaryLight/scripts/profile_primary_prefixes.py --stage full-selected-diffuse --selected-baseline --environment-samples 0 --primary-hit-format identity --single-visibility-dispatch --width 3840 --height 2160 --warmup 12 --frames 40
```

The diagnostic metadata placement was also corrected so scene analytic-light
counts are recorded after renderer construction. No viewer defaults changed.
The single-dispatch option remains opt-in; reducing the large visibility payload
still requires a separate validated design. Logs:
`artifacts/selected-diffuse/single-capture-*.log`.


## Distance-position cache experiment (2026-09-13)

Replay and application callbacks can consume geometric/shading normals, full
identity and address, all texture-coordinate components, and previous position.
Those fields cannot be discarded generically. The native trace contract does
derive current position XYZ from the sampled ray and committed distance.

The opt-in `primary_visibility_format="distance"` compiler variant stores 25
scalar uint32 words (100 bytes) instead of seven vec4 records (112 bytes):
distance bits followed by geometric normal, shading normal, identity, address,
texcoord and previous position, in that order. Float payloads are bit-cast;
normals and identities are not quantized. Replay reconstructs current XYZ as
`origin + distance * direction`, or zero for a miss. Capture and replay must use
the identical camera, sample, geometry snapshot and arithmetic convention.
Exact reconstruction has been GPU-tested here; this is not a universal
cross-compiler floating-point reproducibility guarantee.

The public `primary_visibility_dtype("distance")` describes the scalar ABI.
`primary_operation(..., visibility_format="distance")` validates its capacity.
vxl8r `VisibleFaceGroups(..., visibility_format="distance")` consumes the smaller
records directly; there is no temporary expanded cache. Shader packing,
unpacking and face-key decoding are all typed OrdinaryShade. Every consumer must
agree on the format. Resource replacement/lifetime rules are unchanged.

### Animated 4K result: less storage, slower rendering

Both paths use selected-diffuse transport with environment_samples=0, the
original linker, tiled capture and 8×8 replay. Twelve warmups, forty alternating
animated frames, 3840×2160:

| Metric | Full record | Distance record |
| --- | ---: | ---: |
| Bytes per pixel | 112 | 100 |
| Cache per sample plane | 885.94 MiB | 791.02 MiB |
| Primary GPU median | 10.13 ms | 11.91 ms |
| Total GPU median | 22.42 ms | 24.09 ms |

Storage falls 10.7% (94.92 MiB per sample plane), but total GPU time increases
about 7.4%. This is **not** a viewer optimization and remains opt-in. The
100-byte scalar stride and access instructions warrant investigation; this run
does not prove which hardware effect causes the slowdown. A vector-friendly
field-plane layout is a better next experiment than assuming fewer bytes must
be faster. Prior single-dispatch/quad/replay-shape gains are not combined here.

Validation includes:
- 18 real capture/replay GPU cases with exact complete hits and HDR, three replay
  shapes, adjacent/tiled/single capture, multiple samples, history and resize.
- 48 GPU grouping cases across both formats, both linkers, coherent/alternating
  identities, direct/hashed domains, three budgets, invalid hits and empty reuse.
- Separate GPU pack/restore dispatches with arbitrary non-position payload bits,
  including high-bit identities and nonfinite float payloads.
- An alternating custom-voxel fallback run with full sampled-hit exports and
  per-frame HDR parity: exact agreement in HDR, position, identity and camera rays.
- 43 compiler/operation/ABI checks and 52 unchanged packaged primary shader
  binaries when this experimental format is disabled.

The active selected-diffuse benchmark has finite HDR and matching compact
identities; linked-list scheduling changes the selected noise samples. It is
not a full image-convergence or optical-material quality validation.

```sh
PYTHONPATH=../vxl8r/src .venv/bin/python components/OrdinaryLight/scripts/profile_primary_prefixes.py --stage full-selected-diffuse --selected-baseline --distance-visibility --environment-samples 0 --primary-hit-format identity --width 3840 --height 2160 --warmup 12 --frames 40
```

Logs: `artifacts/selected-diffuse/distance-*.log`. Viewer defaults are unchanged.


## Vector field-plane visibility (2026-09-13)

The opt-in `primary_visibility_format="planes"` variant preserves the complete
NativeIntersection payload as seven contiguous uvec4 arrays per sample. This
isolates layout from compression: it keeps 112 bytes per pixel and performs no
position reconstruction or quantization. Field order is position/distance,
geometric normal, shading normal, identity, address, texcoord, previous position.

For P = width × height, each vector is at
`(sample * 7 + field) * P + pixel`. Total storage is samples × P × 112 bytes.
A diagnostic CPU view is uint32 shaped `(samples, 7, height, width, 4)`; this is
not the per-pixel record layout returned by `primary_visibility_dtype`.
Float fields retain their original bits and are interpreted as float32 on load.

Capture and replay use typed OrdinaryShade store/load helpers.
`primary_operation(..., visibility_format="planes")` validates capacity;
vxl8r `VisibleFaceGroups(..., visibility_format="planes")` reads only the identity
and address planes. No temporary unpacked cache is allocated. Producers,
consumers and shader variants must agree on the format and retain the same
resource lifetime and snapshot ordering as before.

### Measured layout and combined results

Animated 3840×2160, environment_samples=0, twelve warmups and forty alternating
measured frames. The isolated comparison keeps tiled capture, original linking,
8×8 replay and selected-diffuse transport on both sides:

| GPU median | Original record layout | Field planes |
| --- | ---: | ---: |
| Primary aggregate | 10.35 ms | 9.95 ms |
| Total | 22.91 ms | 22.32 ms |

The planar layout saves about 0.59 ms (2.6%) in this paired run, with unchanged
885.94 MiB storage per sample plane. It avoids the prior scalar-packed cache's
observed slowdown; fewer bytes alone were not a useful predictor.

A separate combined test enables planes, single-dispatch capture, quad linking
and 32×2 replay. It compares against maintained full transport in the same run:

| GPU median | Maintained transport | Combined experiment |
| --- | ---: | ---: |
| Primary aggregate | 4.75 ms | 8.59 ms |
| Total | 17.43 ms | 20.73 ms |

The combined experiment remains about 19% slower. Gains from separate sessions
must not be added or treated as a claim of viewer improvement. The experiment
also still requires separate estimator/quality validation before adoption.
Viewer defaults are unchanged.

Validation:
- Nine planar capture/replay GPU cases verify exact HDR and complete hits across
  three replay shapes, adjacent/tiled/single capture, samples, history and resize.
- 72 grouping GPU cases cover all three layouts and both linkers with membership,
  weights, invalid hits, odd dimensions, sample planes and empty-frame reuse.
- Independent GPU pack/unpack dispatches preserve every payload bit, including
  arbitrary float bit patterns and high-bit identities, across three sample planes.
- Custom-voxel fallback checks compare raw/denoised/final HDR each frame and
  sampled camera rays, identities and full positions.
- 45 compiler/API tests pass. All 52 default primary SPIR-V variants remain
  byte-identical to the existing packaged artifacts.

```sh
# Isolated layout comparison:
PYTHONPATH=../vxl8r/src .venv/bin/python components/OrdinaryLight/scripts/profile_primary_prefixes.py --stage full-selected-diffuse --selected-baseline --planar-visibility --environment-samples 0 --primary-hit-format identity --width 3840 --height 2160 --warmup 12 --frames 40

# Combined options versus maintained transport:
PYTHONPATH=../vxl8r/src .venv/bin/python components/OrdinaryLight/scripts/profile_primary_prefixes.py --stage full-selected-diffuse --planar-visibility --single-visibility-dispatch --quad-visible-links --replay-width 32 --environment-samples 0 --primary-hit-format identity --width 3840 --height 2160 --warmup 12 --frames 40
```

Logs: `artifacts/selected-diffuse/planes-*.log`. This is a cache-layout
improvement, not a completed faster face-budgeted renderer.


## Compact vector planes: memory saving, no measured speedup

The experimental `distance_planes` format preserves six complete vec4 payload
planes and stores camera-hit distance in a packed scalar tail. Replay rebuilds
position XYZ from the same sampled ray. This avoids scalarizing the six other
fields, as the earlier 100-byte record experiment did. All shader logic remains
in typed OrdinaryShade. Voxel grouping reads the identity/address planes directly.

Each sample occupies `16 * (6 * pixels + ceil(pixels / 4))` bytes. Tail padding is
per sample, not just per allocation. Allocate with the public
`primary_visibility_byte_size(width, height, samples=..., format=...)`; the
compiler, primary operation and grouping consumer must agree on the format.
At 3840×2160 this uses 791.02 MiB rather than 885.94 MiB per sample, saving
94.92 MiB (10.7%). No payload other than redundant camera-hit XYZ is omitted.

An animated, perspective, native 4K paired run compared **full vector planes
against compact vector planes**, with both sides using selected diffuse transport,
8×8 replay, tiled capture, ordinary linking, zero environment samples, and compact
primary-hit exports. There were 12 warmup frames and 40 measured frames per side.

| GPU median | Full vector planes | Compact vector planes |
| --- | ---: | ---: |
| Primary aggregate | 9.93 ms | 10.06 ms |
| Total | 22.55 ms | 22.82 ms |

The compact cache was about 1.2% slower in this run. This small difference is not
a claim of a stable regression, but there is no evidence for a throughput win.
Retain full vector planes as the faster measured experimental layout; compact
planes remain an opt-in memory tradeoff. This does not change viewer defaults or
make face-budgeted transport faster than the maintained renderer.

Validation:
- Nine compact-plane capture/replay GPU cases pass across workgroup shapes,
  odd extents, multiple samples, resize/history and capture dispatch variants.
- 96 grouping cases cover all four cache layouts, both linkers, sample selection,
  invalid identities, empty frames and direct/hash face tables.
- An independent GPU pack/unpack check preserves arbitrary payload bits and
  reconstructed XYZ across three odd-sized samples; unused distance-tail lanes
  retain their sentinel, checking scalar component writes and sample alignment.
- Custom voxel fallback rendering matches raw, denoised and final HDR exactly,
  with identical sampled primary hits and camera rays. The active selected-diffuse
  benchmark checks finite output and identity validity; its nondeterministic
  grouping/sample selection is not an image-parity test.
- 58 compiler/API checks pass, including per-sample padded capacity validation.
  Generated primary source is current and all 52 default primary SPIR-V variants
  remain byte-identical to packaged shaders.

```sh
PYTHONPATH=../vxl8r/src .venv/bin/python components/OrdinaryLight/scripts/profile_primary_prefixes.py --stage full-selected-diffuse --selected-baseline --baseline-visibility-format planes --distance-planar-visibility --environment-samples 0 --primary-hit-format identity --width 3840 --height 2160 --warmup 12 --frames 40
```

Evidence: `artifacts/selected-diffuse/distance-planes-*.log`.
Further speed work should target the extra capture/replay and per-pixel primary
processing cost rather than assuming a smaller visibility allocation is faster.


## Capture dependency precision and replay cost

A serialized operation diagnostic of the combined full-vector-plane experiment
(single capture, quad linking, 32×2 replay) localizes the remaining primary cost.
The two reported frame-slot snapshots at animated perspective 4K measured:

| Operation | Snapshot range |
| --- | ---: |
| Capture | 1.53–1.63 ms |
| Group | 1.25–1.28 ms |
| Plan | 0.76–0.79 ms |
| Selection mask | 0.07 ms |
| Replay, summed tiles | 4.99–5.00 ms |
| Reconstruction | 0.57 ms |

These are the last query snapshots, **not per-stage medians**. Serialized
per-operation timestamps perturb scheduling; use the separate uninstrumented
paired run below for throughput. Replay remains the largest part of this path.

The primary operation previously declared capture writes to paths, queues,
medium state, reservoirs, guides and hit exports, although capture returns before
accessing these outputs. Capture now omits those untouched native resources from
its uses. Descriptor kind, runtime and lifetime validation remains in place.
Camera/scene inputs and counters retain conservative access; every custom
geometry/material use remains authoritative, including aliases of omitted native
outputs. Replay and fused declarations are unchanged. No shader or lighting
estimator changes were required.

The diagnostic `--compare-capture-dependencies` reproduces the original capture
uses on its baseline, with the **same capture shader**, and matches both sides'
cache layout, capture dispatch, linker and replay shape. A 12-warmup/40-frame
animated perspective native 4K comparison without per-operation timestamps found:

| GPU median | Original uses | Precise capture uses |
| --- | ---: | ---: |
| Primary aggregate | 8.536 ms | 8.530 ms |
| Total | 20.681 ms | 20.441 ms |

The total difference is about 1.2%, while primary time is effectively unchanged.
Do not attribute the whole difference to synchronization or claim a reliable
throughput gain from this single run. The accurate capture contract is retained;
it is not a solution to the remaining full-resolution replay cost. Viewer settings
are unchanged. Both sides here use experimental selected diffuse transport;
this comparison is not evidence that it beats maintained full transport.

Validation: 62 compiler/API checks pass, including retained custom-resource aliases,
profiling counters and rejected foreign-runtime unused descriptors. All 36 triangle
capture/replay GPU cases pass across four layouts, three replay shapes and three
capture patterns, including multiple samples, resize and history. Custom voxel
fallback rendering matches raw/upstream/output HDR exactly and preserves full
sampled hits and camera rays. Active selected-diffuse output remains finite with
matching identity validity; stochastic sample selection is not an HDR-parity test.

```sh
# Serialized operation diagnosis (snapshot timings only):
PYTHONPATH=../vxl8r/src .venv/bin/python components/OrdinaryLight/scripts/profile_primary_prefixes.py --stage full-selected-diffuse --planar-visibility --single-visibility-dispatch --quad-visible-links --replay-width 32 --profile-selected-diffuse --environment-samples 0 --primary-hit-format identity --width 3840 --height 2160 --warmup 8 --frames 12

# Matched uninstrumented dependency comparison:
PYTHONPATH=../vxl8r/src .venv/bin/python components/OrdinaryLight/scripts/profile_primary_prefixes.py --stage full-selected-diffuse --selected-baseline --compare-capture-dependencies --planar-visibility --single-visibility-dispatch --quad-visible-links --replay-width 32 --environment-samples 0 --primary-hit-format identity --width 3840 --height 2160 --warmup 12 --frames 40
```

Evidence: `artifacts/selected-diffuse/capture-replay-profile.log` and
`artifacts/selected-diffuse/capture-uses-*.log`. The next architectural target is
the per-pixel replay path's transport-state setup and memory traffic: most diffuse
continuations are rejected, yet all pixels still traverse primary replay to produce
their local outputs. Direct/specular lighting and denoiser guides must remain intact
if that work is split from selected continuation setup.


## Deferred secondary initialization in selected replay: no consistent total win

Rejected diffuse continuations still produce local radiance and denoiser signals.
Dropping their secondary records outright would lose those outputs. This test
instead reuses the typed OrdinaryShade deferred-initialization variant from
`primary_capture_store_experiment.py`: omit the initial eight-field clear, write
a complete secondary record at normal completion, and explicitly clear the record
on earlier exits or uncaptured completion paths. The existing record layout,
continuation decision, RNG consumption and lighting estimator are preserved.

`--selected-deferred-capture` applies this variant only to the diagnostic candidate
and matches both renderers' full vector planes, single capture, quad linking and
32×2 replay. It does not modify packaged shaders, production recording, or viewer
defaults. The test reuses an existing weak candidate from full transport because
the selected path rejects most diffuse continuations and has a different workload.

Two alternating, matched animated perspective native 4K comparisons, with normal
face selection/reconstruction and no per-frame readbacks during timing, measured:

| Run | Original primary | Deferred primary | Original total | Deferred total |
| --- | ---: | ---: | ---: | ---: |
| 12 warmups, 40 frames | 9.379 ms | 8.747 ms | 22.173 ms | 21.684 ms |
| 16 warmups, 80 frames | 8.706 ms | 8.582 ms | 21.171 ms | 21.291 ms |

The first total improved 2.2%; the longer repeat worsened 0.6%. Primary time fell
in both, but the size varied considerably and the end-to-end result did not hold.
Keep this as a diagnostic experiment, not a production optimization or viewer
recommendation. Total medians are not sums of component medians. Both sides are
selected diffuse transport; these numbers do not establish a win over maintained
full transport.

### Deterministic transport checks

`--selected-fixed-mask none|all` fills the existing mask on the GPU and bypasses
stochastic face reconstruction on both sides. It requires a selected baseline and
per-frame checks; it is not an estimator or throughput comparison. All other
transport, guide production and denoising still run. Matching continuation masks
alone was insufficient when reconstruction selected different representative
pixels from each renderer's independently ordered face lists.

The original all-mask check with reconstruction failed (raw max difference about
0.137). An **identical-shader control** with reconstruction also failed (raw max
about 0.128), confirming that this setup cannot establish shader parity. These
failed checks are retained as evidence. Bypassing reconstruction isolates the
transport comparison; the all-mask test then matches raw, denoised and final HDR
exactly, including full primary hits and sampled camera rays, at odd 161×97 extent.
The rejected-diffuse check also verifies the branch that terminates most paths.

Refraction with environment lighting and local history, resetting every third
frame, matches raw/denoised/final HDR and sampled hits/rays exactly. Both renderers
report five valid-history and three reset frames. The normal 4K comparisons check
finite output and compact identity validity; stochastic reconstruction is not an
image-parity test. This coverage does not establish every raw secondary-record
bit or all sparse-capture/alternate-strategy contracts required for promotion.

```sh
# Matched throughput comparison; repeat with --warmup 16 --frames 80:
PYTHONPATH=../vxl8r/src .venv/bin/python components/OrdinaryLight/scripts/profile_primary_prefixes.py --stage full-selected-diffuse --selected-baseline --selected-deferred-capture --planar-visibility --single-visibility-dispatch --quad-visible-links --replay-width 32 --environment-samples 0 --primary-hit-format identity --width 3840 --height 2160 --warmup 12 --frames 40

# Deterministic active-transport parity; repeat with --selected-fixed-mask none:
PYTHONPATH=../vxl8r/src .venv/bin/python components/OrdinaryLight/scripts/profile_primary_prefixes.py --stage full-selected-diffuse --selected-baseline --selected-deferred-capture --selected-fixed-mask all --planar-visibility --single-visibility-dispatch --quad-visible-links --replay-width 32 --environment-samples 0 --primary-hit-format full --check-each-frame --width 161 --height 97 --warmup 2 --frames 5
```

Evidence: `artifacts/selected-diffuse/selected-init-*.log`; the identical-shader
negative-control launcher is stored alongside the logs. Production initialization
remains unchanged. A larger reduction would need to separate local lighting/guide
output from continuation-only state, with matching resolve and denoiser consumers,
rather than simply omitting records for rejected paths.


## Local-only secondary state: correct prototype, slower rendering

`--selected-local-state` tests a private scratch representation for inactive,
selected-diffuse paths with zero PDF and a diffuse lobe. These paths still need
local radiance, primary specular radiance, position/roughness and geometry identity.
They do not need a secondary hit, normal/PDF, throughput or indirect lobe fraction.
A negative `position_valid.w` marks their local-only record. Matched typed
OrdinaryShade preparation supplies the known canonical continuation fields in
registers. Continuing, specular and fallback paths keep full records.

The prototype writes five vec4 fields (80 bytes) instead of eight (128 bytes) for
qualifying records. It retains the 128-byte allocation stride and does **not**
reduce allocation size or compact the active-ray workload. Both sides use the
previous deferred complete-record initialization, isolating the local-only change.
No production shader or public record ABI is modified.

The initial version retains fields at their original offsets. With
`--selected-local-packed`, the five vectors instead occupy the first 80 bytes;
resolve and preparation decode them into canonical local values. This exposed a
native scratch alias: absent reservoir/seed buffers alias live secondary state.
See [the confirmed upstream finding](resolve_scratch_alias.md). Packed tests give
both sides separate persistent scratch buffers, so their baseline differs from
the original scattered test. The workaround is diagnostic only.

Animated perspective native 4K, 12 warmups and 40 alternating measured frames,
full vector visibility planes, single capture, quad linking, 32×2 replay, one
sample, four bounces, environment samples zero, normal face selection and
reconstruction, no per-frame readbacks during timing:

| Layout experiment | Baseline primary | Candidate primary | Baseline total | Candidate total |
| --- | ---: | ---: | ---: | ---: |
| Scattered local fields | 8.282 ms | 10.268 ms | 20.032 ms | 22.189 ms |
| Contiguous local prefix, separate resolve scratch on both sides | 8.382 ms | 9.790 ms | 20.426 ms | 21.960 ms |

Both regress substantially: about 10.8% and 7.5% total respectively. Fewer logical
stores did not improve throughput. Branch divergence, memory transactions and
compiler scheduling are possible explanations, not established hardware-counter
findings. Do not compare absolute values across sessions as an isolated layout
speedup. Neither prototype is adopted, and viewer defaults remain unchanged.

Validation uses deterministic all/none continuation masks before stochastic face
reconstruction. Final packed runs compare every channel of normal/roughness,
motion, view depth, diffuse and specular images exactly on every checked frame.
Both active and rejected paths pass at odd 161×97 extent. A refractive fallback
with environment lighting and alternating local-history validity/reset also
passes. Raw, denoised and final HDR, sampled camera rays and full exported primary
hits match exactly. All valid runs exit cleanly. The 4K throughput tests check
finite output and identity validity, not stochastic HDR equality.

The original packed history failure and the first separate-scratch run's teardown
failure are retained in the logs; only `*-valid.log` packed runs are final evidence.
The latter was diagnostic ownership ordering: retained kernels must close before
their scratch-buffer contexts. Generated production primary source remains current.

```sh
# Scattered local-only state versus complete deferred state:
PYTHONPATH=../vxl8r/src .venv/bin/python components/OrdinaryLight/scripts/profile_primary_prefixes.py --stage full-selected-diffuse --selected-baseline --selected-local-state --planar-visibility --single-visibility-dispatch --quad-visible-links --replay-width 32 --environment-samples 0 --primary-hit-format identity --width 3840 --height 2160 --warmup 12 --frames 40

# Contiguous local prefix and separate resolve scratch: add --selected-local-packed.
# Deterministic guide/transport parity: add --selected-fixed-mask none (or all)
# --check-each-frame, use --primary-hit-format full and --width 161 --height 97.
```

Evidence: `artifacts/selected-diffuse/local-state-*.log` and
`artifacts/selected-diffuse/local-packed-*.log`. A true compact continuation dispatch
would require a different scheduling/storage design; this experiment only tests
conditional record contents within the existing per-pixel replay dispatch.


## Upstream resolve alias fixed

The [native scratch alias](resolve_scratch_alias.md) discovered by the local-state
experiment is now fixed in OrdinaryLight. Reservoir seeding is a separate resolve
policy, enabled natively only for indirect reuse. Sampled-denoiser-only resolve is
HDR-only; legacy denoiser signals still resolve without touching reservoir/seed
placeholders. Active seeding rejects aliased storage. No extra buffers, waits or
per-frame allocations were added.

The packed-state diagnostic's separate-scratch workaround has been removed. Its
native-placeholder reproduction now passes exact signal/guide, HDR and sampled-hit
checks under animation and history reuse. Real native indirect reuse, multi-sample
resolve and resize also pass. This correctness fix does not promote either slower
local-state prototype or change viewer settings. See `resolve-alias-*.log` in the
selected-diffuse artifact directory for the current validation results.

### Compact continuation scheduling

The [compact continuation experiment](compact_continuation_experiment.md) adds
a GPU-only tile list and indirect resume dispatch. Animated 4K total GPU time
regressed from 21.52 to 23.70 ms against matched selected replay. Correctness
checks passed, but repeated primary setup outweighs the scheduling benefit in
this implementation. The path remains diagnostic-only and disabled by default.

## Return to maintained baseline

Fresh animated perspective 3840x2160 control run, 12 warmup and 40 measured
frames, 4 bounces, identity hits, environment samples 0, all selected-diffuse
flags disabled. Matched maintained runs measured 17.66 / 17.59 ms median total
GPU time. This confirms the previous sub-20 ms baseline under these settings;
it does not include CPU/presentation latency. Evidence:
`artifacts/selected-diffuse/maintained-return-4k.log`.
