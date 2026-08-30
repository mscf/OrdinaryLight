# Renderer roadmap status

## Backend parity and fallback contract

- Global illumination and rasterization consume the same public scene, camera,
  material, lighting, animation, interaction, and output abstractions whenever
  the rendering techniques can represent the same semantics.
- Vulkan defaults to ``backend_preference="auto"``: hardware ray-query GI is
  preferred, while native Vulkan rasterization is the compatibility fallback
  when no matching ray-query adapter exists. Explicit ``"gi"`` and
  ``"raster"`` requests never silently switch renderer classes.
- Capability discovery records the requested backend, selected backend,
  fallback status, and reason. Initialization or shader failures are not
  disguised as compatibility fallbacks.
- Cross-backend visual gates cover shared raster semantics. Expected physical
  differences—indirect illumination, traced reflection/refraction, and
  multiple scattering—remain GI capabilities rather than raster regressions.
- Raster remains an independently supported renderer, not only a degraded GI
  mode. Vulkan/WebGPU raster parity and raster/GI semantic parity are both
  maintained as native execution paths evolve.
- Python Ordinary Shade definitions are the authoritative built-in raster
  source. Wheel CI compiles, validates, checksums, and packages SPIR-V/WGSL;
  installed raster fallback therefore has no runtime compiler dependency.
  Edited source checkouts deliberately recompile rather than loading artifacts
  whose recorded source hash is stale.

## Implemented foundation

- Backend-neutral vertex/fragment programs authored with Ordinary Shade,
  including location/builtin reflection and cross-stage validation.
- Verified offscreen scene raster paths for native Vulkan/SPIR-V and
  WebGPU/WGSL, sharing meshes, instances, transforms, cameras, base materials,
  textures, analytic lights, shadows, depth state, and named geometry products.
- Portable screen-space raster optics use an explicit opaque color/depth
  prepass and a second reflection/refraction composite on Vulkan and WebGPU,
  with roughness-aware environment/probe fallback. The original environment
  path remains the default low-cost tier.
- A portable raster render graph, static temporal/post processing, volume
  slicing, hybrid backend composition, and backend-neutral surface output.
  Native MRT/shadow-map/volume compute paths and direct swapchain/external-image
  output remain performance work rather than missing public semantics.

- Semantic public namespaces for loaders, cameras, lights, animations,
  materials, outputs, integrations, capabilities, and backends.
- A backend-neutral `Renderer`, named render products, structured statistics,
  caller-owned NumPy outputs, and explicit capability discovery.
- Perspective, orthographic, and panoramic cameras plus point, directional,
  spot, environment, emissive-triangle, and area-light workflows.
- Stable scene resources, instancing, mutation, morph targets, skinning, node
  animation, and deterministic animation playback.
- glTF/GLB ingestion with sparse and normalized accessors, scene transforms,
  instancing, morph targets, skins, animation, texture transforms,
  transmission, emissive strength, and unlit materials.
- Python-authored material programs and custom vertex attributes, including a
  deterministic staged wavefront path.
- General-purpose points, lines, glyphs, textures, volumes, overlapping media,
  transfer functions, and multi-volume composition.
- Headless linear-HDR NumPy rendering, SDR conversion, and optional streamed
  FFmpeg video encoding.
- Vulkan ray-query acceleration and a deterministic CPU reference backend.
- A script-extensible Qt workbench with worker-owned Vulkan presentation,
  background scene loading, and one-frame backpressure. Individual interactive
  showcase programs have been replaced by reusable scene builders and
  declarative workbench catalog scripts; automation gates remain standalone.
- Ordered non-blocking `RenderJob` submissions for headless/notebook/GUI use.
- Runtime-checkable backend protocols with `render_frame()` as the canonical
  entry point. Vulkan is isolated behind the optional `vulkan` extra, while
  the core package and CPU reference backend remain independently usable.
- Consumer-level wheel verification covering HDR arrays, named products,
  asynchronous jobs, and glTF loading from a clean downstream environment.
- A manifest-driven, deterministic shader inventory with checks for missing
  sources, missing binaries, and unmanaged compiled outputs.
- An incremental Ordinary Shade migration path: built-in effect color helpers
  and the renderer's shared ACES color transform are authored as typed Python,
  generated deterministically, and consumed by checked-in GLSL/SPIR-V artifacts.
  The indirect-reservoir clear, indirect-dispatch preparation, path resolve,
  packed/storage-image tone-map, NV12, HDR-to-P010 conversion, à-trous
  denoising, final display/overlay, primary-ray generation, ray intersection,
  subgroup-bucketed intersection, and packed indirect-reservoir probe passes
  plus HDR path accumulation and indirect-reservoir seeding are complete
  generated compute stages. Temporal/spatial indirect-candidate validation and
  merging are generated as well, covering
  composable typed helpers, scalar and structured runtime storage arrays, push
  constants, fixed structured storage records, vector swizzles, and write-only
  output, image dimensions, dynamically bounded loops, helper dependency graphs,
  integer packing, image ping-pong, math intrinsics, short-circuit logic,
  conditional expressions, loop continuation/termination, heterogeneous
  storage blocks with runtime structure arrays, portable projection math,
  typed Vulkan acceleration structures, ray-query control flow, declared
  subgroup ballot capabilities, and atomic queue compaction while retaining
  their Vulkan resource ABI. Structured and `void` helper functions can share
  descriptor-backed buffers, enabling typed half/unorm/RGB9E5 reservoir
  serialization, explicit storage barriers, and reuse of that ABI in a
  production accumulation stage. Typed integer storage images, structured
  multi-value helper results, reprojection, visibility ray queries, and atomic
  profiling complete the candidate pass without changing its Vulkan ABI.
  The generated debug/apply pass completes this shader group with typed
  diagnostic colors and material-aware filtered correction results. The full
  reconstruction stage is generated as well, including reprojection, history
  validation, diffuse filtering, composed effects, and ABI-compatible RGBA and
  formatless BGRA storage-image-array variants. The baseline buffer-output ray
  query is generated with its custom-material specialization contract intact.
  Its accumulated image-output counterpart now covers direct-light MIS,
  primary metadata, adaptive sampling, history reprojection, moments, and
  neighborhood clamping with the same descriptor and push-constant ABI.
  The final wavefront block is complete: its typed GGX/PBR sampling graph and
  backend-neutral packed-texture sampling graph lower as independently tested
  Ordinary Shade modules. Typed hit loading covers queued and fused ray
  queries; miss MIS, bounce termination, nested-medium transmission, Russian
  roulette, ray-cone transport, and continuation-ray construction cover the
  path-state transition graph integrated by the shade entry point.
  Subgroup-compacted continuation enqueue and overflow termination are typed as
  well. Point, directional, and spot-light preparation, visibility, and PBR
  contribution form the first direct-light module, with the fused-primary path
  consuming the same typed direction, attenuation, cone, cosine, shadow-
  distance, incident-radiance, and contribution helpers. Buffer iteration and
  ray-query visibility remain Vulkan orchestration, while volume transmittance
  is an explicit compositional input rather than a hidden renderer dependency.
  Fused-primary triangle-area sampling and reservoir-candidate evaluation now
  use those same typed barycentric, emitter-cosine, solid-angle PDF,
  power-heuristic MIS, and contribution primitives. Shadow visibility remains
  thin Vulkan orchestration. Reciprocal emissive-hit MIS and digitally shifted secondary
  NEE scheduling are typed too; the latter added portable bit reversal to
  Ordinary Shade's GLSL/WGSL intrinsic set. Analytic and texture-backed
  environment discovery, spherical UV mapping, HDR decoding, cosine sampling,
  visibility, MIS, octahedral candidate directions, unified area/environment
  selection, NEE probability compensation, and opaque BSDF scattering complete
  the non-volume lighting and continuation graph. Portable `atan2`, `acos`, and
  fractional intrinsics support the environment mapping in both targets. The
  fused-primary wrapper now consumes those typed environment mapping, PDF, MIS,
  candidate-direction, and contribution primitives directly, retaining only
  texture-resource access and ray-query visibility as Vulkan orchestration. The
  unified primary/secondary scheduler is now consumed directly too, including
  square-root domain weighting, sample-budget weighting, candidate luminance,
  and the digitally shifted secondary-NEE sequence. RNG mutation and
  resource-dependent candidate evaluation remain explicit orchestration. The
  fused path also consumes typed reciprocal environment-miss and emissive-hit
  MIS. Triangle geometry, sidedness, selection density, previous-path PDF, and
  unified-domain probability are explicit inputs, followed by a typed emission
  contribution rather than entry-point-local radiometric arithmetic. The
  fused secondary loop now consumes typed opaque throughput, compensated direct
  contribution, cone growth, continuation metadata, and Russian-roulette
  survival/renormalization helpers. Random draws and queue ownership remain
  explicit orchestration, keeping the transport policy independently reusable.
  Secondary transmission consumes typed target-medium selection, relative-IOR
  refraction, total-internal-reflection handling, stack-depth transitions, and
  throughput tinting. Only indexed medium-stack reads/writes remain in GLSL.
  Secondary surface preparation now directly consumes the shared typed hit,
  barycentric, ray-cone, normal, UV-footprint, and tangent helpers. Vertex and
  descriptor fetches plus custom-material dispatch remain explicit resource and
  specialization boundaries.
  Stop-bounce selection, throughput termination, environment-miss accumulation,
  maximum-bounce termination, and indirect-capture eligibility are typed as
  secondary control policy. Volume integration and capture-buffer stores remain
  resource orchestration. Custom-material insertion now targets an explicit
  semantic marker instead of depending on neighboring implementation text.
  SER hint construction, NEE probability and sample-count normalization,
  contribution averaging, emissive visibility, and capture payload construction
  complete the fused-loop policy migration. A structural boundary gate now
  requires these decisions to remain generated while preserving ray queries,
  indexed storage, volume invocation, profiling, reorder intrinsics, and queue
  mutation as explicit Vulkan orchestration. The completed boundary is
  bit-exact across every supported execution strategy on the feature-parity
  scene.
  The first backend-orchestration slices are generated as well: TLAS queries
  and intersection metadata, SER invocation, subgroup queue reservation,
  continuation payload/overflow writes, and dispatch swizzling. Ordinary Shade
  now exposes portable scheduling/barrier constructs plus Vulkan-only reorder,
  typed external ABI values, opaque types, and mutable parameters. The remaining
  fused body to translate has narrowed further: vertex/attribute/material
  descriptor reads, fixed-array medium-stack access, path-state stores,
  secondary-capture stores, and continuation queue writes now cross typed
  generated ABI functions. What remains is the enclosing fused control loop,
  volume-stage invocation, optional profiling calls, and persistent-coarse
  entry-point scheduling.
  The volume migration now covers the production header ABI, world/local interval
  math, transfer lookup, phase and extinction math, corrected-alpha
  compositing, bounded emission/absorption marching, proxy-geometry traversal,
  path application, shadow transmittance, brick-based empty-space skipping,
  and the extinction/source-composition primitives used by overlapping media.
  Opaque visibility, approximate volume transmittance, point/area/environment
  scattering, and bounded multiple-scattering orders are now integrated into
  the typed single-volume marcher. Fixed-size function-local arrays were added
  to Ordinary Shade for both GLSL and WGSL, allowing the complete overlapping-
  volume interval, empty-space, scattering, and extinction marcher to lower
  without changing its production algorithm. Ordinary Shade exposes an
  explicitly Vulkan-only reflected `sampler3D[]` resource for the production
  scalar textures, while rejecting that combined-sampler model for WGSL. Its
  Vulkan resource model now also covers non-uniform sampled `sampler2D[]`
  descriptors, explicit LOD/size/level queries, fixed local arrays, and typed
  work-counter atomics. These close the native-texture and profiling gaps in
  the shade-stage migration. A typed material-evaluation application seam
  preserves generated user material dispatch independently of the production
  entry point. The complete production entry now
  links the typed hit, volume, surface, material, emission, direct-light,
  BSDF/transmission, roulette, and subgroup-enqueue graphs and is checked as
  real Vulkan SPIR-V. Compile-time grafts cover native/packed textures,
  work profiling, overlapping volumes, empty-space skipping, volume
  scattering, and bounded multiple scattering; representative individual and
  all-features combinations compile in the automated gate without altering
  disabled descriptor layouts. The default
  `wavefront_ordinaryshade_shade=True` path is within strict HDR numerical
  tolerance of the handwritten stage across the formal area-light, diffuse, glossy/glass,
  textured, small-emitter, occlusion, nested-glass, and dense matrix, as well
  as single, overlapping, scattering, and bounded multiple-scattering volume
  captures. Native-texture parity is also exact. It is now the production
  default, with the handwritten stage retained as an explicit fallback.
  Representative 4K wavefront A/B measured 35.54 FPS generated versus 35.45
  FPS handwritten with matching GPU time. Generated
  secondary shading now accepts deterministic Python `MaterialEvaluation`
  specializations and custom vertex attributes through the existing material
  API; the dedicated material parity gate is within the same strict HDR
  tolerance. At 1080p,
  median generated shade dispatches matched or slightly beat the handwritten
  stage (about 3.06 ms summed across secondary bounces versus 3.13 ms), but
  cold command recording/pipeline creation took about 3.15 seconds versus
  roughly 36 ms. A device-keyed persistent Vulkan pipeline cache now preserves
  compiled pipelines across renderer processes: a controlled two-renderer gate
  fell from roughly 7 seconds on its first cache-building process to 1.42
  seconds on the next process while retaining bit-exact output. The first-ever
  compile after a driver/GPU cache miss remains a module-size optimization
  target, but is amortized by the persistent cache; steady-state shading and
  HDR correctness have reached parity. Lazy auto-strategy creation additionally
  reduced a
  genuine cold-cache first-frame record from 58.6 seconds to 7.62 seconds and
  safely replaces the executor on scene-driven strategy changes without
  replacing the Vulkan device or external image pool.
  Python-authored `SurfaceResponse` programs now share the staged material ABI
  across the fused primary and generated secondary shade stages. Arbitrary
  absorb, diffuse, reflection, and transmission events retain stochastic RNG
  inputs and nested-medium IOR tracking; the dedicated surface parity gate
  measures approximately `4.8e-7` relative RMSE against the handwritten
  fallback on a feature-parity capture with 93% primary-hit coverage.
  Fused-primary migration is underway through shared generated camera-ray and
  initial path-state helpers used by every execution strategy. Perspective,
  orthographic, and panoramic construction, deterministic RNG jitter, path
  identity/flag packing, barycentric hit reconstruction, surface normals and
  orientation, ray-cone spread, and surface classification retain compile-time
  handwritten fallbacks. UV/tangent interpolation, UV-density footprints,
  triangle-tangent generation, and mapped-normal correction are generated too,
  followed by sampled material-channel application and tangent-space normal
  transformation. Thin resource wrappers retain packed/native sampling,
  profiling, and descriptor ownership. Typed G-buffer payload preparation,
  sided emission, maximum-bounce termination, and active-state clearing cover
  the primary output boundary while retaining explicit image/queue operations.
  Initial transmission now covers opaque/transmissive classification, target
  IOR clamping, refraction and total-internal-reflection fallback, first medium
  entry/depth, stack initialization, and throughput tinting.
  The post-BSDF continuation boundary now includes throughput application,
  ray-cone growth, direction normalization, path metadata/PDF packing,
  secondary-capture eligibility, ray offsets, and queue payload construction.
  PBR sampling remains a separate module and is now typed end to end: cosine
  and GGX sampling, Fresnel/distribution/masking, mixed PDFs, BRDF evaluation,
  reflection, occlusion weighting, and invalid-sample fallback retain the exact
  RNG consumption contract through a thin wrapper. It agrees with the existing
  generated secondary graph within `4.9e-7` relative HDR RMSE. Direct-light
  estimator policy remains the next separate module.
  All wavefront, hybrid, megakernel, persistent, and SER variants compile and
  the textured real-geometry execution matrix remains HDR-identical across
  strategies, including the one-bounce termination path; the native-texture
  comparison is within `3.5e-8` relative HDR RMSE. Sixteen-bounce feature and
  nested-glass execution matrices remain exactly HDR-identical.
  Vulkan-side secondary orchestration is typed as well: ray queries,
  descriptor reads, medium/path storage, volume and profiling callbacks,
  subgroup queue reservation, continuation writes, custom-material dispatch,
  group swizzling, and the persistent-coarse shared-memory scheduler are
  generated from Ordinary Shade. The enclosing fused transport loop is now
  generated too, with path, medium, RNG, and ray-cone state threaded through a
  typed per-bounce Vulkan callback. Handwritten branches remain as explicit
  diagnostic fallbacks and backend ABI adapters rather than portable shader
  policy.
- Focused examples for headless, asynchronous, video, custom-material, scene-
  update, and workbench workflows, backed by a concise getting-started guide.

## Deferred, measured work

- Add a Vulkan headless staging ring so multiple readback jobs can overlap GPU
  execution, synchronization, and CPU consumption. The current public async
  contract is stable, while backend execution is deliberately serialized.
- Move animation deformation and high-frequency scene updates from CPU array
  rebuilding toward GPU skinning/morphing and asynchronous acceleration-
  structure updates where measurements justify the complexity.

- Revisit interactive path-tracing noise and reconstruction quality. Current
  behavior is accepted for development, but quality remains equal in priority
  to performance. A tracked multi-scene HDR baseline now prevents silent
  regressions in accuracy, temporal stability, structured noise, fireflies,
  banding, and edge-detail preservation while this work proceeds.
- Redesign volume empty-space skipping before enabling it by default. The
  current sparse-volume gate is image-exact but measured 0.93x rather than a
  speedup, so the feature remains opt-in and experimental.
- Continue the 4K performance program without lowering image quality. The
  persistent target remains at least 50 FPS; environmental exceptions require
  an explicit recorded reason.

## Natural extensions

- Additional file formats belong in `ordinarylight.loaders.<format>` and register
  with `ordinarylight.loaders.load()`.
- Additional output transports belong in `ordinarylight.outputs`; GUI-specific
  adapters remain in `ordinarylight.integrations`.
- Additional renderer implementations belong in `ordinarylight.backends` and
  implement the same HDR frame and capability contracts.
