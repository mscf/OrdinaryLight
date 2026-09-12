# Changelog

Notable user-facing changes are recorded here. This project follows semantic
versioning while its public API develops toward 1.0.

## Unreleased

## 0.5.0

- Add public persistent wavefront graph operations with reusable recording and borrowed GPU outputs.
- Support typed OrdinaryShade intersection, material, optical-boundary and emitter callbacks over application-owned acceleration structures and GPU buffers.
- Export sampled-ray primary hits, application instance/sub-element identity, raw HDR, denoised linear HDR and denoiser signals.
- Add public content/resource update and history controls with explicit synchronization and lifetime contracts.
- Fix orthographic denoising and ReSTIR reprojection; retain the same transport/history for direct and application-processed output.
- Document the vxl8r adapter and its GPU face averaging, viewer integration and resource ownership.

## 0.4.1

- Add experimental four-bounce, one-sample GI with reflections and sensible viewer defaults.
- Eliminate repeated GI command recording during animation by uploading ReSTIR history policy per frame.
- Reuse primary BRDF calculations and specialize volume-free, opaque, and production ReSTIR shaders without reducing sampling quality.
- Fix long fullscreen/resize/shutdown stalls after switching from raster to GI by renewing the Vulkan instance during renderer handoff.
- Recover from out-of-date raster swapchains and bound GI acquisition retries.
- Avoid compiling unused shared-primary pipelines and improve shutdown behavior.
- Add viewer timing, fullscreen, command-recording, and primary-cost diagnostics, including statistics for scene-compiled pipelines.

## 0.4.0

- Complete the OrdinaryShade shader migration, including viewer GI, transport,
  material dispatch, texture/lighting libraries, volumes, and WebGPU shaders.
- Enforce shader authorship and regenerate all packaged shader artifacts.
- Require OrdinaryShade 0.1.0a5 and remove the temporary compiler patch.
- Add shared-primary ReSTIR, bounded command history, lifecycle diagnostics,
  and stable-width viewer performance counters.
- Preserve upstream FSR temporarily; document the pre-existing difference
  between the two volume multiple-scattering implementations.

- Added reusable declarative material graphs, including emission, uniforms,
  storage-buffer and sampled-texture inputs for non-camera transport.
- Added metallic/rough PBR, rough dielectric scattering, analytic-light NEE and
  optional environment MIS to non-camera transport.
- Added per-sample initial medium stacks and explicit contribution/normalization
  weights; accumulation radiance.w now stores the normalization sum.
- Added aligned buffer views with interval-based graph hazards, uniform-buffer
  descriptors, separate sampled-image/sampler bindings and owned samplers.
- Added GPU record validation and custom-geometry acceleration refits/rebuilds
  within reserved capacity, plus a public-API material-graph client (0.4.0).

- Added single-queue execution graphs with resource versions, alias/hazard checks,
  OrdinaryShade access-reflection adapters, and recordable rendering operations.
- Replaced per-dependency host waits with GPU queue ordering/barriers; added
  bounded frame rings and semaphore-based graph presentation.
- Added custom-geometry slot activation/removal, bounds refits/rebuilds, and
  capacity growth preserving unrelated allocations and rebinding integrators.
- Added persistent tone-map targets and migrated the external client to graphs,
  including an animated fixed-grid workload with GPU-generated occupancy.

- Added declared read-only buffer/image resources to custom intersections, with
  generated bindings, dependency barriers, and allocation lifetime guards.
- Added reusable GPU sample allocations and in-place integrator input updates.
  Explicit sample-to-output reduction supports multiple faces per output without
  floating-point atomics; GPU-generated inputs are validated before traversal.
- Migrated the external transport client to resource-backed geometry, reusable
  samples, and many-to-one reduction while preserving existing NumPy callers.

- Added public application-indexed multi-bounce Vulkan transport for Lambertian
  and ideal dielectric surfaces, with strict nested media, Fresnel/TIR decisions,
  distance-dependent absorption, and explicit invalid/truncated path diagnostics.
- Added common triangle/custom ray-query intersections, bounded field contracts,
  uniform transforms, CPU composition helpers, and an analytic GPU SDF sphere.
- Added persistent per-identity GPU accumulation and HDR resolve, plus a separately
  installable public-API transport client. Existing camera GI shaders and
  scientific rendering entry points remain unchanged.

- Extracted a public Vulkan runtime and reusable resident scene upload service;
  GI can borrow application-owned runtimes and scene snapshots.
- Added typed ordered Vulkan passes, persistent allocations, fence completions,
  versioned transport components and application-identity history contracts.
- Added independent external-HDR tone mapping, native presentation and RGBA8
  GPU-frame export. Existing scientific render paths retain their defaults.
- Fixed Vulkan Raster render-pass CFFI attachment lifetimes uncovered by the
  RT-volume integration regression.

- Forwarded same-device compute contexts through the high-level `Renderer`, so
  workbench extensions retain device identity on wrapped rendering paths.
- Added a public same-device compute context to headless Wavefront GI and moved
  Qt scene/renderer startup off the event thread, with visible progress,
  deferred non-blocking close, cancellable extension initialization, and fair
  compute/presentation scheduling.
- Raised every Vulkan instance path to the renderer's Vulkan 1.2 baseline,
  matching generated SPIR-V 1.5 shaders in both native Qt and headless use.
- Added reflected multi-pass SPIR-V compute sequences on an existing Vulkan
  renderer device and direct resident-buffer uploads into Vulkan raster volumes.
- Added same-device WebGPU compute-buffer views and direct GPU-resident volume
  sources for downstream simulation and visualization integrations.
- Added an explicit `module:factory` Qt workbench extension hook so downstream
  applications can own domain-specific panels without reverse dependencies.

## 0.3.5 - 2026-08-27

- Added explicit picking policies, asynchronous GPU picking with a portable
  CPU fallback, and DPI/letterbox/dynamic-resolution coordinate mapping.
- Added up to four simultaneous object effects plus built-in outline, tint,
  emissive, isolation, projected-bounds, and X-ray-bounds responses. Existing
  singular v0.3.4 effect calls remain compatible.

## 0.3.4 - 2026-08-27

- Added transport-independent scene picking with stable object identifiers,
  detailed mesh hit results, and camera-aware pixel-to-ray conversion.
- Added a composable object-effect API, with a configurable GPU-rendered
  `effects.Outline` for windowed and encoded output. Picking remains independent
  from application selection state and visual response.
- Added click selection and selection status to the Qt workbench as an
  interactive demonstration of the public selection API.

## 0.3.3 - 2026-08-27

- Added zero-recreation NVENC recovery controls: periodic IDRs, queued
  `request_keyframe()` recovery, per-frame IDR/header flags, repeated SPS/PPS,
  and a cumulative forced-keyframe counter.
- Added optional GPU-time-driven motion SPP scaling. It selects the highest
  sample count up to `samples_per_pixel` that fits the interactive FPS target,
  composes with motion resolution scaling, and reports the effective SPP on
  GPU-resident frames.

## 0.3.2 - 2026-08-27

- Fixed hosted CI collection by installing the Vulkan loader and optional
  Python binding for hardware-independent renderer unit tests while keeping
  GPU gates opt-in.
- Added optional motion-aware internal resolution scaling for the wavefront
  backend. Moving and settling frames can render at a configured lower scale
  while presentation and exported video allocations remain full resolution.
- Added target-FPS motion scaling, which automatically chooses a bounded
  interactive render scale and restores full quality while stationary.

## 0.3.1 - 2026-08-27

- Added stationary-aware temporal accumulation for conventional and wavefront
  rendering, including zero-copy NV12/P010 streaming. GPU frames now report
  moving, settling, and accumulating state without readback; camera, scene,
  extent, and hot-setting changes invalidate history without recreating the
  Vulkan renderer or NVENC frame pool.

## 0.3.0 - 2026-08-27

- Added transactional `Renderer.replace_scene()` and hot
  `Renderer.reconfigure()` APIs. Scene and common settings transitions now
  retain the Vulkan device, compiled pipelines, history/output allocation, and
  two-frame external NV12/P010 pool; failed scene uploads leave the prior scene
  resident.
- Reduced first-frame startup work by creating only the configured execution
  strategy's shader pipelines (the scene-dependent `auto` strategy retains its
  complete strategy set). Added a hardware gate for startup, transition
  latency, device identity, and external-pool identity.

## 0.2.1 - 2026-08-27

- Added direct linear-HDR-to-P010 BT.709 limited-range output and zero-copy
  10-bit HEVC/AV1 NVENC input, including explicit bit-depth metadata, an
  example, and 4K hardware-gate coverage.

## 0.2.0 - 2026-08-27

- Added two-frame GPU-resident Vulkan output with explicit external-memory and
  ready/release semaphore ownership, plus a zero-copy H.264 path that performs
  tone mapping and NV12 conversion in Vulkan and feeds CUDA/NVENC without host
  pixel readback. Added a 4K end-to-end NVENC gate and optional
  `video-gpu` dependency group.

- Added an accepted multi-scene HDR noise-quality gate covering diffuse,
  area-light, glossy/glass, fast-motion dense, and volume rendering, with
  explicit-review baseline replacement and firefly/detail-preservation metrics.

- Established the `ordinarylight` package, semantic public API namespaces,
  formal quality gates, and extensible Qt workbench.
- Added formal backend protocols, optional/lazy Vulkan dependencies, downstream
  wheel-consumer validation, and supported headless/integration examples.

## 0.1.0

- Initial development release.
