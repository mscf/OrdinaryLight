# Changelog

Notable user-facing changes are recorded here. This project follows semantic
versioning while its public API develops toward 1.0.

## Unreleased

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
