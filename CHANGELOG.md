# Changelog

Notable user-facing changes are recorded here. This project follows semantic
versioning while its public API develops toward 1.0.

## Unreleased

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
