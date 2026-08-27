# Changelog

Notable user-facing changes are recorded here. This project follows semantic
versioning while its public API develops toward 1.0.

## Unreleased

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
