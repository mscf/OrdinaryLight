# Changelog

Notable user-facing changes are recorded here. This project follows semantic
versioning while its public API develops toward 1.0.

## Unreleased

- Added an accepted multi-scene HDR noise-quality gate covering diffuse,
  area-light, glossy/glass, fast-motion dense, and volume rendering, with
  explicit-review baseline replacement and firefly/detail-preservation metrics.

- Established the `ordinarylight` package, semantic public API namespaces,
  formal quality gates, and extensible Qt workbench.
- Added formal backend protocols, optional/lazy Vulkan dependencies, downstream
  wheel-consumer validation, and supported headless/integration examples.

## 0.1.0

- Initial development release.
