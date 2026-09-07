# Standard wavefront stock-material transmission capture

Packaged primary denoiser variants now share the same capture/marker macro as
the dynamically compiled custom primary. Four variants cover ordinary,
native-texture, profiling, and native-texture+profiling configurations. The
standard wavefront executor selects the denoiser variant when capture or ReLAX
is active. No runtime GLSL compiler is required for stock-material execution.

The macro includes transmitted samples in primary-geometry capture and marks
material transmission in the sign bit of nonnegative barycentric u, as already
documented for the custom pipeline. The old string substitutions in the custom
compiler are replaced by enabling this shared macro.

Validation uses only built-in materials on the glass/opaque striped fixture.
The native test asserts custom_primary_pipeline is absent. With the cap off,
no motion.w pixels are marked. With it on, 10,976 pixels are marked, all instance
ID 0 (glass). The fixed central patch has valid denoising depth throughout in
both runs. Opaque instances are unmarked. This verifies capture/selection, not
physical equivalence of the built-in and custom Fresnel scattering programs.

All four existing non-denoiser primary binaries recompile byte-identically.
90 focused tests and ten subtests passed, including manifest coverage of the
four variants and shader compilation. Hybrid, megakernel and SER paths are not
extended by this change; this support is for standard staged wavefront primary
execution. The UI cap remains opt-in and adaptive rejection remains experimental.

Run capture.py from the repository root with PYTHONPATH=. on a working Vulkan
desktop. It writes sequences and guide buffers to /tmp/glass-stock and asserts
that the stock primary pipeline is actually used.

Custom-path parity rerun after the shared-macro refactor: complete off/on
sequences remain byte-identical to their prior baseline/prototype; reordered
glass still receives the marker at instance ID 1. See custom-parity.log.
