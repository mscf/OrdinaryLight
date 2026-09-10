# OrdinaryLight shader authoring

All OrdinaryLight-authored shader logic must be written in typed OrdinaryShade
Python. This includes entry points, shared helpers, shader specializations, and
portable shaders. Treat GLSL, WGSL, and SPIR-V as generated build artifacts.

- Implement shader behavior in OrdinaryShade sources and regenerate artifacts.
- Do not add handwritten shader bodies, hide GLSL/WGSL in Python strings, or use
  opaque external functions to conceal renderer algorithms. Backend intrinsics
  and mechanical ABI/compiler directives are distinct from shader algorithms.
- Do not edit generated shader bodies directly. Fix the typed source or compiler.
- Migrate existing handwritten logic without dropping rendering features or
  changing estimators to make a port appear equivalent.
- Preserve resource layouts and compilation variants. Validate generation,
  compilation, image parity, and relevant GPU performance before switching paths.
- Track incomplete migrations explicitly; a generated header alone is not proof
  that the underlying logic is authored in OrdinaryShade.

The user permits existing third-party FSR paths to remain temporarily while this
migration proceeds. The intended end state is still OrdinaryShade-only: FSR2 is
planned for removal and FSR1 must be replaced or rewritten as needed. This is
not permission to introduce new handwritten shader dependencies.
