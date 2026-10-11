# Portable surfaces and lines

`ordinarylight.portable.surface.surface_assets()` returns the mesh runtime,
drag controller, and generated shader. `line_assets()` also includes the generic
screen-space stroke runtime. OrdinaryScience supplies the plotting semantics
and UI around these renderer assets; OrdinaryLight does not depend on it.

Shader behavior is authored in `shaders/portable_surface_programs.py` using
OrdinaryShade. Run `python scripts/generate_portable_surface.py --check` to
verify the packaged WGSL. Existing browser tests cover clipping, visibility,
picking, wheel policies, lines and overlays. Their software WebGPU checks do
not establish performance or parity on every physical GPU.

These assets were migrated from the untracked portable-surface work in the
local wave-render checkout without changing that original checkout.
