# Material-marked glass motion cap integration

The opt-in RendererConfig.denoiser_transmission_motion_cap replaces the fixture
identity selector with material metadata. Custom primary signal capture marks
material transmission > 0.001, independent of whether this random sample chose
reflection or refraction. Opaque primary surfaces remain unmarked.

The cold secondary scratch contract reserves the sign bit of nonnegative
barycentric u (primary_geometry.y) as the marker. Preparation recovers u with
absolute value, preserving precision and signed-zero behavior, and leaves
primitive and instance identity unchanged. No scratch stride or descriptor
layout changes are needed. Other scratch consumers do not interpret this lane.
This extension is documented beside SECONDARY_PATH_STATE_DTYPE.

Preparation writes motion.w = 1 only for marked material when the option is
enabled. Otherwise it writes zero. Temporal filtering caps history at four when
motion.w > .5 and length(motion.xy) > 1 pixel. Existing eligibility and rejection
rules still apply. The marker is not a refracted-background motion vector.

Native validation at 320x240 on the moving-glass fixture:

- Option off: complete image sequence matches the prior baseline exactly.
- Option on: complete sequence matches the identity-selected cap4 prototype exactly.
- Reordered scene with an extra offscreen opaque triangle: 10,976 marked pixels
  select glass instance ID 1 instead of ID 0; no opaque instance is marked.
- Default-off marker texture has zero marked pixels.

75 focused material/config/viewer/fixture tests and ten subtests passed; 12
additional denoiser motion/mirror tests passed. Generated WGSL/SPIR-V artifacts
were verified. This does not extend marker production to stock shader variants.

The viewer checkbox is off by default and requires Apply and restart renderer.
The capture script requires the earlier /tmp/glass-rejection sequences only
for its final parity comparison; it writes raw captures under /tmp/glass-cap-native.
Run from the repository root with PYTHONPATH=. and a working Vulkan desktop.

The actual Qt checkbox/restart flow completed 601 animated frames and verified
that the active renderer had the cap enabled. The run exited cleanly. Viewer
status reports the requested extent, not a verified internal render resolution.
No performance claim is made from this smoke test.
