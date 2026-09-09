# Glass-detail viewer scenes

Start with the moving sphere:

```bash
.venv/bin/python tools/raster_feature_viewer.py --target wavefront-gi --showcase glass-detail-motion
```

The scene selector offers three related fixtures:

- **Glass detail: camera motion** (`glass-detail-camera`): stationary glass and
  target, using the viewer's normal camera orbit. Manual orbit and zoom work too.
- **Glass detail: moving sphere** (`glass-detail-motion`): only the glass moves.
- **Glass detail: moving target** (`glass-detail-target`): only the striped target moves.

Keep **Animate scene / camera** enabled to run the selected motion. Sphere and
target motion repeat a six-second cycle: move for one second, hold for two,
reverse for one, then hold for two. Uncheck Animate to freeze the current pose
and inspect denoiser settling. The glass and target modes keep the camera still
unless you manipulate it manually. Switching scenes creates a fresh history;
it is not a continuation of the preceding scene's history.

ReLAX and ReSTIR use the normal GI viewer defaults. The planar-mirror guide
checkbox does not apply to these scenes. Bright/dark/orange emissive bars make
blurred or displaced refracted detail easier to identify. The glass uses IOR
1.52 and radius 1; its denoising guides still describe the front surface.

The offline comparison shares this fixture builder:

```bash
PYTHONPATH=. .venv/bin/python -m tools.denoiser_motion.glass_detail
```

That diagnostic uses frame-indexed translation, independent reference renders,
and fixed comparison patches. The interactive camera orbit is intentionally not
an exact replay of its horizontal camera translation. See the
[history investigation](../artifacts/denoiser-motion/glass-history/README.md) for
known residual error. The optional cap below is disabled by default.

Validation: 24 fixture/viewer tests passed. Native viewer checks completed
603 frames (motion), 600 frames (target), 121 frames (camera).
All exited cleanly. Status dimensions describe the requested viewer extent,
not a verified internal rendering resolution.

## Experimental glass motion history cap

Enable **Glass motion history cap (experimental)**, then click **Apply and
restart renderer**. The option is off by default. It limits history to four
frames on marked transmissive primary surfaces whose screen-space guide motion
exceeds one pixel. History grows normally after motion stops. Both denoiser
lobes use the same cap. Compare identical movement/hold phases with it off/on.

The low-level option is `RendererConfig.denoiser_transmission_motion_cap`.
The marker is supported by the standard wavefront primary pipeline (including
native-texture and profiling variants) and dynamically compiled custom-material
primary pipelines. Hybrid, megakernel, and SER variants are not extended by this
change. The glass showcase uses the custom Fresnel material. Opaque surfaces and unmarked samples retain
the existing temporal policy. This is not a refracted-background motion model:
a stationary glass surface cannot detect a moving target through this marker.

See the [integration validation](../artifacts/denoiser-motion/glass-cap-native/README.md)
and [eight-case diagnostic sweep](../artifacts/denoiser-motion/glass-cap-sweep/README.md).

## Viewport size and fullscreen

The direct viewer renders to the viewport size in physical pixels; its former
resolution selector has been removed. Press **F11** to enter an undecorated
fullscreen view with controls hidden. Press **F11** again or **Escape** to
restore the window and controls. Render scaling configured by a renderer can
still reduce its internal resolution. The separate offscreen comparison mode
retains its output-resolution selector.

A mouse-transparent FPS overlay stays at the top-left of the render surface
in windowed, maximized and fullscreen views. It reports the rolling completed-
frame rate used by the status panel, including presentation and scene-update
costs, rather than an isolated GPU timing.

The GI overlay also reports actual internal dimensions, GPU frame time, the
four largest GPU stage groups, and host scene/record/wait/present times.
GPU timings come from an earlier completed frame; host and GPU times overlap
and should not be summed. **Copy live diagnostics** exports the recent timing
samples for comparison. Timestamp collection uses the production shaders;
it does not enable the heavier work-counter profiling variants.

See the [4K performance investigation](../artifacts/denoiser-motion/4k-performance/README.md)
for measured costs and the distinction between native 4K and upscaled output.

## Experimental inline continuations

For `glass-detail-camera`, `glass-detail-motion`, or `glass-detail-target`,
enable **Inline continuations (experimental)** and click **Apply and restart
renderer**. This keeps the first three bounces in the custom primary shader
before continuing through the staged queues. It is off by default and the
control is disabled for other showcases.

The tested sequences had essentially unchanged reference error, bias and
settled noise. Tiny ray differences can still produce isolated pixel
outliers; this is experimental rather than bit-identical execution. See the
[quality gate](../artifacts/denoiser-motion/inline-quality/README.md).
Copied diagnostics distinguish requested, resolved and actually dispatched
execution modes; enabled inline execution should report `hybrid` as the
actual `wavefront_execution_strategy`.

## GI render scale

Choose **GI render scale** and click **Apply and restart renderer**. Native
100% remains the default; 75%, two-thirds, 50%, and 25% reduce both internal
width and height while presentation follows the viewport, including F11
fullscreen. At 4K output, 50% renders GI at 1920 × 1080 and 25% at
960 × 540 (one sixteenth as many pixels as native). The overlay shows
`internal → output` dimensions when they differ, and copied diagnostics
include both extents. The control is disabled for the NRD reference preview
and other rendering targets.

This uses the existing reconstruction path and does not change the ReSTIR
reservoir count. Reduced resolution can soften glass, reflected detail and
thin edges. The earlier ~29 ms moving / ~22 ms stationary measurements used
both 50% scale and **one** reservoir, under uncontrolled GPU load; they are
not a performance promise for the default four-reservoir configuration.

For a sharper spatial upscale, select **GI upscale filter → Clamped cubic
(experimental)** and click **Apply and restart renderer**. It only changes
reduced-resolution output; 100% bypasses cubic. Bilinear remains the default.
Cubic uses 16 HDR image reads instead of four in the existing reconstruction
pass, with no additional image allocations or temporal history. Each result
is clamped to the central 2×2 color range to limit overshoot. It can emphasize
jagged edges and residual noise as well as detail.

See the [GPU output comparison and provisional timings](../artifacts/denoiser-motion/cubic-upscale/README.md).

**FSR 1 EASU (experimental)** is also available in **GI upscale filter**.
Select it at a reduced render scale and apply/restart. RCAS sharpening is
not enabled. The prototype uses AMD's pinned FP32 EASU code, with tone-mapped,
sRGB-encoded input samples fetched from denoised HDR. It adds no temporal
history or new images. No additional anti-aliasing stage is introduced, so
input aliasing and residual noise can still affect its quality.

FSR 1 bypasses scaling at 100%. It currently rejects the separate temporal
reconstruction, stationary accumulation, and diffuse reconstruction filter
options, since its input bypasses those operations. Ordinary Shade ReLAX
and the viewer's existing temporal denoising remain supported.

See the [three-filter comparison and timing results](../artifacts/denoiser-motion/fsr1-upscale/README.md).

## Experimental FSR 2 temporal upscaling

The Linux Vulkan viewer also supports **FSR 2 (experimental, native bridge)**.
Build the optional bridge once from the checkout:

```sh
.venv/bin/python scripts/build_fsr2.py
```

This requires g++, Vulkan development headers/libraries, and glslangValidator
(the existing `.tools/glslang` extraction is supported). Then select FSR 2
in **GI upscale filter** and **Apply and restart renderer**. Start at 50% scale.
For an installed wheel, set `ORDINARYLIGHT_FSR2_LIBRARY` to the built `.so`'s
absolute path. Bilinear, cubic, and FSR 1 do not need the bridge.

FSR 2 uses AMD's temporal upscaler on the presenter's existing GPU device,
with coherent frame jitter, HDR color, reversed finite depth, pixel motion
vectors, and a conservative reactive mask for glossy or invalid surfaces.
RCAS sharpening is disabled. At 100% it still runs temporal reconstruction;
it is not a spatial-filter bypass like FSR 1 or cubic.

This initial mode requires a perspective camera and Ordinary Shade ReLAX.
Disable **Planar mirror guides** and any separate temporal reconstruction,
stationary accumulation, diffuse reconstruction filtering, or object effects.
The ordinary ReLAX histories remain active. FSR history is recreated on extent
changes and reset when the denoiser history is invalid. FSR dispatch commands
are recorded each frame rather than reused from the command cache.

The reactive mask reduces reliance on history for shiny surfaces; it does not
provide motion through refraction or reflected-object motion. Glass/mirror
quality during arbitrary motion remains experimental. The GPU overlay and
copied diagnostics include a separate `fsr2` stage.

See the [FSR 2 comparisons and provisional timings](../artifacts/denoiser-motion/fsr2-upscale/README.md)
and [native build details](../native/fsr2/README.md).

## OrdinaryShade EASU comparison

**EASU OrdinaryShade (experimental)** is a typed FP32 port of AMD's EASU
algorithm. Select it in **GI upscale filter**, then apply/restart. The original
**FSR 1 EASU** option remains the reference implementation. Both have sharpening
disabled and bypass spatial scaling at 100%. No native library is needed for
either option.

The port lives in `ordinarylight/shaders/easu.py`, with AMD's copyright/MIT
notice and pinned source attribution. Its pure filter helpers generate both
GLSL and WGSL without the AMD headers. The current combined Vulkan shader
still includes AMD's implementation so both paths can be compared in one
viewer; the new path executes the OrdinaryShade helpers.

See the [numerical, rendered-output and timing comparisons](../artifacts/denoiser-motion/easu-shade/README.md).
