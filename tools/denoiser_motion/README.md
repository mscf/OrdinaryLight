# Moving optics diagnostic

Run from the repository root with the project's Vulkan environment and the
optional NRD bridge built (`python tools/nrd_reference/bootstrap.py`):

```bash
python -m tools.denoiser_motion.run --output /tmp/optics-motion \
  --width 256 --height 160 --frames 16 --reference-samples 128
```

The output directory must be new. Requirements are native Vulkan ray queries,
wgpu with storage-image support, the pinned NRD bridge, NumPy and Pillow.
No window is required. This is an offline quality comparison, not a timing
benchmark or a production denoiser selection.

The fixture contains a glossy metal sphere, a planar perfect mirror, and a
Fresnel glass sphere, plus colored objects and an emissive room light. The
camera holds for three frames, moves sideways, then holds for two more frames.
All geometry is static. This isolates camera motion; animated optics and
independently moving reflected objects need additional fixtures.

At each viewpoint we capture one stochastic sample and a matching reference
averaged from two independent batches. Each batch has half the requested
reference samples (up to 64). Their disagreement is reported to expose residual
reference noise. These are finite-sample references, not exact ground truth.

The exact same canonical noisy inputs feed:

- The checked-in production OrdinaryShade temporal and A-trous WGSL shaders,
  executed by WebGPU on the native GPU. Temporal thresholds and history limits
  use the Vulkan renderer's camera-motion policy; spatial constants match the
  native three-iteration configuration.
- The same shader replay with only the motion history cap removed. Geometry
  rejection, reactive rejection and clamping remain active. This is a diagnostic
  ablation, not a recommended runtime setting.
- NVIDIA NRD RELAX through the optional native bridge, retaining history across
  the sequence. The diagnostic supplies separate projection/view matrices in
  NRD's column-major layout using bridge input format version 2. Frame time is
  fixed at 1/60 second; host upload and replay speed do not affect its weights.

Outputs include:

- `frame-*.npz`: canonical radiance, hit-distance, normal, roughness, depth,
  motion, material and frame metadata shared by both denoisers.
- `raw-*.npz`: original traced radiance, geometry and path-state captures.
- `reference.npy`, `raw.npy`, `ordinaryshade.npy`, `uncapped.npy`, `nrd.npy`:
  linear RGB sequences.
- `diagnostics.npz`: separate filtered lobes, both reference batches,
  per-lobe history-length and exact acceptance maps, and object masks.
- `report.json`: whole-frame motion metrics, per-object log-RGB errors,
  per-frame object errors, reference disagreement, motion policy and history
  summaries, shader hashes and adapter identities.
- `comparison-*.png` and `comparison.gif`: identical fixed tone mapping of
  reference, raw, OrdinaryShade and NRD images.

The temporal shader gets one diagnostic-only storage output for its final
`accepted` flag. Original shader hashes are retained, and the added write does
not change the numerical path. A replay with and without instrumentation was
image-exact. Acceptance and history length are distinct: an accepted sample can
still have history length one when the configured cap is one. NRD internal
acceptance maps are not exposed by this bridge.

Reuse an original capture to change reconstruction without retracing:

```bash
python -m tools.denoiser_motion.run --replay /tmp/optics-motion \
  --output /tmp/optics-motion-replay
```

Replay outputs refer to the original capture and do not copy its raw files.
Use the original capture directory for subsequent replays.

## Interpretation boundaries

This isolates filter behavior on canonical captures. It does not establish
parity with live Vulkan signal preparation or measure its performance. The
capture API reconstructs lobe attribution from path state; its signal semantics,
material demodulation and hit-distance conventions need auditing before using
this to rank production backends. Separate filters cannot repair missing
reflection/refraction motion information.

For this fixture each object has a distinct material, which is also used as
its stable identity in shader replay. NRD's bridge packs material IDs into two
bits (clamping larger IDs); its material boundary handling is consequently not
identical to our replay. Both receive the same world-space normal and 2D primary
surface motion, but only our replay additionally gets previous-camera depth
computed from the captured world position. NRD computes its own camera geometry
from the supplied transforms. Neither input includes tracked reflected or
refracted surfaces. Glass remains outside NRD's opaque-surface signal contract;
its numbers are diagnostic symptoms rather than evidence of supported glass
reconstruction.

The legacy NRD bridge format (without `camera_matrices`) is retained for older
callers. It substitutes combined world-to-clip matrices for projections and
must not be used as evidence for perspective-camera motion quality.

## Experimental motion-history floor

`RendererConfig(denoiser_motion_history_floor=3)` sets a minimum history cap of three
frames for finite camera motion, even when the motion-footprint cap
would permit only one. The explicit `denoiser_history_limit` remains an upper
bound. Geometry/reactive rejection and camera cuts still reset individual
pixels to one sample. The default floor remains **1**.

Compare the option on a saved capture:

```bash
python -m tools.denoiser_motion.run --replay /tmp/optics-motion \
  --history-floor 3 --output /tmp/optics-floor3
python -m tools.denoiser_motion.check_rejection
```

The latter runs GPU checks for moving-surface tracking, newly exposed
background, lighting changes, camera cuts, and the three-frame bound.

The live Vulkan gate also accepts `--history-floor 3`; an experimental floor
is recorded in its configuration and does not silently reuse the default
baseline. The optics capture showed lower fast-motion noise, but the existing
live camera test lost edge correlation. This option is therefore experimental,
not a new default. See [measured results](../../artifacts/denoiser-motion/history-policy.md).

## Live edge audit

The [live edge audit](../../artifacts/denoiser-motion/edge-audit/README.md) uses
actual Vulkan denoiser guides and independent high-sample references from the
same live rendering path. It revises the earlier interpretation of the
16-spp edge-correlation failure and documents the surface-local clamp fix.

## Signal audit correction (2026-09-06)

New captures now use the live renderer's probability partition of total
radiance, replacing the former sampled-indirect-lobe reconstruction. Existing
saved captures are unchanged. Neither partition is a physical lobe
decomposition; see [the signal audit](../../docs/denoiser_signal_audit.md)
for hit-distance, modulation, motion, and multisample limitations.

For the evaluated-lobe experiment, run:

```bash
python -m tools.denoiser_motion.check_sampled_indirect
python -m tools.denoiser_motion.check_lobe_capture
```

The first checks GPU preparation with controlled samples; the second checks
native optics capture, pure-metal attribution, and unchanged raw output when
toggling `denoiser_sampled_indirect`.

Live quality can be compared with `live_edges --scene optics` (or
`--scene object-motion`) and a matching run with `--evaluated-lobes`.
Use `compare_lobes BASELINE CANDIDATE --output NEW_DIRECTORY` to validate
matching references and generate metrics and a comparison image.
The [recorded study](../../artifacts/denoiser-motion/lobe-quality/README.md)
found mixed results; evaluated lobes remain experimental.

Broader-filter motion/stop/detail validation is available through
`validate_broader --kind object|camera|detail --output NEW_DIRECTORY`.
The [recorded results](../../artifacts/denoiser-motion/broader-validation/README.md)
support keeping the broader setting optional: recovery error improves, while
checker-texture contrast decreases.

A bounded static planar-mirror guide experiment is available via
`planar_mirror --motion camera|object --output NEW_DIRECTORY`. It compares
primary and reflected guides on identical reflected-camera radiance. The
[initial results](../../artifacts/denoiser-motion/planar-mirror/README.md)
preserve more contrast but increase noise, so this is not integrated into
native presentation.

Mirror captures now save per-frame signals and expected previous depth.
`ablate_mirror CAPTURE --output NEW_DIRECTORY` measures temporal-only,
reactive, spatial and depth-tolerance variants on finite reflected hits.
The [corrected diagnosis](../../artifacts/denoiser-motion/mirror-ablation/README.md)
supersedes the initial object-motion result and identifies depth rejection
as a major bottleneck; native defaults remain unchanged.

The optional replay-only depth-footprint shader experiment is recorded in
[adaptive mirror depth validation](../../artifacts/denoiser-motion/mirror-footprint/README.md).
It adds a ninth mirror ablation variant without changing production shaders.
