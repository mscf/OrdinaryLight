# GI denoiser signal audit

Audited 2026-09-06 against the live Vulkan path-to-HDR resolve, signal
preparation, temporal/spatial filters, and offline capture.

## Findings

| Signal | Current behavior | Consequence |
| --- | --- | --- |
| Diffuse/specular RGB | Live resolve partitions the complete nonnegative HDR value by the primary material's specular probability. | Energy is conserved before filtering, but these are not independently evaluated physical lobes. |
| Offline RGB | Previously split primary light by probability and indirect light by the sampled branch. Now reproduces the live probability partition. | Newly captured comparisons use the production partition; old captures retain different semantics. |
| Material modulation | Radiance includes material response; no demodulation/remodulation stage exists in the local filter. | Texture and material variation remain in the signal being filtered. |
| Hit distance | Both channels contain the same primary-to-first-recorded-secondary Euclidean distance, or zero without a recorded secondary hit. | This is neither a per-lobe distance nor a tracked optical chain. Zero does not distinguish sky, termination, and an unavailable record. |
| Distance consumption | Local temporal and A-trous stages preserve alpha but do not use it for rejection or spatial weights. | Better distance data alone would not improve the current local filter. |
| Geometry and motion | Live preparation uses primary position, primary normal, primary roughness, and previous transformed primary vertices. | Mirror/glass imagery has no separately tracked reflected/refracted motion. |
| Transmission | Resolve exposes only two probability-partitioned channels and no transmission channel or optical-chain identity. | A specular channel cannot be interpreted as a validated reflection/transmission decomposition. |
| Multiple samples | Live resolve partitions the averaged HDR result using the final sample's secondary record. | It cannot recover independent per-sample lobe contributions or distances after accumulation. |

Source locations:

- Live partition: `scripts/generate_core_shaders.py::wavefront_path_to_hdr`
  and generated `ordinarylight/shaders/wavefront_path_to_hdr.comp`.
- Bounce metadata: `scripts/generate_core_shaders.py` secondary capture in the
  shade stage; `scripts/generate_primary_shaders.py` primary capture helpers.
- Offline partition: `ordinarylight/targets/vulkan/api.py::capture_denoiser_raw`.
- Guide preparation and local filtering: `ordinarylight/denoising/kernels.py`.
- Portable signal documentation: `ordinarylight/denoising/signals.py`.

## Correction made

Offline capture now partitions total radiance by probability, exactly as the
live resolve formula does. It no longer substitutes sampled indirect-lobe
attribution. This fixes a comparison discrepancy without changing production
rendering or pretending to supply a physically improved decomposition.

The focused regression includes identical radiance/probability with opposite
sampled branches, probability endpoints, valid/missing secondary hits, energy
conservation, and preservation of the executor's original records.

This establishes formula agreement, not complete live/offscreen parity:
sampling, guide construction, half-float storage, and execution settings can
still differ. Historical optics reports must not be presented as measurements
of the corrected capture. The live edge audit remains a separate live capture
and is unaffected.

## Next implementation

Accumulate actual primary diffuse/reflection/transmission contributions during
transport, before sample averaging. Separate primary emission/direct terms
explicitly instead of assigning them by a sampling probability. Record valid
per-channel secondary distances with explicit miss semantics.

Validate that recomposition reproduces raw HDR across diffuse, mixed glossy,
mirror, glass, emissive and environment-lit fixtures at one and multiple
samples per pixel. Then define material demodulation together with its inverse
composition step; never divide radiance without restoring the response.

Once this foundation is measured, prototype reflected-surface guides for
perfect mirrors, including cuts and disocclusions. Transmission requires its
own branch/chain treatment. Neither a backend comparison nor the presence of
hit-distance fields establishes readiness for those optical guides.

## Experimental sampled-indirect preparation (2026-09-07)

`RendererConfig(denoiser_sampled_indirect=True)` opts into a partial
decomposition. It is disabled by default.

For each completed sample, preparation subtracts the captured primary
radiance from total radiance and assigns the remainder to the sampled primary
branch (diffuse or specular). Primary radiance is still probability-partitioned.
Both channels are accumulated into full-frame images before averaging is
complete, rather than applying the final sample's branch to the entire batch.
The first sample overwrites the prior frame's signal images. No new images
are allocated; existing preparation images now support read/write access.

Capture uses the same single-sample formula. In experimental mode the first
secondary distance is supplied only to the sampled channel. At multiple
samples this is the final sample's guide, not a representative distance for
the whole accumulated channel. Refraction remains grouped with specular
events; no independent transmission channel is introduced.

This implements per-sample indirect classification, **not complete physical
lobe separation**. Direct BSDF contributions, emission isolation, material
demodulation, multisample distance reduction and reflected/refracted guides
remain outstanding. No visual-quality improvement or default promotion is
claimed yet.

Run `python -m tools.denoiser_motion.check_sampled_indirect` to execute the
packaged preparation shader on synthetic mixed-branch samples. It checks
channel values, recomposition, new-frame reset, branch-specific distance and
disabled-path compatibility.

## Evaluated primary lobe capture (2026-09-07 follow-up)

The experimental path now records the evaluated primary specular direct
contribution in the primary transport shader. Point/directional/spot lights,
sampled area/environment lights, and accepted ReSTIR area/environment
contributions retain the componentwise specular fraction of their actual
BSDF evaluation. Candidate target evaluations are not accumulated.
Visibility, light sampling, MIS and reservoir normalization are shared with
the original contribution; no additional rays or random draws are introduced.

The sampled indirect contribution is also split by its evaluated componentwise
BSDF ratio, rather than solely by the chosen sampling branch. This matters
because mixed-BSDF sampling returns a combined BSDF weight. The OrdinaryShade
PBR continuation uses its own diffuse expression for this ratio; it must not
reuse the layered direct-light evaluator's ratio.

The existing secondary-record channel slots serve as temporary storage:
negative alpha marks primary-specular RGB in the specular slot and an
indirect-specular fraction in the diffuse slot. Experimental path-to-HDR
preserves that scratch until preparation consumes it. Ordinary capture and
preparation outputs still contain nonnegative distance alpha. Unsupported
producers without this marker retain the earlier probability/branch fallback.

Primary emission and any other non-specular primary residual currently remain
in the diffuse output. This preserves energy but is not an isolated emission
channel. Material demodulation, a separate transmission channel, representative
multisample distances, and optical motion guides remain future work.
The experiment remains disabled by default, pending visual-quality validation.

Validation of the follow-up:

- Both live primary GLSL and the generated wavefront shade producer now emit
  the evaluated-lobe scratch contract. The latter was necessary for ordinary
  offscreen captures; updating only the live shader left capture on fallback.
- The native optics check uses 96×48 pixels, seed 9, two bounces and tiled
  capture. The glossy metal's summed diffuse RGB falls from 15.650883 to
  0.000001425 (float32 accumulation residual), with the energy transferred
  to specular. Raw RGBA is image-exact between enabled and disabled captures.
- GPU preparation checks cover evaluated primary specular, emission residual,
  componentwise mixed-BSDF indirect fractions, multisample accumulation,
  recomposition, frame reset and legacy mode.
- 643 tests passed, 117 skipped, 74 subtests passed. The shader inventory
  compiled, denoiser artifacts verified, and the experimental native viewer
  started and closed successfully.
- No motion-quality or timing improvement is claimed by these checks.

Reproduce the native check with
`python -m tools.denoiser_motion.check_lobe_capture`.

## Motion-quality decision

The [paired live quality study](../artifacts/denoiser-motion/lobe-quality/README.md)
found an optics improvement but a motion-room regression. At history floor
three, overall error changes by -13.6% and +13.9%, respectively. The experiment
therefore remains disabled by default. Signal correctness and recomposition
checks do not imply a general denoising-quality improvement.

## Canonical capture motion correction

Canonical capture previously reused general output motion (current-minus-previous).
The documented DenoiserSignals contract and both portable/native temporal filters
expect previous-minus-current. Capture now converts the displacement and adds the
current ray sample's offset from its integer pixel center. Reset/background motion
remains zero. General output motion and native GPU preparation are unchanged.

CPU tests cover static samples, camera translation and object translation. A
rendered sloped fixture compares captured motion against independent previous
camera-matrix projection. Historical offline motion quality studies need rerunning;
see the [motion audit](../artifacts/denoiser-motion/motion-audit/README.md).
