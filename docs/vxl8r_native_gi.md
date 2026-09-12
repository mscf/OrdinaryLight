# vxl8r native GI integration

Inspected the application clone at `/home/mscf/projects/personal/vxl8r`.
This describes the reusable native implementation and its vxl8r adapter.
The public native renderer and viewer integration are implemented. A controlled
visible-window test passed; the existing hidden-window presentation quarantine
remains in place.

## Existing application contracts

| Source | Current contract | Native integration consequence |
| --- | --- | --- |
| `backends/resident.py` | Stable coordinate-to-slot map over a reserved domain; occupancy and geometry generated on the GPU | Keep slot allocation, occupancy, traversal and dirty-slot tracking in vxl8r |
| `backends/resident_geometry.glsl` | 64-byte records: lower/upper bounds, parameters, uint4 metadata; metadata.w is slot identity | Slot identity is application data, not a TLAS primitive index |
| `backends/box.glsl` | `intersectCell` returns status, world distance and geometric normal | The current v1 callback does not return face identity, shading attributes or an optical boundary |
| `backends/resident_samples.glsl` and `resident_scan*.glsl` | Exposed-face samples are compacted and reduced to cells | Preserve these application algorithms where needed; camera GI must instead consume hits from its actual sampled rays |
| `backends/resident.py` | `VulkanTransportScene` and `GpuCustomGeometry` own/update the current acceleration structures | The clone does not currently supply its own raw TLAS; an external owner must retain both the TLAS and referenced BLAS dependencies |
| `backends/renderer.py` | Lighting, separate visibility, application effect, tone map, presentation | Replace the surface-sample transport/visibility pair with native GI, then average HDR using native primary identities |
| `backends/visibility.glsl` | Separate unjittered orthographic camera queries; color fetched by slot | These visibility hits cannot be reused as identities for jittered native radiance |
| `backends/resident_history_*.glsl` | Per-slot material/occupancy comparison, local resets and blending | Keep dirty-slot policy in vxl8r; return invalid previous-position correspondence for dirty slots |

The resident renderer already produces geometry, updates acceleration structures,
and runs transport in a GPU graph. The simpler `backends/ordinarylight.py` adapter
rebuilds a scene and reads cell means back to the CPU; it is not the resident path.
The existing legacy renderers still use that path. The new experimental public
`NativeResidentVulkanRenderer` consumes the same resident GPU population, native
camera transport, denoising, and optional post-denoiser face averaging.

The clone currently creates diffuse materials from albedo and emission. It does
not yet define application optical-boundary or custom-emitter sampling callbacks.
Those extension contracts must therefore be defined and tested independently;
the existing box callback cannot validate them by itself.

## Required native contract

1. Import an application-owned acceleration resource and typed GPU bindings with
   explicit leases and no dummy triangle upload. Its owner keeps referenced BLAS
   allocations alive. Buffer content updates and AS refits can precede lighting
   in the same queue submission; replacement occurs at a lifetime boundary.
2. Accept typed OrdinaryShade intersection and surface evaluation callbacks.
   Intersection returns the actual instance, slot and face identity along with
   distance, geometric/shading normals and material/boundary information.
   Keep the nearest candidate payload associated with the committed intersection;
   a later candidate must not overwrite the selected hit's attributes.
3. Invoke the same geometry contract for primary, secondary, shadow and ReSTIR
   visibility. Candidate generation, resampling and emissive-hit MIS must share
   the custom emitter's sampling and PDF contract. Merely adding primary AABB
   traversal would leave black reflections and incorrect shadows/lighting.
4. Export camera hits directly from transport. For each GI sample/pixel, identity
   must survive through denoiser guides and history correspondence. Custom previous
   positions cannot be reconstructed by indexing the native triangle vertex ABI.
5. Average denoised linear HDR in the application before display processing.
   Keep native raw HDR and signals available separately for diagnostics. Preserve
   native history consistently in both direct and application-composed output.

## Verified library foundation

- Presentation-independent `VulkanWavefrontPipeline` shares native transport and
  history with `present_wavefront()`; moving-camera HDR parity is GPU-tested.
- Native HDR is consumable in a `VulkanGraph` before presentation; the public tone
  mapper accepts its RGBA16F format directly and can exclude allocation padding.
- Optional per-sample primary-hit records export the exact triangle camera rays.
- Native resident resource replacement is leased and invalidates stale graphs;
  in-place updates can preserve command caches and history explicitly.
- `runtime.import_scene()` accepts an existing application TLAS and optional
  native-layout buffers without building any acceleration structures. Empty
  scene metadata is accepted; no dummy triangle is necessary. The query test
  now runs through this importer with the existing transport AS owner. Native
  four-bounce triangle GI also matches the uploaded scene when importing the
  TLAS and material allocation, including custom material program IDs.
- Native primary, secondary, split-intersection, surface shadow and indirect
  ReSTIR visibility queries now use the same typed `nativeTraceSurface` helper.
  Volume-entry queries retain their separate visibility mask and volume policy.
- The shared query's procedural candidate path is GPU-tested with a typed box
  callback and the existing transport scene's application-borrowed AABB TLAS.
  Tests cover closest-hit payload selection, slot/face identity, inside-origin
  rays, misses, finite visibility intervals and early-terminating shadow queries.
  The query mask is explicit: vxl8r's current builder uses mask 2 for custom AABBs,
  whereas native surface queries use mask 1.

A complete custom-geometry native camera test now uses `NativeGeometryProgram`
and `VulkanNativeGeometryResources` with the imported AABB TLAS. It verifies a
non-emissive primary surface receiving light from a secondary custom emitter,
one-bounce versus four-bounce lighting, and exact slot/face IDs from the primary
sampled ray. Custom queries include surface masks 1 and 2; ordinary triangle-only
queries retain mask 1. Application buffers occupy descriptor set 2, independently
of the native scene ABI at set 0. No private descriptor overrides are required.

Custom material parameter evaluation now feeds the native BSDF implementation in
primary and fused OrdinaryShade secondary shading. Emissive custom surfaces
contribute through path samples and application-defined emitter sampling/PDFs
for next-event estimation. Primary/secondary NEE and ReSTIR use the same
application-owned distribution. An optional typed optical-boundary callback
handles smooth lossless dielectrics in primary/secondary transport and straight visibility. Absorption,
rough interfaces, and boundary-topology validation remain unsupported.

Custom denoiser history now consumes the callback's previous world position and
full-identity fingerprint. GPU tests verify accumulation, moving correspondence,
and local rejection without resetting unaffected pixels. Exact primary-hit
identities remain available for application averaging.

The current custom geometry configuration explicitly rejects inline continuation,
split or bucketed secondary shading, and indirect reservoir reuse. These paths are
not silently treated as supported. The vxl8r viewer still defaults to its legacy
renderer; native GI is available through the public API and the viewer's
`--renderer native-gi` option.

Custom buffers support `notify_content_changed` (same-queue publication without a
CPU wait, optionally preserving commands) and `replace_buffers` (leased resource
replacement at an explicit idle boundary). Tests verify pending-frame guards,
lighting changes after updates, command invalidation, and borrowed lifetimes.

GPU boundary tests cover primary/secondary slab entry and exit, a camera starting
inside glass, Fresnel reflection toward an emitter, and total internal reflection.
Straight visibility passes index-matched interfaces and blocks refractive ones.
Both reflected and refracted delta events bypass continuous-light MIS weighting.

Emitter GPU tests compare primary/secondary NEE, multiple light samples, ReSTIR
temporal/spatial reuse, and live emitter-count changes against path-only energy.
Work counters verify actual shadow queries and accepted reservoir history. The application owns
the complete area-emitter distribution and must preserve emitter IDs and sample
coordinates while retaining reservoir history.

## Resident application adapter progress

`vxl8r_render.backends.native_geometry` now supplies typed OrdinaryShade callbacks
over the real 64-byte voxel records, 48-byte palette, exposed-face flags, and
dirty-slot buffer. `bind_native_voxels` leases these buffers, including the public
`scene.resource("materials")` view. OrdinaryLight accepts storage-buffer resource
views from same-runtime owners with `require_open`, `retain`, and `release`;
byte offsets/ranges survive descriptor replacement.

The app's population-only graph generates occupancy, bounds, acceleration,
exposed faces and dirty flags before native GI in one submission. It omits the
legacy surface-sample transport and reduction. GPU validation moves population
between slots, checks hit/miss and slot/face identities from the sampled camera
rays, and checks linear emissive HDR. Buffer uploads, reads, and image/buffer
allocations are forbidden during this graph's execution in the test.

Emitter IDs are stable `slot * 6 + face`; the initial uniform distribution
includes null samples for inactive, hidden or non-emissive faces. This is a
correctness baseline, not an optimized emitter distribution for sparse domains.
The native material maps resident albedo/emission to rough dielectric PBR; it
retains native specular reflection, unlike the legacy Lambertian transport.
Dirty slots invalidate previous-position correspondence locally. Nonlocal
lighting changes still need an application history policy.

The public `NativeResidentVulkanRenderer` now provides persistent post-denoiser
face averaging in a separate RGBA32F image, followed by tone mapping/presentation.
`averaged=False` preserves upstream transport and denoiser history. GPU checks
compare the toggled renderer's raw/denoised history against an independent direct
renderer and compare its face colors against NumPy using the exact primary IDs.
The native population owner omits legacy sample/accumulator/integrator allocations.

Averaging currently requires one camera sample per GI pixel. Both output modes
use nearest GI pixel-center mapping at differing output resolutions, before tone
mapping. Native HDR and raw HDR remain separately accessible; explicit readback
helpers export HDR and sample-major primary records for diagnostics. Settings,
resize and close retire output resources, while display toggling and unchanged
settings preserve them. Camera reprojection for native orthographic denoising and
ReSTIR now uses parallel-ray math, verified against known world points at different
depths as well as the perspective case.

All averaging shaders are typed OrdinaryShade. Striped GPU lists and float32
partial sums avoid HDR quantization and float atomics; a typed integer
`atomic_exchange` intrinsic was added to OrdinaryShade's GLSL backend. Scratch
resources persist across frames. Standalone output-stage submission/wait medians
on the RTX 5090 Laptop GPU were 2.68 ms at 1080p and 7.20 ms at 4K for a single
full-screen face; direct mapping measured 0.71 ms and 1.20 ms respectively.
These exclude native transport/denoising and are not full-renderer FPS results.

The viewer supports direct/face-average output switching without history reset,
bounce changes, public history invalidation, and a typed OrdinaryShade help
overlay on the separate display output. A 12-frame visible-window test completed
12 queue-present calls, three scene renderer lifetimes, and four swapchain
lifetimes, including resize and full cleanup, in 8.38 seconds. Nine headless GPU
and viewer checks passed, including unchanged upstream HDR with the help overlay
and no output allocations/uploads after warming both frame slots. These checks
do not lift the application's hidden-window quarantine or establish sustained
performance. Sampling/reconstruction tuning remains application-specific.
Generic custom geometry still has the explicit unsupported continuation modes
and optical limits listed above; the new adapter does not silently enable them.
