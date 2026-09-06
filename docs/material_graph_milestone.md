# Shared material graphs and extended transport

`ordinarylight.materials.MaterialGraph`, `MaterialNode`, and `MaterialResource`
are reusable material facilities with no dependency on OrdinaryLattice. A graph
compiles to the same `MaterialProgram` contract accepted by camera materials.
OrdinaryLight owns scattering, PDFs, lighting, and medium transitions. Geometry
graphs, voxel construction, and artistic averaging remain outside this work.

```python
from ordinarylight.materials import MaterialGraph, MaterialNode
from ordinarylight.transport import TransportMaterial

metal = MaterialGraph({
    "rough": MaterialNode("constant", value=0.35),
    "metal": MaterialNode("constant", value=1),
}, {"roughness": "rough", "metallic": "metal"})
material = TransportMaterial("pbr", albedo=(0.8, 0.6, 0.2), program=metal)
# A camera scene material accepts program=metal.compile().
```

Graphs validate types, missing connections and cycles, including disconnected
nodes. Outputs inherit fixed material parameters unless connected. Supported
nodes include constants, context inputs, arithmetic, mix/select, vector dot and
normalization, component selection, uniforms, storage-buffer reads and sampled
textures. Parameter layers remain available through the existing material API.

## Materials, emission and lighting

`TransportMaterial.kind` chooses `diffuse`, `pbr`, `dielectric`, or `emission`.
All kinds may emit; emission-only surfaces terminate after emission. PBR supports
metallic fraction, roughness and dielectric IOR; dielectric supports ideal and
rough reflection/transmission. Graph and fixed definitions use the same evaluator
and scattering implementation in the non-camera path. Camera and non-camera
shaders share the `material_contracts` record; `shader_source("bsdf")` exposes
the new BSDF helpers to other public transport clients. Existing `TransportMaterial`
diffuse/ideal-glass defaults retain their behavior. `from_material` now adapts
opaque scene materials to PBR and admits rough glass; unsupported additional
lobes still require an explicit override.

The isotropic GGX implementation evaluates and samples matching densities.
Rejected microfacets are null samples rather than samples from an unaccounted
fallback distribution. Rough glass uses reflection/refraction Jacobians and the
radiance-mode index-of-refraction factor; see the
[PBRT derivation](https://www.pbr-book.org/4ed/Reflection_Models/Rough_Dielectric_BSDF).
This is a single-scattering microfacet model, so rough white surfaces can lose
energy through masking. It does not claim microfacet multiple-scattering recovery.

`VulkanTransportScene(lights=...)` accepts existing point, directional and spot
lights; by default it snapshots source-scene lights. Analytic direct-light samples
use visibility queries and homogeneous absorption. These lights are delta
sources, so their direct samples do not compete with finite-density BSDF samples.
`accumulate(environment_nee=True)` enables uniform-environment next-event sampling
with power-heuristic MIS on both light samples and BSDF escapes. It defaults off
to preserve the existing constant-environment sampling sequence/behavior.
Emissive geometry remains visible through BSDF continuation. The optional
`emissive_nee=True` surface-sampling path adds matched forward/reverse MIS; see
the follow-up contract below.

Boundary identities and optical media remain authoritative. Graphs cannot
silently change glass membership, transmission class, medium IOR, or absorption.
Incompatible outputs produce invalid-path diagnostics. Rough-glass scattering uses
the geometric normal by default; the opt-in `shading_clipped` policy is described
below. The ray origin is offset onto the classified geometric side to avoid
skipping very short grazing chords. Offset distances introduce a finite-epsilon
approximation, as with other ray-offset methods. The default transport SDF budget
is 8192 steps; difficult fields may still report traversal diagnostics.

## Resource-backed material graphs

Declare a `MaterialResource(name, kind)` with kind `uniform`, `buffer`, or
`texture`, and connect corresponding nodes. Uniforms contain one vec4; storage
buffers contain vec4 records indexed by a scalar node. Invalid buffer indices
return zero. Textures take a vec2 coordinate and sample explicit mip level zero.
Use a `components` node (e.g. value `"rgb"`) to select material outputs.

Supply `VulkanTransportScene(material_resources={...})`:

- Uniform/buffer declarations bind a runtime `VulkanBuffer`.
- Texture declarations bind an `(image, sampler)` pair, using `VulkanImage` and
  `VulkanSampler`.

Scene and kernels borrow these resources, with same-runtime and close-order
checks. Uniform and buffer contents can change without recompiling the material.
GPU producers participate in execution-graph dependencies; host uploads retain
explicit synchronization. Applications reset affected history themselves.

External-resource graphs also work in Vulkan camera GI through
`VulkanMaterialResources` (see below). Resource-free graphs continue to use
existing camera GI/raster/WebGPU material compilation. External-resource graphs
are not yet supported by the raster/WebGPU graph adapters.
The camera renderers retain their existing material models and resource systems;
this milestone does not replace their integrators with the non-camera tracer.
Triangle hits interpolate texcoord0. Known surface samples accept `texcoords`;
custom intersections currently leave UV at zero, so use explicit graph coordinates
or application resources for procedural geometry.

## Per-sample media and weighted accumulation

`initial_stacks` on the integrator declares additional validated boundary
sequences. Slot zero remains the legacy `initial_boundaries` sequence.
`ray_samples` and `surface_samples` accept `initial_stack` as a scalar or per-input
index. This uses reserved `media.w`, retaining the 96-byte input layout. GPU
indices are validated before stack-table access.

`SampleReduction.weights` scales radiance; `normalization_weights` defaults to the
same values for a coverage-weighted mean. Set normalization to one for a mean of
explicitly weighted contributions, such as a correctly specified importance
estimator. This API does not infer the intended measure or sampling PDF.
Weights must be finite and nonnegative. Zero total normalization resolves to zero.

The 48-byte accumulation record now uses previously reserved `radiance.w` for the
normalization sum. Integer path/event counts remain unweighted. CPU means and GPU
HDR resolve use the same denominator. External consumers that previously divided
RGB sums by valid-path counts must use this sum when weighting is enabled.
Host mappings and weights can be replaced with `set_reduction`. The optional
`GpuSampleReduction` contract accepts GPU-authored counts and maps instead.
Reset history when changing estimator semantics.

## Resource views and reflection

`VulkanResource.byte_range(offset, size)` creates a bounded buffer view used by
both descriptors and hazards. Offset is relative to its parent view. Descriptor
offsets must satisfy device alignment. The graph compiler splits overlapping
ranges into intervals for RAW/WAR/WAW and version validation; disjoint writers
need no artificial ordering. Whole-buffer declarations remain conservative.
Image dependencies still cover the whole image.

`VulkanResource.uniform_buffer`, `sampled_image`, and `sampler` extend the public
kernel bindings. `VulkanSampler` owns immutable nearest/linear, clamp/repeat
single-mip sampler state. Kernels retain it until close. OrdinaryShade reflection
supports set-zero uniform buffers, separate 2D sampled textures/samplers, storage
buffers/images, acceleration structures and push constants. Depth comparison,
array texture descriptors, and multiple descriptor sets are outside this adapter.

## GPU-discovered geometry

`GpuCustomGeometry(scene, source)` consumes one `GPU_CUSTOM_GEOMETRY_DTYPE`
64-byte record per reserved slot. A GPU producer can discover active slots,
compact records, change bounds, and assign materials without CPU readback.

Each record contains lower/upper vec4 bounds, a vec4 callback parameter payload,
and metadata: program-table index, custom-local material index, application
boundary ID, and application geometry ID. Program index `0xffffffff` disables a
slot. The scene's program table order follows intersection registration order.

`operation(mode="refit" | "rebuild")` validates records on the GPU, writes safe
bounds and metadata, then updates BLAS/TLAS in the graph. Invalid active records
are disabled before acceleration builds. Per-slot diagnostics distinguish invalid
program/material (1), bounds/parameters (2), and boundary assignment (4).
`read_diagnostics()` is an explicit readback; rendering does not require it.

Capacity, program registry and material palette remain host-reserved. Active count
and occupancy can vary freely within capacity; acceleration builds cover reserved
slots, including disabled ones. GPU allocation algorithms belong in the producer;
the renderer does not allocate Vulkan memory from a shader. Close GPU update
clients before `reserve_custom_geometry`; this explicit growth boundary synchronizes
the current device-authored state before reallocating. Recreate affected clients
and compiled graphs after growth. Accumulation invalidation remains application
policy, and graph resource contracts cannot prove an arbitrary callback's geometry
is contained in its supplied bounds.

## External proof and compatibility

Install `examples/transport_client` and run:

```bash
ordinarylight-material-graph --samples 64 --output /tmp/material-graph.png
```

This public-API client demonstrates graph-controlled metal and emission, rough
glass, GPU slot discovery and acceleration updates, unequal coverage reduction,
analytic lighting/environment MIS, persistent HDR and tone mapping. Its final
PNG/JSON readback reports geometry and transport diagnostics. The original
transport and animated-grid clients remain available as regression workloads.

### Validation

The final core suite passed 552 tests (59 optional skips); the focused GPU suite
passed 44 tests. Shader inventory checks found all 462 planned outputs present.

The acceptance workload includes mixed-medium absorption, unequal coverage and
explicit estimator normalization, graph/fixed emission and metal agreement,
rough-glass and equal-IOR boundaries, white-furnace energy checks, independent
solid-angle quadrature versus BSDF sampling, analytic inverse-square lighting,
environment MIS, sampled-resource production/consumption, buffer-range hazards,
and GPU geometry validation/refits/growth. The standalone client wheel is built,
installed separately, and run from outside the repository; both the new graph
client and original transport client complete successfully.

Scientific regressions include exact RT histogram agreement across Vulkan GI,
Vulkan Raster and WebGPU, likelihood data/presentation tests, and native viewer
interaction smokes. One initial GI smoke reached its success marker but reported
a device-loss error during teardown; two isolated reruns completed cleanly. No
specific teardown fix is claimed from that non-reproducing observation.

The non-camera material palette now contains three vec4 records per material
(albedo/kind, emission/sidedness, roughness/metallic/IOR/reserved). External raw
palette readers must use the expanded layout. Public fixed-material constructors
and scene/integrator entry points remain available.


## Resource-backed camera GI

Create a fixed binding bundle from the union of graph programs used by the camera
scene, then pass it through `RendererConfig`:

```python
program = graph.compile()
with ol.VulkanMaterialResources(
    runtime, [program], {"gain": gain_buffer, "texture": (image, sampler)}
) as bindings:
    config = ol.RendererConfig(material_resources=bindings)
    with ol.renderers.gi.VulkanGlobalIlluminationRenderer(
        runtime=runtime, config=config
    ) as renderer:
        # Mesh materials use ol.Material(program=program).
        frame = renderer.render_wavefront(scene, camera, 640, 480)
```

Declarations and allocations must match exactly at bundle construction; each
camera program may use a subset of that declared union. Conflicting declarations,
wrong allocation types/usages, incompatible runtimes and premature resource close
are rejected. Bindings occupy descriptor set 1, separate from existing scene
bindings. Both the monolithic camera renderer and staged primary/secondary
evaluators use this bundle. Staged resource graphs currently require
`wavefront` execution; `auto` selects that strategy and other explicit strategies
are rejected. Raster and WebGPU adapters remain outside this binding contract.

The bundle retains allocations, and attached renderers retain the bundle.
Close renderers before the bundle, then close allocations and runtime.
Changing contents does not rebuild descriptors or material pipelines. Call
`bindings.synchronize(after=[producer_completion])` after GPU producers and before
rendering; it waits for producers and establishes shader-read visibility and
GENERAL image layouts. Host uploads also require synchronization before rendering.
Do not mutate resource contents while a frame is using them. Reset affected
progressive/temporal histories explicitly; resource edits do not infer invalidation.
Changing declarations or allocations requires a new bundle and renderer.

Camera GI retains its existing scattering implementation. Tests compare bound
material evaluation with equivalent constant programs; this is not a claim of
identical sampling or scattering across camera and non-camera integrators.


## Emissive-source MIS follow-up

Non-camera transport now offers `emissive_nee=True` alongside environment MIS.
It samples triangle surfaces and opt-in custom surface samplers, evaluates the
actual per-hit graph emission, and balances those estimates with BSDF-hit emission.
The built-in SDF sphere supplies a sampler. See
[transport foundations](transport_foundations.md#emissive-geometry-sampling) for
PDF, visibility and sparse-capacity semantics. This improves small-source
convergence without asserting camera/non-camera scattering equivalence.


## Optical normals and GPU-authored scheduling follow-up

Non-camera accumulation now accepts the explicit experimental
`dielectric_normal_policy="shading_clipped"` policy. It preserves geometric medium
classification and treats wrong-hemisphere events as valid null samples.

`GpuSampleReduction` accepts GPU-authored active counts, sorted reduction groups,
indices and weights. GPU validation precedes indirect trace/reduce dispatch, and
invalid maps poison the batch's accumulator status rather than accessing unchecked
indices. Existing CPU grouping and geometric-normal defaults remain supported.
See [transport foundations](transport_foundations.md) for exact contracts and limits.
