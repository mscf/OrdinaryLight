# Non-camera transport foundations

OrdinaryLight provides a reusable Vulkan path tracer for application-selected
rays and surface samples. It handles traversal, diffuse continuation, dielectric
boundaries, absorption, and accumulation without constructing the camera GI
renderer. The [external client package](../examples/transport_client) exercises
this API using only public imports, without implementing its own transport loop.

Materials now include diffuse, metallic/rough PBR, ideal/rough dielectric, and
emission, with shared graph-defined parameter evaluation. Point/directional/spot
light NEE, optional constant-environment MIS, and opt-in emissive-surface MIS
are supported. See
[material graphs and extended transport](material_graph_milestone.md) for resource
bindings, medium/weight contracts and current integration boundaries. Participating
scattering is not implemented in this path.
Existing camera GI, raster, WebGPU, and scientific viewer entry points retain
their behavior. Custom geometry is currently available through this Vulkan
transport API, not through those camera renderers.

## Public API

`ordinarylight.geometry` owns bounded fields, transforms, intersection programs,
and the common hit contract. `ordinarylight.transport` owns medium definitions,
`VulkanTransportScene`, `VulkanTransportIntegrator`, `GpuSampleAccumulator`,
`ray_samples`, `surface_samples`, `GpuTransportSamples`, `SampleReduction`, and
diagnostic `intersect_rays`.

A hardware Vulkan ray-query adapter and `glslangValidator` or `glslc` are required.
Custom callbacks and transport kernels compile at runtime.

```python
from ordinarylight.runtime import VulkanRuntime
from ordinarylight.geometry import SdfSphere
from ordinarylight.transport import (
    OpticalMedium, MediumBoundary, TransportMaterial,
    VulkanTransportScene, VulkanTransportIntegrator,
    GpuSampleAccumulator, ray_samples,
)

with VulkanRuntime() as runtime:
    with VulkanTransportScene(
        runtime,
        custom_geometry=[SdfSphere().geometry(boundary=17)],
        custom_materials=[TransportMaterial("dielectric")],
        media=[OpticalMedium(), OpticalMedium(1.5, (0.2, 0.4, 0.6))],
        boundaries=[MediumBoundary(17, outside=0, inside=1)],
    ) as scene:
        with GpuSampleAccumulator(runtime, 1) as accumulation:
            samples = ray_samples([[0, 0, 3]], [[0, 0, -1]])
            with VulkanTransportIntegrator(scene, samples, accumulation) as transport:
                transport.accumulate(
                    samples_per_element=1024, max_bounces=32,
                    environment=(1, 1, 1),
                )
                completion = accumulation.resolve()
                # Feed accumulation.hdr to VulkanOutput with after=completion.
                print(accumulation.means())  # Explicit CPU readback.
```

`surface_samples` starts at a known surface. Incoming directions point toward
the surface. Material indices refer to the transport scene's table: triangle
records followed by the custom material palette. Custom geometry material
indices refer to its local custom palette. Boundary arguments are application
identities. For existing NumPy callers, sample IDs select accumulator slots by default.
An explicit `SampleReduction` instead maps each input slot to an output ID,
allowing multiple face samples to contribute to one application output.

## Fields and custom geometry

`BoundedField` evaluates a signed value, supplies a gradient and declares bounds.
`FieldKind.EXACT_DISTANCE` permits distance-sized steps.
`CONSERVATIVE_DISTANCE` means a safe signed-distance bound whose magnitude cannot
overestimate distance to the zero set. An estimate with no such guarantee is
`SCALAR`; the generic sphere tracer rejects it. Scalar-field callbacks must
implement their own valid root finder.

`SdfSphere` supplies analytic distance/gradient and bounded GPU sphere tracing.
Step-budget exhaustion reports unresolved traversal, not a miss. Near-tangent
rays may require more steps. Tolerance and ray displacement use world units;
features or gaps smaller than the displacement are outside this numerical
contract. `UniformTransform` supports translation, rotation and positive uniform
scale; nonuniform scale and reflections are rejected. CPU `FieldComposition`
supports union/intersection/difference with conservative stepping guarantees.
It is not an automatic GPU composition compiler. Voxel reconstruction and
smoothing remain downstream.

`CustomGeometry` declares a world-space AABB, four float parameters, material,
optional boundary identity, application identity, and an `IntersectionProgram`:

```glsl
uint entry(vec3 origin, vec3 unit_direction, float t_min, float t_max,
           vec4 parameters, float tolerance, uint max_steps,
           out float distance, out vec3 geometric_normal);
```

Return 0 for miss, 1 for hit, 2 for unresolved/error. The caller clips the interval
to the AABB and current closest hit. A hit must return a finite distance inside
that interval and an outward unit geometric normal. Invalid output fails the
path. Callbacks can additionally declare read-only application buffers/images with
`IntersectionResource`; the four per-primitive parameters can index those data.
See the resource binding contract below.

Triangle BLASes and a procedural AABB BLAS share one TLAS and closest-hit query.
Both produce position/distance, geometric/shading normals, primitive/application/
material identity and boundary metadata. Geometric normals stay outward
regardless of ray side; triangle winding must describe the intended boundary.
Shading normals do not decide medium membership.

`intersect_rays` runs the same GPU traversal for diagnostics. Its 80-byte
`HIT_DTYPE` contains five aligned groups: `position_distance`, `geometric_normal`,
`shading_normal`, `identity`, and `boundary`. Identity contains kind (0 miss,
1 triangle, 2 custom), primitive, application identity, and material index.
Boundary contains table index (0xffffffff for none), outside medium, inside
medium, and traversal status. The host `SurfaceHit` names equivalent fields.

## Per-hit custom attributes (version 2)

Existing callbacks default to `IntersectionProgram(hit_version=1)`. Opt into
`hit_version=2` to replace the final two output arguments with
`inout OrdinaryLightCustomHit hit`. All earlier arguments are unchanged.
The runtime initializes the record before each call. Return codes remain 0
(miss), 1 (hit), and 2 (unresolved/error).

The record contains `float distance`, `vec3 geometric_normal`, `uint flags`,
`uint material`, `uint boundary`, `uint identity`, `vec2 uv`, and
`vec3 shading_normal`. Always set distance and geometric normal for a hit.
Set only the desired override flags:

| Flag | Meaning |
| --- | --- |
| `OL_HIT_MATERIAL` | Select `material` from the scene's local custom-material palette. |
| `OL_HIT_BOUNDARY` | Resolve application boundary ID `boundary` through the scene table; `OL_NO_BOUNDARY` explicitly clears it. |
| `OL_HIT_IDENTITY` | Return unsigned 32-bit application `identity`, independently of the chunk primitive index. |
| `OL_HIT_UV` | Supply finite UV coordinates; otherwise zero. |
| `OL_HIT_SHADING_NORMAL` | Supply an outward unit shading normal in the geometric normal's hemisphere; otherwise use the geometric normal. |

Absent identity/material/boundary flags inherit the enclosing primitive's
metadata. The inherited material/boundary pair must remain valid at scene
construction. A callback may override both to choose a different transmission
class. Material graphs themselves still cannot change transmission class.
Unknown flags, palette indices, boundary IDs, or incompatible material/boundary
pairs fail with status 32 before material/boundary dereferences. Nonfinite UVs
and invalid normals fail with status 1. Application IDs are opaque uint32 values,
not array indices. Callbacks remain responsible for bounds-checking their own
resource accesses.

The committed-hit record remains 80 bytes. UVs occupy geometric-normal.w and
shading-normal.w for both triangles and custom geometry. Optional shading normals
do not change the default dielectric scattering policy.

A heterogeneous chunk can select a material and cell identity on each hit.
A connected optical region must use one boundary identity across chunks; chunk
edges and internal cell faces are not automatically optical interfaces.
OrdinaryLight validates per-hit references, not watertightness or region topology.
See `examples/custom_hit_attributes.py` for a minimal public-API diagnostic.

## Declared custom resources

Each `IntersectionProgram` may declare `resources=(IntersectionResource(...),)`.
Supply their allocations by name in `VulkanTransportScene(custom_resources=...)`:

```python
from ordinarylight.geometry import IntersectionProgram, IntersectionResource

program = IntersectionProgram(
    "intersectGrid", callback_source,
    resources=[IntersectionResource("distanceGrid", element_type="float")],
)
# callback_source can read distanceGrid[index] and distanceGrid.length().
# Build CustomGeometry with program, then supply:
# custom_resources={"distanceGrid": application_buffer}
```

OrdinaryLight generates read-only declarations and assigns set-0 bindings from
16 upward, consistently for diagnostic and transport kernels. Callbacks must
not hard-code those bindings or redeclare the named resources. Shared names
across programs must have identical declarations and refer to one allocation.
Missing, extra, conflicting, wrong-kind, wrong-stride and wrong-runtime resources
are rejected. Multiple names may alias an allocation; pass uses merge aliases.

Buffers are std430 arrays of float/int/uint scalars or two-/four-component vectors.
Their sizes must be multiples of the declared stride. Current images are
read-only RGBA32F `image2D` storage images, accessed with `imageLoad`/`imageSize`.
Dense 3-D fields and sparse structures can live in linear storage buffers;
3-D images, samplers, descriptor arrays and arbitrary struct declarations are
not part of this initial resource schema. Callbacks are trusted GLSL: they must
bounds-check application data accesses and honor their stepping guarantees.

Allocations must belong to the scene runtime and remain alive until consumers
close. Scenes and descriptor kernels retain them; early close raises before
destroying Vulkan handles. CPU `buffer.upload()` is synchronized. For GPU writes,
declare a producer `VulkanPass` and pass its completion through
`accumulate(after=[producer])` or `intersect_rays(after=[producer])`. Consumers
declare shader-read barriers and transition images to GENERAL automatically.
There is no automatic inference of an application's producer dependency.

Resource contents can change without rebuilding shaders or scenes while their
allocation and the geometry's conservative AABB remain valid. If an edit moves
the surface outside its AABB, request update_custom_geometry with new bounds
(or its recordable operation). Adding slots beyond capacity uses
reserve_custom_geometry; see [execution graph](execution_graph.md). Applications must reset
affected accumulated outputs when the expected lighting changes. Field uploads
do not perform bounds refits or history invalidation automatically.

## Reusable samples and reduction

Existing construction with NumPy `ray_samples`/`surface_samples` still works.
`integrator.update_samples(new_samples, reduction=..., after=...)` uploads new
records into the existing allocation without recompiling kernels. The active
count can change within its capacity. CPU-mapped capacity growth requires new
inputs and an integrator; GPU-mapped clients can use `grow_capacity` below. Updates preserve history and the sampling epoch deliberately.

Use `GpuTransportSamples` for spare capacity or GPU-generated inputs:

```python
from ordinarylight.transport import GpuTransportSamples, SampleReduction

with GpuTransportSamples(runtime, 64, samples=face_samples) as inputs:
    mapping = SampleReduction(face_to_output_ids)  # one ID per active input
    with VulkanTransportIntegrator(
        scene, inputs, accumulation, reduction=mapping,
    ) as integrator:
        # GPU producers bind inputs.buffer using VulkanResource.buffer(...).
        integrator.accumulate(after=[producer_completion])
        # Or perform a synchronized CPU update:
        integrator.update_samples(next_faces, reduction=next_mapping)
```

GPU inputs use the same 96-byte `SURFACE_SAMPLE_DTYPE`/`OrdinaryLightSurfaceSample`
layout as host inputs. `identity.w` is 0 for rays or 1 for known surfaces;
`identity.z` is the known surface's scene material index. `media.z` contains an
application boundary ID, or 0xffffffff for none. The integrator resolves it on
the GPU; applications must not pre-remap it to a scene table index. The scene
boundary table is authoritative; this integrator does not use input media.xy
to override it. Directions
and normals must be finite unit vectors. Invalid records produce status 32
before any material/boundary array dereference, and still propagate to the
output diagnostics. No input readback is required.

An explicit `SampleReduction` is required when borrowing `GpuTransportSamples`.
It maps **input slots**, not GPU record identities, to output IDs. Record identities
remain sampling identity/stream metadata. The map is host-declared and uploaded
as stable groups. For GPU-authored maps, use the optional contract below. Contribution and
normalization weights support unequal coverage and explicit estimators. With the
default unit weights, output remains total radiance divided by valid path count.
With weighting, use the normalization sum in `radiance.w`, not the valid count.
Attempted/valid/truncated counts and events are summed; status flags are ORed.

`set_reduction(mapping, after=...)` changes grouping without rebuilding pipelines.
After `inputs.set_count(...)` or a count-changing `inputs.update(...)`, set a
matching reduction before dispatch. A direct `inputs.update(...)` with unchanged
count preserves the current map. In contrast, `integrator.update_samples(...)`
without a reduction uses the new host records' output IDs, matching initial
NumPy construction. Reset affected accumulator IDs explicitly when changing the
quantity being estimated; updates never silently erase application history.

Close integrators before borrowed inputs. The legacy `SampleReduction` path uses
host-declared active counts and adds one reduction pass and scratch record per
input. The GPU-authored mode below uses validated indirect dispatch instead.
Both avoid float atomics and CPU result copies.

## GPU-authored counts and reduction maps

Use `GpuSampleReduction(runtime, capacity, group_capacity=...)` with
`GpuTransportSamples` and pass it as `VulkanTransportIntegrator(..., reduction=...)`.
Sample and reduction capacities must match. Changing them requires new input
and reduction allocations and a new integrator. This API does not allocate sparse
memory from GPU shaders; scene geometry retains its explicit growth API.

A GPU producer writes these public storage buffers:

| Buffer | Layout |
| --- | --- |
| `counts` | One uvec4: active input count, active group count, two ignored words. |
| `groups` | uvec4 per group: output ID, first index, index count, ignored word. |
| `indices` | uint per active input: sample slot. |
| `weights` | vec2 per sample slot: contribution and normalization weights; initially one. |

Groups must have strictly increasing output IDs and partition the active indices
contiguously, without empty groups. Input indices within each group must be
strictly increasing, and each referenced sample's `identity.x` must match the
group output ID. Together with bounded indices and counts, this ensures every
active sample occurs exactly once. It preserves the deterministic summation order
of the host grouping implementation. The producer owns grouping/sorting; the
renderer validates the map rather than constructing it on the CPU.

OrdinaryLight validates counts, ranges, ownership, ordering and finite nonnegative
weights on the GPU before tracing or reducing. Validated counts generate indirect
dispatch commands. Both counts may be zero for a no-work batch. Malformed maps
suppress tracing/reduction and set invalid-input status 32 on **all accumulator
slots**, without adding contributions. Clear/reset affected history before retrying.
Malformed sample records retain the existing per-sample transport diagnostics.

Declare producer writes in a `VulkanGraph`, or pass their completion through
`after`. No count/map readback or CPU grouping is required. The integrator's
`count` property reports allocated capacity in this mode; `counts.x` is the
authoritative active GPU count. Host `update_samples`/`set_reduction` calls are
rejected on these integrators: update producer buffers instead. Sample IDs can
still carry independent sample-index metadata in `identity.y`.

### Growing GPU-mapped capacity

`replacement = integrator.grow_capacity(new_capacity, group_capacity=..., after=...)`
allocates larger input/map storage and copies the existing records on the GPU.
It preserves active counts, weights, initial medium stacks, the random sampling
epoch, and the existing accumulator. No input or map readback is performed.
Group capacity defaults to its current value; request it explicitly when growing
the number of groups. Shrinking and unchanged capacities are rejected.

On success the source integrator is closed. The replacement owns its new inputs
and reduction; access them through `replacement.samples` and
`replacement.gpu_reduction` to bind producers. Release producer/resource bindings
before migrating owned allocations, and recreate graph operations afterward.
Externally owned original inputs/maps remain the caller's responsibility. Failed
validation or replacement allocation leaves the source usable.

This is a synchronized allocation boundary: the migration waits for its copy.
GPU producers must fit within allocated capacity; this is not GPU heap allocation
or automatic overflow retry. Added sample slots are zero initialized and added
weights default to one. Existing active counts are preserved, so new slots are
inactive until a producer initializes them and expands the counts/maps. Accumulator
capacity is unchanged. The external `gpu_reduction` client demonstrates two growth
steps, producer rebinding and uninterrupted weighted history.

Close integrators before the reduction and input objects. Counts start at zero,
so a newly allocated map cannot dispatch uninitialized samples. Dispatch limits
are checked against the device. GPU validation and command preparation add four
passes to the existing trace/reduce pair; this is a correctness-oriented contract,
not a claim that GPU grouping is faster for small CPU-authored batches.

## Dielectric semantics

The default `dielectric_normal_policy="geometric"` keeps boundary classification
and ideal/rough dielectric scattering on the geometric normal. Existing callers
retain that behavior, including when custom hits return shading normals.

Opt into `dielectric_normal_policy="shading_clipped"` on an accumulation call
to use the shading normal for Fresnel, reflection/refraction, rough scattering,
BSDF evaluation and PDFs. Entry/exit classification, medium stack updates and
ray displacement still use geometry. If the shading frame hides the incident
direction, or the selected reflection/transmission crosses the wrong geometric
hemisphere, that event contributes zero further radiance. It is retained as a
valid null sample, with its original sampling probability: there is no resampling,
fallback normal, or renormalization of surviving directions. Previous emission
and direct-light contributions remain intact.

This is an explicit experimental approximation for optical normals on blocky
boundaries. Large normal tilts can darken the result; it is not equivalent to
intersecting a different smooth surface and does not establish energy conservation
for arbitrary normal fields. This radiance-only integrator retains its eta-squared
transmission weighting; it does not implement an adjoint/light-tracing normal
correction. See PBRT's discussions of
[geometric hemisphere classification](https://www.pbr-book.org/3ed-2018/Materials/BSDFs)
and [shading-normal transport asymmetry](https://www.pbr-book.org/3ed-2018/Light_Transport_III_Bidirectional_Methods/The_Path-Space_Measurement_Equation).
Tests cover unchanged equal-normal results, clipped Fresnel branch probabilities,
unit-environment bounds for ideal/rough custom surfaces, and medium diagnostics.

Medium zero is vacuum. Each dielectric surface identifies a
`MediumBoundary(identity, outside, inside)`; multiple faces can share one closed
boundary. Triangle mappings use mesh/instance IDs. Scene material conversion
accepts diffuse or ideal transmission; explicit `TransportMaterial` overrides
allow applications to choose a simpler material deliberately.

Transmission pushes/pops a strict nested stack; reflection leaves it unchanged.
The stack holds vacuum plus seven nested regions. Rays starting inside geometry
must supply `initial_boundaries` in outer-to-inner order. Outside media must match
the enclosing stack. Non-LIFO exits, repeated active boundaries, overflow, and
escape with an open medium are diagnosed. Arbitrary overlapping media and
coincident ambiguous boundaries are unsupported. This is per-path validation,
not a global watertightness or region-nesting proof.

Both representations use exact unpolarized Fresnel, Snell refraction, total
internal reflection, and radiance-mode transmission weight
`(eta_incident / eta_transmitted)**2`. Fresnel branch probability already accounts
for its weight. Homogeneous Beer–Lambert absorption is
`exp(-sigma_a * distance)` per RGB channel, including deliberate outgoing ray
displacement. Absorption coefficients use inverse world-distance units. There
is no scattering inside these media.

## Emissive geometry sampling

Pass `emissive_nee=True` to `accumulate` or `accumulate_operation` to sample
one surface point at every eligible scattering vertex. Triangle geometry is
sampled uniformly in area. `SdfSphere` includes a uniform sphere sampler.
BSDF-hit emission and sampled emission use matched area-to-solid-angle PDFs and
power-heuristic MIS. This option is independent of `environment_nee`; both may
be enabled together. Both default to false for compatibility.

The initial selection distribution is uniform over all triangle primitives and
reserved custom slots. Inactive custom slots, custom programs without samplers,
nonemissive surfaces and occluded points contribute null samples. Their selection
probability is not redistributed. This deliberately keeps forward/reverse PDFs
consistent during GPU geometry edits, without discovering emitters on the CPU.
Large sparse capacities or mostly nonemissive scenes can waste samples; this is
not yet a power-weighted emitter hierarchy. Unsampled custom emitters retain
unweighted BSDF-hit emission.

Custom geometry opts in with
`IntersectionProgram(..., sampling=SurfaceSamplingProgram(name, source))`:

```glsl
uint name(vec4 parameters, vec3 randoms, out vec3 position,
          out vec3 geometric_normal, out float area_pdf);
float name_pdf(vec4 parameters, vec3 position, vec3 geometric_normal);
```

Coordinates and normals are world-space; densities are per unit world-space
area, conditional on selecting this primitive but including any null-sample
probability. Random inputs lie in [0,1). Return 0 for a null sample, 1 for a
sample, and 2 for failure. The sampler and reverse PDF may read the intersection
program's declared resources. They must describe the same distribution over the
actual surface. The runtime cannot prove normalization or full support.

Successful samples require finite positions inside the primitive AABB, unit
normals, and finite positive densities. Forward and reverse densities must agree
at sampled points; visible samples must agree with the intersection geometric
normal. Invalid results fail paths with intersection status 1.
Visibility queries recover the actual closest-hit material, boundary, UVs and
application identity; samplers cannot override those attributes. Consequently
one chunk sampler can illuminate from heterogeneous graph-driven cells.
Emission sidedness, geometric visibility and homogeneous absorption still apply;
shadow rays do not transmit through intervening glass.

Changing sampler declarations requires rebuilding the scene/integrator. Updating
declared resources or existing custom records uses the existing synchronization
and history-invalidation contracts. Capacity changes rebuild dependent kernels.
The sample, hit and accumulation buffer layouts and 64-byte transport push
constant size are unchanged.

## Accumulation and output

`GpuSampleAccumulator` owns persistent per-ID sums/counts and an RGBA32F HDR image.
Multiple integrators on one runtime can share it; submissions chain through its
last completion. Transport writes one temporary record per input slot, then a
deterministic grouped reduction writes one accumulator record per output ID.
This supports duplicate destinations without float atomics. `reset()` clears
all records on the GPU; `reset(identities=[...])` clears selected records.
Applications decide which identities become invalid.

The 48-byte `ACCUMULATION_DTYPE` has these groups:

| Group | Components |
| --- | --- |
| radiance | weighted RGB sum, normalization-weight sum |
| counts | attempted, valid, ORed status flags, truncated samples |
| events | diffuse bounces, reflections, transmissions, total internal reflections |

Status bits are 1 unresolved/invalid intersection, 2 inconsistent boundaries,
4 stack overflow, 8 escape with an open medium, 16 nonfinite transport, and
32 invalid GPU input (including out-of-range material/boundary references).
`read()` and `means()` raise on invalid paths by default; `read(strict=False)`
exposes diagnostics. HDR resolve marks affected IDs magenta. Excluding invalid
paths must not be interpreted as a converged estimate.

`max_bounces` defines a finite-order estimate. A path needing another scatter
increments the truncation count, separately from invalid paths. Increase this
budget when approximating infinite-bounce transport. There is no implicit
Russian roulette or convergence claim. Each integrator epoch is limited to
2**24 samples; applications must retire/reset long-running histories before
floating-point sum and integer-counter limits become relevant.

`resolve()` writes resident HDR means without CPU readback. Pass its completion
and `accumulation.hdr` to `VulkanOutput` for tone mapping, native presentation or
GPU frame export. The external client deliberately reads back only for its final
PNG/JSON/NPZ exports.

## Ownership and scheduling direction

Triangle source scenes remain snapshots; source mutation requires replacement.
Custom geometry now supports mutable slots, bounds refits/rebuilds, and explicit
capacity growth. It can borrow triangle buffers/BLASes while owning the combined
TLAS and custom acceleration data. Integrators borrow scenes and accumulators:
close integrators first, then resources, then the runtime. The runtime lock
serializes host state and queue/pool use.

[Execution graphs](execution_graph.md) compose recordable transport, reduction,
reset, resolve, acceleration updates, tone mapping and presentation. Same-queue
dependencies use GPU ordering/barriers. Frame rings bound outstanding work;
readback, CPU uploads, capacity growth and frame-slot reuse can still wait.
Accumulation, HDR and prepared tone-map outputs are persistent. Applications
continue to own history invalidation and whether work needs recomputing.

## Validation

CPU tests cover field guarantees, transforms/composition, near-root handling,
normals, Fresnel/Snell/TIR, absorption and medium stacks. Opt-in GPU tests compare
two diffuse bounces to an exact finite cavity series and sphere/triangle-box
glass to a Fresnel/absorption series. They also exercise common closest hits,
nested media, inside starts, internal reflection, invalid overlaps, persistent
IDs, reset and ownership.

```bash
python -m pytest -q tests/test_transport_fields.py
ORDINARYLIGHT_TEST_VULKAN_TRANSPORT=1 python -m pytest -q tests/test_transport_gpu.py
```

The separately installable client renders diffuse cavity samples, SDF glass and
triangle glass using only these public APIs, including accumulation and HDR.

The resource/input extension was validated on the development RTX 4070 Laptop
GPU with 548 core tests (46 optional tests skipped) and 27 focused CPU/GPU/runtime
tests with GPU tests enabled. These include resource-backed distance fields,
GPU-written storage images, GPU-generated samples with input readback forbidden,
many-to-one reduction, in-place updates, lifetime guards, and the original
physical references. The scientific RT counts agree exactly across Vulkan GI, Vulkan Raster and
WebGPU. The likelihood viewer passes its 11 tests and native render/update smoke
tests on all three targets. The migrated client builds as a wheel and runs with
native GPU presentation.

## Downstream migration

The external transport client is updated to version 0.3.0 and exercises declared
resources, reusable GPU samples, and two-to-one surface reduction. It still
requires this unreleased OrdinaryLight checkout. OrdinaryShade, OrdinaryLattice,
OrdinaryScience, LatticeModel, LatticeVisualization and the scientific RT and
likelihood viewers had no callers of the changed low-level transport interfaces
in the inspected sources. Their current renderer APIs remain compatible; no
source adaptation was needed in those projects.

For other clients, retain existing NumPy construction or opt into the new input
and reduction objects. Close kernels before their buffers/images, and close
transport integrators before their borrowed sample allocations. Reset affected
output history explicitly after edits. Material/analytic-light expansion should
continue upstream using shared transport components and parity tests; this
extension does not claim full access to camera GI behavior.

### SDF entry-root regression coverage

CPU and GPU traversal distinguish a ray starting near a surface from an AABB-clipped
interval starting at that surface. This prevents floating-point rounding at a
clipped entry from skipping directly to the exit and corrupting nested-medium
tracking. Regression coverage includes fractional-radius spheres and nested
ideal/rough dielectric paths under both normal policies.

## Grouped independent boxes

`ordinarylight.geometry.BoxBatch` provides a general resource-backed custom
geometry reference. It packs arbitrary world-space axis-aligned boxes, active
flags, custom-material indices, boundary application IDs, and full uint32
application identities. There is no grid or voxel topology requirement.

```python
batch = BoxBatch(bounds, materials=material_ids, boundaries=boundary_ids,
                 identities=application_ids)
# Keep this allocation alive until the scene and its clients are closed.
records = runtime.buffer(batch.records.nbytes, data=batch.records)
scene = VulkanTransportScene(
    runtime, custom_geometry=[batch.geometry()], custom_materials=materials,
    custom_resources={batch.resource_name: records},
    media=media, boundaries=boundaries,
)
```

`batch.geometry(first, count)` encloses a contiguous record group, including
inactive records. Bind one shared record allocation while choosing per-box,
per-group or whole-batch primitives. The immutable host records are `(N, 3, 4)`
float32 storage: lower.xyz/active, upper.xyz/reserved, then **bitcast uint32**
material/boundary/identity/reserved. Default boundaries use `0xffffffff` for no
medium interface. Do not numerically convert the metadata to float; use a uint32
view. Custom-material indices are relative to the custom palette.

Upload changed contents or declare GPU producer writes through the execution
graph. Activation and metadata edits need no acceleration update within existing
bounds; geometry moving outside those bounds requires an explicit bounds update.
Reset affected lighting history after scene/material edits. Allocation growth
and producer dependencies retain the existing explicit resource contracts.

Each candidate scans its records, bounded by `max_steps`; active invalid bounds
or flags report traversal failure. Boxes are independent surfaces, not a solid
union: shared/internal faces are not removed, and ambiguous touching/overlapping
glass remains unsupported. No area sampler, smoothing, graph topology, or
spatial-index builder is implied. The external `chunk_probe` tests grouped hits
and graph-material radiance against per-box geometry, while analytic regression
tests separately validate entry/exit distances, normals and full-width IDs.
