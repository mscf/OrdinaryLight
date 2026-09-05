# Non-camera transport foundations

OrdinaryLight provides a reusable Vulkan path tracer for application-selected
rays and surface samples. It handles traversal, diffuse continuation, dielectric
boundaries, absorption, and accumulation without constructing the camera GI
renderer. The [external client package](../examples/transport_client) exercises
this API using only public imports, without implementing its own transport loop.

The initial material set is Lambertian diffuse and ideal dielectric. Illumination
is constant environment radiance plus emissive geometry sampled by BSDF
continuation. Analytic-light NEE/MIS, rough glass, textures, and participating
scattering are not implemented in this path. Unsupported scene features raise.
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
the surface outside its AABB, replace the scene snapshot. Applications must reset
affected accumulated outputs when the expected lighting changes. Field uploads
do not perform bounds refits or history invalidation automatically.

## Reusable samples and reduction

Existing construction with NumPy `ray_samples`/`surface_samples` still works.
`integrator.update_samples(new_samples, reduction=..., after=...)` uploads new
records into the existing allocation without recompiling kernels. The active
count can change within its capacity; growing capacity requires new inputs and
an integrator. Updates preserve history and the sampling epoch deliberately.

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
as stable groups; GPU-generated grouping and weighted reductions are deferred.
Each input contributes equal weight: output radiance is total radiance divided
by total valid path count, not a sum of already averaged face colors.
Attempted/valid/truncated counts and events are summed; status flags are ORed.

`set_reduction(mapping, after=...)` changes grouping without rebuilding pipelines.
After `inputs.set_count(...)` or a count-changing `inputs.update(...)`, set a
matching reduction before dispatch. A direct `inputs.update(...)` with unchanged
count preserves the current map. In contrast, `integrator.update_samples(...)`
without a reduction uses the new host records' output IDs, matching initial
NumPy construction. Reset affected accumulator IDs explicitly when changing the
quantity being estimated; updates never silently erase application history.

Close integrators before borrowed inputs. GPU sample generation still needs a
known host dispatch count; indirect dispatch and GPU-discovered reduction maps
are not introduced here. Reduction adds one GPU pass and scratch record per
input; it avoids float atomics and CPU result copies.

## Dielectric semantics

The scattering-normal policy remains geometric for ideal dielectrics: both
boundary classification and Fresnel/refraction/reflection use the outward
geometric normal. Diffuse scattering uses the shading frame with a geometric
hemisphere check. A GPU parity test verifies that tilted shading normals leave
dielectric paths unchanged. Intersecting a smooth SDF supplies a smooth geometric
normal; using a smooth optical normal on blocky intersections is not supported.
That future mode needs an explicit wrong-hemisphere/throughput policy, rather
than substituting the normal in the current equations.

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
| radiance | RGB sum, reserved |
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

Transport scenes are immutable snapshots. They may borrow an existing resident
scene's triangle buffers/BLASes and own their combined TLAS/custom AABB resources.
Source scene mutation requires a replacement snapshot. Integrators borrow scenes
and accumulators: close integrators first, then resources, then the runtime.
The runtime lock serializes host state and queue/pool use.

The runtime/pass layer owns completion dependencies, future public scene updates
and refits, and presentation allocations. Current dependencies still use
conservative barriers and host waits; timeline scheduling and incremental refits
are not part of this milestone. Accumulation/HDR allocations are persistent;
output allocation behavior follows the existing
[extension interface](renderer_extensions.md). Applications own pass ordering
and invalidation policy. Voxel identities and artistic averaging remain outside
the renderer core.

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

The external transport client is updated to version 0.2.0 and exercises declared
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
