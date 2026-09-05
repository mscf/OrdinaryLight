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
`ray_samples`, `surface_samples`, and diagnostic `intersect_rays`.

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
identities. Sample IDs select accumulator slots, independently of input order
or camera pixels; they must be unique within an input batch.

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
path. Additional application buffer bindings are not exposed by this initial
four-parameter callback interface.

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

## Dielectric semantics

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
last completion. Unique IDs per dispatch avoid float atomics. `reset()` clears
all records on the GPU; `reset(identities=[...])` clears selected records.
Applications decide which identities become invalid.

The 48-byte `ACCUMULATION_DTYPE` has these groups:

| Group | Components |
| --- | --- |
| radiance | RGB sum, reserved |
| counts | attempted, valid, ORed status flags, truncated samples |
| events | diffuse bounces, reflections, transmissions, total internal reflections |

Status bits are 1 unresolved/invalid intersection, 2 inconsistent boundaries,
4 stack overflow, 8 escape with an open medium, and 16 nonfinite transport.
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

Foundation validation on the development RTX 4070 Laptop GPU: 547 core tests
passed (42 optional tests skipped), all eight opt-in transport GPU tests passed,
and all 440 rebuilt camera shader binaries matched the existing binaries exactly.
The scientific RT viewer produced identical count volumes on Vulkan GI, Vulkan
Raster and WebGPU. The likelihood viewer passed its 11 tests and native smoke
tests on all three targets. The external client was built as a wheel, installed
under a separate prefix, run outside the checkout, and separately exercised with
native GPU presentation.
