# Application execution graphs

OrdinaryLight can execute an application-defined graph of GPU operations without
prescribing animation, voxelization, denoising, or presentation stages. Only nodes
added by the application execute. The first implementation uses one Vulkan queue,
whole-resource access tracking, conservative barriers and explicit resource
versions. Persistent allocations survive graph executions.

`VulkanPassPipeline` remains the ordered low-level API. `VulkanOperation` groups
recordable passes with validation and submission bookkeeping. `VulkanGraph`
orders operations and submits all their passes in one command buffer. Existing
transport/output convenience methods remain available and use these operations.

## Graph connections and versions

```python
from ordinarylight.pipeline.graph import VulkanGraph
from ordinarylight.pipeline.vulkan import VulkanResource

history = VulkanResource.buffer(accumulator.buffer)
graph = VulkanGraph()
graph.add("trace", integrator.accumulate_operation(),
          reads=[history.version(1)], writes=[history.version(2)])
graph.add("reset", accumulator.reset_operation(), writes=[history.version(1)])
graph.add("resolve", accumulator.resolve_operation())
graph.add("tone_map", tone_target.operation())
compiled = graph.compile()
completion = compiled.execute(runtime, after=[producer_completion])
```

Node insertion order is not execution order. Here reset precedes tracing even
though it was added later. Resource identity is the native allocation handle and
kind, so different binding names or `VulkanResource` wrappers cannot hide aliases.
Versions label successive values **in the same allocation**, not separate copies.

- Version zero is imported data, including previous-frame history.
- A sole writer defaults to version one; other implicit readers consume it.
- A read/write node implicitly reads the version preceding its output version.
- Multiple writers require distinct consecutive versions, or a complete explicit
  `after=[node_name]` ordering. Unordered writers are rejected.
- Readers of an older version precede a later overwrite, even when they use a
  different resource alias. Producer and write-order edges are also enforced.
- Missing producers, duplicate writers of a version, cycles, invalid connections
  and conflicting alias versions fail during compilation.

Use explicit versions when reading old data rather than a graph's new result.
Core attachment, vertex/index/indirect, shader and transfer accesses are classified,
as are acceleration accesses. Unknown extension access bits are rejected; raw
recorders can use conservative MEMORY_READ/WRITE declarations instead.
Whole-resource tracking intentionally serializes disjoint regions of one buffer;
suballocation/transient memory aliasing and region-level dependence analysis are
not implemented. Graph connections must agree with pass access declarations.

Compiled graphs can execute repeatedly with unchanged bindings and operation
shapes. Create a distinct operation instance per node. Graphs do not automatically
skip work based on revisions: applications decide whether to omit a node, reuse
a result, or clear history. Cross-frame feedback is imported version-zero data,
not a cycle inside one execution.

## Recordable operations

These methods construct operations without submitting them:

| Owner | Method |
| --- | --- |
| `VulkanTransportIntegrator` | `accumulate_operation(...)` |
| `GpuSampleAccumulator` | `reset_operation(...)`, `resolve_operation(...)` |
| `VulkanTransportScene` | `update_custom_geometry_operation(...)` |
| `VulkanToneMapTarget` | `operation(exposure=...)` |
| `VulkanOutput` | `present_operation(frame, surface_size=...)` |

Transport includes its sample-to-output reduction. Sampling epochs advance only
after successful submission; distinct transport operations within one submission
reserve separate epoch ranges. Recreate a transport operation after changing
active sample count or reduction group count. Parameter content updates with
unchanged dispatch shape can reuse operations, subject to synchronization.

An application can wrap any `VulkanPass` directly or construct
`VulkanOperation(passes, validate=..., dependencies=..., submitted=...)`.
Recorders must declare every accessed resource, record commands only, and avoid
submitting, closing or changing allocations. Validation happens before recording;
submission callbacks receive the completion token and should not raise. Built-in
operations use these hooks to track histories and resource revisions.

Operations retain Python references, but graphs do not take ownership of all
their allocations. Keep kernels/resources open until consumers finish. Closed
resources fail execution validation. Close descriptor consumers before their
allocations; closing native resources waits for outstanding GPU use where needed.

## OrdinaryShade reflection

```python
from ordinarylight.pipeline.graph import reflected_operation

operation = reflected_operation(
    kernel, compiled_shader.reflection,
    workgroups=(groups_x, 1, 1), push_constants=packed_parameters,
)
graph.add("populate", operation)
```

The kernel's bindings connect reflection to actual allocations. The adapter uses
OrdinaryShade's existing declared `read`, `write`, and `read_write` metadata for
whole-resource access; it does not claim new compiler effect inference. Aliased
bindings combine their accesses. Atomic buffers must declare read/write access.

The initial adapter supports the `VulkanKernel` descriptor family: set-zero
storage buffers, storage images, acceleration structures and push constants.
Uniform-buffer descriptors and sampled textures need a different kernel adapter;
they are rejected here. Raw shaders can use explicit pass declarations.
Reflection cannot establish application ordering intent, physical alias identity,
or absence of races within a shader invocation group.

## GPU ordering and bounded frame reuse

All runtime completion tokens refer to work already submitted on the same queue.
`after=[completion]` validates that relationship without waiting on the host.
Queue order plus GPU barriers provides execution/memory dependencies. Submission
polls completed fences to retire command storage without blocking. This is not
multi-queue or cross-device synchronization, and it does not require timeline
semaphores. Conservative barriers may serialize more work than necessary.

`VulkanFrameRing(runtime, frames=2)` bounds application frames in flight:

```python
with VulkanFrameRing(runtime, 2) as ring:
    for frame in animation:
        slot = ring.acquire()  # waits only if this slot is still in use
        # Use slot.index to select application-owned per-frame resources.
        completion = ring.submit(compiled_graph)
```

The ring owns completion tracking, not application allocations. If preparing a
frame fails, `ring.cancel()` releases its acquired CPU slot. Use separate slot
resources for data that the CPU updates while another frame executes. Shared
GPU-only resources can remain shared when queue dependencies order their uses.
CPU buffer uploads/readbacks, explicit capacity growth, shutdown and slot reuse
can still wait. The change removes host waits between dependent GPU stages;
it does not make CPU mutation of in-flight memory safe.

## Incremental custom acceleration

`VulkanTransportScene(custom_capacity=N, ...)` reserves custom primitive slots.
Unused slots retain valid Vulkan AABBs and disabled callback metadata. Activation
and removal therefore do not change Vulkan primitive counts or rely on changing
an inactive Vulkan primitive during an update.

```python
# A new or changed shape in slot 3; removal in slot 7.
operation = scene.update_custom_geometry_operation({3: shape, 7: None})
graph.add("update_geometry", operation)

# Convenience submission remains available:
scene.update_custom_geometry({3: changed_shape}, mode="refit")
```

Updates upload changed metadata/bounds with GPU transfer commands, then refit
the custom BLAS and combined TLAS. `mode="rebuild"` rebuilds in the existing
allocations; `"auto"` currently chooses refit for bound-bearing updates. Removal
only disables callback metadata and does not require a bounds update. Triangle
BLASes are reused. Bounds and metadata requests are host-declared; direct
GPU-generated acceleration build counts/bounds are not exposed in this milestone.

Resource-backed occupancy/field data can change within existing conservative
bounds with **no acceleration update**. The animated client uses this for its
fixed-grid slots. Callbacks remain responsible for honoring their declared bounds
and intersection guarantees. Voxelization and sparse-grid algorithms remain in
the application.

`reserve_custom_geometry(larger_capacity)` grows custom storage at an explicit
synchronized allocation boundary. It rebuilds the custom BLAS/combined TLAS while
preserving triangle, material, sample, accumulator and HDR resources. Live
integrators rebind their scene descriptors. Previously compiled graphs and
operations reject changed bindings and must be recreated. New shader programs,
material palettes or triangle source edits still require a replacement scene;
optional `intersection_programs` can register programs before activating slots.

`geometry_revision` advances on submitted slot changes and capacity growth;
`binding_revision` advances on allocation replacement. These describe scheduled
state: use the returned completion before CPU observation. They do not prove
lighting history is valid. Applications own nonlocal lighting invalidation and
must explicitly reset the affected output estimates.

## Persistent output and presentation

`output.prepare(hdr)` creates a `VulkanToneMapTarget` containing a persistent
RGBA8 image and kernel. Add `target.operation()` after HDR production. Its image
and completion can also be used for explicit final readback. One-shot
`tone_map()`/`export()` retain their owned-frame interface.

`output.present_operation(target)` acquires a swapchain image and returns a
single-use operation (or `None` if the surface must be recreated/is unavailable).
Add it to the graph before submission. It declares the GPU blit and layout
transition, waits on acquisition via a binary semaphore, then signals a
per-swapchain-image semaphore consumed by presentation. Two persistent acquisition
slots bound reuse. No per-frame tone-map allocation or queue-idle wait is needed
in steady state. Surface resize/recreation and shutdown can still wait.

Submit each acquired presentation operation exactly once, or call
`output.cancel_presentation()` if graph preparation fails. A compiled graph
containing an acquired presentation operation cannot be replayed: acquire a new
operation for the next frame. The convenience `present(hdr, after=...)` builds
this graph and caches its tone-map target. Prepared targets and that cache borrow
the HDR allocation: close the targets/output before closing HDR. The migrated
client uses this ownership order and keeps HDR stable across frames. A runtime
still owns one surface.

## External workload and compatibility

The separately installable [transport client](../examples/transport_client)
uses these APIs without private renderer imports. `ordinarylight-animated-grid`
animates occupancy with an OrdinaryShade shader, grows and activates/deactivates
chunk slots, traces, reduces, resolves, tone maps and optionally presents.
Only its final reference check and PNG export read GPU results. A two-frame ring
bounds execution; material/sample/HDR/output allocations remain persistent.

The scientific renderer constructors and `VulkanComputeSequence` remain
compatible. OrdinaryShade's reflection schema is unchanged, and OrdinaryLattice
retains its compiler-bridge role. The external client is migrated to graph APIs;
other inspected dependent projects do not call these new low-level interfaces.
The scientific viewers are checked on Vulkan GI, Vulkan Raster and WebGPU.

```bash
python -m pytest -q tests/test_vulkan_graph.py
ORDINARYLIGHT_TEST_VULKAN_GRAPH=1 python -m pytest -q tests/test_vulkan_graph.py
ordinarylight-animated-grid --frames 8 --output /tmp/grid.png
ordinarylight-animated-grid --present --frames 120 --output /tmp/grid.png
```

Validation on the development RTX 4070 Laptop GPU: the full suite passed 549
core tests, and 31 focused tests with GPU execution enabled passed. GPU tests
exercised graph ordering, live OrdinaryShade
reflection, no-wait completion dependencies, activation/refit/rebuild/growth,
persistent output, native resize and presentation cancellation. The animated
client passed its exact final-radiance reference and ran from an installed wheel
outside the checkout. Scientific RT volumes agree exactly across all three
targets; the likelihood viewer passed its 11 tests and native smoke tests on all
three targets. Shader packaging reports no missing or unmanaged outputs.
