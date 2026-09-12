# Public persistent acceleration

Import `VulkanAabbBlas`, `VulkanTlas`, `VulkanBlasInstance` and
`buffer_copy_operation` from `ordinarylight.runtime`.

A BLAS borrows a same-runtime buffer or `VulkanResource.byte_range` containing
six float32 values per AABB (minimum XYZ, maximum XYZ). The buffer must have
`VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR` and be
allocated with `device_address=True`. A stride >=24 and divisible by 8 is allowed.
A TLAS borrows an array of 64-byte Vulkan instance records; pass every BLAS whose
address can appear in those records through `referenced_blas`. The application
must supply valid instance transforms, indices and addresses. `VulkanBlasInstance`
packs one host-authored record; GPU-authored records use the same Vulkan ABI.

```python
from ordinarylight.runtime import VulkanAabbBlas, VulkanTlas, VulkanBlasInstance
from ordinarylight.pipeline.graph import VulkanGraph

# bounds and instance_buffer are persistent, addressable build-input buffers.
blas = VulkanAabbBlas(runtime, bounds, primitive_count)
tlas = VulkanTlas(runtime, instance_buffer, instance_count,
                  referenced_blas=(blas,))
graph = VulkanGraph()
graph.add('populate', application_bounds_and_instances_operation)
graph.add('blas', blas.operation())
graph.add('tlas', tlas.operation())
# Compose GI using a scene imported with acceleration=tlas.resource.
```

Construction allocates AS storage and aligned scratch but does not build the AS.
`operation(mode='rebuild')` records its first build. After successful submission,
`mode='refit'` is available for update-enabled objects. Neither operation allocates
GPU resources or inserts CPU waits. Graph resource barriers order source writes,
BLAS builds, TLAS builds and ray queries on the runtime's queue. A build operation
must precede the first query. Explicit dependencies are needed if an application's
external recorder does not declare the relevant resource hazards.

Primitive counts and layouts are fixed for each object. Vulkan refit also requires
unchanged active/inactive primitive classification; use rebuild when uncertain.
Finite tiny AABBs can represent reserved empty records if application intersection
callbacks reject them. Updating part of a large BLAS still refits that BLAS;
use smaller independent structures for bounded updates.

`buffer_copy_operation(source, destination, [(src_offset, dst_offset, size), ...])`
copies GPU byte ranges relative to supplied views. It rejects out-of-bounds,
overlapping destination and in-place copies. There is no upload, readback or
CPU wait; put the operation before its dependent build/refit in the graph.

`structure.resource` is a borrowed acceleration resource for kernels and
`runtime.import_scene`. Instance custom indices are 24-bit application values;
hardware instance IDs are separate and depend on TLAS record order. Applications
must maintain stable semantic instance/slot/sub-element identities themselves.

Close imported scenes/kernels and TLAS borrowers first, then TLAS, BLASes and
input buffers. Structures lease inputs/referenced BLASes and reject premature
close while borrowed. Close retires pending GPU uses; it is an explicit lifetime
boundary, not a render stage. Handle/count replacement requires new objects and
rebinding consumers; mutating GPU input content preserves handles.
