"""Persistent public AABB BLAS/TLAS resources and same-queue build operations.

Construction allocates storage; operation() records builds without allocations,
readbacks or CPU waits. Inputs and referenced BLASes are leased until close.
"""
from dataclasses import dataclass
from contextlib import ExitStack
from operator import index
import struct
import math
import vulkan as vk
from ..pipeline.vulkan import VulkanResource, VulkanResourceUse, VulkanPass
from ..pipeline.graph import VulkanOperation

_BUILD = vk.VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR
_READ = vk.VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR
_WRITE = vk.VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR
_INPUT = vk.VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR


def _view(value, runtime):
    result = value if isinstance(value, VulkanResource) else VulkanResource.buffer(value)
    if result.kind != 'buffer' or result.owner.runtime is not runtime:
        raise ValueError('Acceleration inputs must be same-runtime buffer views')
    result.owner.require_open()
    if not result.owner.usage & _INPUT or not result.owner.usage & 0x00020000:
        raise ValueError('Acceleration input requires build-input and device-address usage')
    return result


def _address(runtime, view):
    info = vk.VkBufferDeviceAddressInfo(buffer=view.handle)
    return int(runtime._raw_buffer_address(runtime.device, vk.ffi.addressof(info))) + view.offset


class _Acceleration:
    def _initialize(self, runtime, geometry, count, kind, source, dependencies=(), allow_update=True):
        self.runtime, self.source = runtime, source
        self.geometry, self.primitive_count, self.kind = geometry, count, kind
        self.allow_update = bool(allow_update)
        self.closed, self.built = False, False
        self.binding_revision = 0
        self.last_completion = None
        self._borrowers = set()
        self._resources = ExitStack()
        self.handle = None
        self._dependencies = tuple(dict.fromkeys(dependencies))
        runtime.require_open()
        runtime.retain(self)
        try:
            source.owner.retain(self)
            self._resources.callback(source.owner.release, self)
            for owner in self._dependencies:
                owner.require_open()
                if owner.runtime is not runtime:
                    raise ValueError('BLAS instances must share the TLAS runtime')
                owner.retain(self)
                self._resources.callback(owner.release, self)
            self.flags = vk.VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR
            if allow_update:
                self.flags |= vk.VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR
            build = vk.VkAccelerationStructureBuildGeometryInfoKHR(type=kind, flags=self.flags,
                mode=vk.VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR,
                geometryCount=1, pGeometries=[geometry])
            sizes = vk.VkAccelerationStructureBuildSizesInfoKHR()
            runtime.get_as_sizes(runtime.device, vk.VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
                                 build, [count], sizes)
            self.storage = self._resources.enter_context(runtime.buffer(int(sizes.accelerationStructureSize),
                memory='device', usage=vk.VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR))
            self.handle = runtime.create_as(runtime.device, vk.VkAccelerationStructureCreateInfoKHR(
                buffer=self.storage.buffer, size=sizes.accelerationStructureSize, type=kind), None)
            # Query scratch alignment instead of relying on allocator coincidence.
            properties = vk.VkPhysicalDeviceAccelerationStructurePropertiesKHR()
            vk.vkGetPhysicalDeviceProperties2(runtime.physical_device,
                vk.VkPhysicalDeviceProperties2(pNext=properties))
            alignment = int(properties.minAccelerationStructureScratchOffsetAlignment)
            self.scratch = self._resources.enter_context(runtime.buffer(
                max(int(sizes.buildScratchSize), int(sizes.updateScratchSize), 1) + alignment,
                memory='device', device_address=True))
            base = _address(runtime, VulkanResource.buffer(self.scratch))
            self._scratch_address = ((base + alignment - 1) // alignment) * alignment
        except BaseException:
            if self.handle is not None:
                runtime.destroy_as(runtime.device, self.handle, None)
            self._resources.close()
            runtime.release(self)
            self.closed = True
            raise

    @property
    def resource(self):
        self.require_open()
        return VulkanResource(self, 'acceleration_structure', self.handle)

    @property
    def device_address(self):
        self.require_open()
        info = vk.VkAccelerationStructureDeviceAddressInfoKHR(accelerationStructure=self.handle)
        return int(self.runtime._raw_as_address(self.runtime.device, vk.ffi.addressof(info)))

    def require_open(self):
        self.runtime.require_open()
        if self.closed:
            raise RuntimeError('Acceleration structure is closed')
        self.source.owner.require_open()

    def retain(self, consumer):
        self.require_open()
        self._borrowers.add(consumer)

    def release(self, consumer):
        self._borrowers.discard(consumer)

    def operation(self, *, mode='rebuild'):
        """Build/refit the fixed primitive set; geometry writes must precede it.

        Refit requires an earlier submitted build and unchanged primitive count,
        layout and active/inactive classification. Use rebuild when those Vulkan
        update constraints are not met. Range writes do not make a BLAS refit
        partial: use independent small BLASes to bound acceleration work.
        """
        self.require_open()
        if mode not in ('rebuild', 'refit'):
            raise ValueError('Expected rebuild or refit')
        if mode == 'refit' and (not self.allow_update or not self.built):
            raise ValueError('Refit requires a submitted update-enabled build')
        build = vk.VkAccelerationStructureBuildGeometryInfoKHR(type=self.kind, flags=self.flags,
            mode=(vk.VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR if mode == 'refit'
                  else vk.VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR),
            srcAccelerationStructure=self.handle if mode == 'refit' else vk.VK_NULL_HANDLE,
            dstAccelerationStructure=self.handle, geometryCount=1, pGeometries=[self.geometry],
            scratchData=vk.VkDeviceOrHostAddressKHR(deviceAddress=self._scratch_address))
        item = vk.VkAccelerationStructureBuildRangeInfoKHR(primitiveCount=self.primitive_count)
        ranges = vk.ffi.new('VkAccelerationStructureBuildRangeInfoKHR*[]', [vk.ffi.addressof(item)])
        uses = [VulkanResourceUse(self.source, _BUILD, _READ),
                VulkanResourceUse(self.resource, _BUILD, _READ | _WRITE),
                VulkanResourceUse(VulkanResource.buffer(self.scratch), _BUILD, _READ | _WRITE)]
        uses += [VulkanResourceUse(owner.resource, _BUILD, _READ) for owner in self._dependencies]
        def record(command, _range_owner=item):
            self.runtime.build_as(command, 1, [build], ranges)
        def submitted(completion):
            self.built = True
            self.last_completion = completion
        return VulkanOperation([VulkanPass('acceleration_' + mode, tuple(uses), record)],
            validate=self.require_open, submitted=submitted,
            dependencies=lambda: (() if self.last_completion is None else (self.last_completion,)))

    def close(self):
        with self.runtime.lock:
            if self.closed:
                return
            if self._borrowers:
                raise RuntimeError('Close acceleration borrowers before their owner')
            from .resources import VulkanCompletion
            for completion in tuple(self.runtime._consumers):
                if isinstance(completion, VulkanCompletion) and self in completion.resources:
                    completion.wait()
            if self.last_completion is not None:
                self.last_completion.wait()
            self.runtime.destroy_as(self.runtime.device, self.handle, None)
            self._resources.close()
            self.closed = True
            self.runtime.release(self)

    def __enter__(self):
        self.require_open()
        return self

    def __exit__(self, *_exc):
        self.close()


class VulkanAabbBlas(_Acceleration):
    """Borrow count AABBs: six float32 values per record, with optional stride."""
    def __init__(self, runtime, bounds, count, *, stride=24, allow_update=True):
        count, stride = index(count), index(stride)
        source = _view(bounds, runtime)
        if count < 1 or stride < 24 or stride % 8 or source.offset % 8 or (count-1)*stride+24 > source.size:
            raise ValueError('Invalid AABB count, stride, alignment or input range')
        geometry = vk.VkAccelerationStructureGeometryKHR(geometryType=vk.VK_GEOMETRY_TYPE_AABBS_KHR,
            geometry=vk.VkAccelerationStructureGeometryDataKHR(aabbs=vk.VkAccelerationStructureGeometryAabbsDataKHR(
                data=vk.VkDeviceOrHostAddressConstKHR(deviceAddress=_address(runtime, source)), stride=stride)))
        self._initialize(runtime, geometry, count, vk.VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR,
                         source, allow_update=allow_update)


@dataclass(frozen=True)
class VulkanBlasInstance:
    blas: VulkanAabbBlas
    custom_index: int = 0
    mask: int = 255
    transform: tuple = (1.,0.,0.,0., 0.,1.,0.,0., 0.,0.,1.,0.)

    def pack(self):
        if not isinstance(self.blas, VulkanAabbBlas):
            raise TypeError("Instances require an AABB BLAS")
        custom, mask = index(self.custom_index), index(self.mask)
        values = tuple(float(v) for v in self.transform)
        if not 0 <= custom < 2**24 or not 0 <= mask < 256 or len(values) != 12 or not all(map(math.isfinite, values)):
            raise ValueError('Invalid instance custom index, visibility mask or 3x4 transform')
        return struct.pack('<12fIIQ', *values, custom | (mask << 24), 0, self.blas.device_address)


class VulkanTlas(_Acceleration):
    """Borrow a GPU VkAccelerationStructureInstanceKHR array and referenced BLASes.

    Input records are 64 bytes each. referenced_blas must include every structure
    whose address can appear in the array; they are leased for this TLAS lifetime.
    The application writes valid transforms/indices/addresses before operation().
    """
    def __init__(self, runtime, instances, count, *, referenced_blas, allow_update=True):
        source, count = _view(instances, runtime), index(count)
        if count < 1 or source.offset % 16 or count * 64 > source.size:
            raise ValueError('Invalid TLAS instance count, alignment or input range')
        geometry = vk.VkAccelerationStructureGeometryKHR(geometryType=vk.VK_GEOMETRY_TYPE_INSTANCES_KHR,
            geometry=vk.VkAccelerationStructureGeometryDataKHR(instances=vk.VkAccelerationStructureGeometryInstancesDataKHR(
                arrayOfPointers=vk.VK_FALSE,
                data=vk.VkDeviceOrHostAddressConstKHR(deviceAddress=_address(runtime, source)))))
        self._initialize(runtime, geometry, count, vk.VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR,
                         source, tuple(referenced_blas), allow_update=allow_update)


def buffer_copy_operation(source, destination, regions):
    """Copy explicit (source offset, destination offset, byte count) GPU ranges.

    Offsets are relative to the supplied buffer views. No allocation, upload,
    readback or CPU wait occurs. Ranges must be disjoint in the destination;
    in-place copies are rejected. Place the operation before dependent AS builds.
    """
    src = source if isinstance(source, VulkanResource) else VulkanResource.buffer(source)
    dst = destination if isinstance(destination, VulkanResource) else VulkanResource.buffer(destination)
    if src.kind != 'buffer' or dst.kind != 'buffer' or src.owner.runtime is not dst.owner.runtime:
        raise ValueError('Copy requires same-runtime buffer views')
    if src.handle == dst.handle:
        raise ValueError('In-place range copies are not supported')
    if not src.owner.usage & vk.VK_BUFFER_USAGE_TRANSFER_SRC_BIT or not dst.owner.usage & vk.VK_BUFFER_USAGE_TRANSFER_DST_BIT:
        raise ValueError('Copy buffers require transfer usage')
    ranges = tuple(tuple(map(index, r)) for r in regions)
    if not ranges:
        raise ValueError('At least one copy range is required')
    for i, (a,b,size) in enumerate(ranges):
        if min(a,b) < 0 or size < 1 or a+size > src.size or b+size > dst.size:
            raise ValueError('Copy range exceeds a buffer view')
        if any(max(b, y) < min(b+size, y+n) for _,y,n in ranges[:i]):
            raise ValueError('Destination copy ranges overlap')
    copies = [vk.VkBufferCopy(srcOffset=src.offset+a, dstOffset=dst.offset+b, size=n) for a,b,n in ranges]
    def validate():
        src.owner.require_open()
        dst.owner.require_open()
    def record(command):
        vk.vkCmdCopyBuffer(command, src.handle, dst.handle, len(copies), copies)
    return VulkanOperation([VulkanPass('copy_buffer_ranges', (
        VulkanResourceUse(src, vk.VK_PIPELINE_STAGE_TRANSFER_BIT, vk.VK_ACCESS_TRANSFER_READ_BIT),
        VulkanResourceUse(dst, vk.VK_PIPELINE_STAGE_TRANSFER_BIT, vk.VK_ACCESS_TRANSFER_WRITE_BIT),
    ), record)], validate=validate)
