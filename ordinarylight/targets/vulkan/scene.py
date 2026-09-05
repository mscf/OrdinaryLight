"""Resident scene upload shared by GI and application rendering algorithms."""
import struct
import hashlib
import time
from types import MappingProxyType

import numpy as np
import vulkan as vk

BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT = 0x00020000
MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT = 0x00000002
MAX_NATIVE_TEXTURES = 64
MAX_NATIVE_VOLUMES = 16

class Buffer:
    def __init__(self, buffer, memory, size):
        self.buffer = buffer
        self.memory = memory
        self.size = size


class AccelerationStructure:
    def __init__(self, handle, storage, scratch=None):
        self.handle = handle
        self.storage = storage
        self.scratch = scratch


class SceneBlas:
    """One refittable BLAS shared by instances of the same mesh resource."""

    def __init__(self, structure, mesh, vertex_buffer):
        self.structure = structure
        self.mesh = mesh
        self.vertex_buffer = vertex_buffer
        self.indices = mesh.indices.copy()
        self.vertices = mesh.vertices.copy()


class SceneTlasInstance:
    """One TLAS placement and its global packed-triangle offset."""

    def __init__(self, mesh, blas, triangle_offset):
        self.mesh = mesh
        self.blas = blas
        self.triangle_offset = triangle_offset
        # Volume bounds are ray-entry proxies, not opaque surfaces.  Keep them
        # in a separate visibility group so path rays can enter volumes while
        # binary surface-shadow queries do not mistake the proxy box for an
        # occluder.
        self.visibility_mask = (
            0x02 if mesh.metadata.get("volume_index") is not None else 0x01
        )


class SampledTexture:
    def __init__(self, image, memory, view, sampler):
        self.image = image
        self.memory = memory
        self.view = view
        self.sampler = sampler


class VulkanSceneResources:
    """Owns one uploaded scene's buffers and acceleration structures."""

    def __init__(self, core, scene, *, config=None):
        from ...runtime.vulkan import VulkanRuntime
        if isinstance(core, VulkanRuntime):
            core = VulkanSceneUploader(core, config=config)
        elif config is not None:
            raise ValueError("config belongs to the scene uploader")
        core.runtime.require_open()
        self.runtime = core.runtime
        self._core = core
        self._borrowers = set()
        self.scene = scene
        self.scene_revision = scene.revision
        self.geometry_revision = scene.geometry_revision
        self.shading_revision = scene.shading_revision
        self.transform_revision = scene.transform_revision
        self.previous_transform_revision = scene.transform_revision
        self.texture_signature = tuple(id(item) for item in scene.textures)
        self._update_content_signatures()
        buffer_start = len(core._buffers)
        structure_start = len(core._structures)
        texture_start = len(core._sampled_textures)
        try:
            self.tlas = core._build_scene(scene)
        except Exception:
            core._release_resources(
                core._structures[structure_start:], core._buffers[buffer_start:]
            )
            core._release_sampled_textures(core._sampled_textures[texture_start:])
            del core._structures[structure_start:]
            del core._buffers[buffer_start:]
            del core._sampled_textures[texture_start:]
            raise
        self.vertex_buffer = core.scene_vertex_buffer
        self.previous_vertex_buffer = core.scene_previous_vertex_buffer
        positions = scene.render_triangles().reshape((-1, 3))
        self.vertex_data = np.ascontiguousarray(
            np.column_stack((positions, np.ones(len(positions), np.float32))),
            dtype=np.float32,
        )
        self.material_buffer = core.scene_material_buffer
        self.light_buffer = core.scene_light_buffer
        self.area_light_buffer = core.scene_area_light_buffer
        self.attribute_buffer = core.scene_attribute_buffer
        self.custom_attribute_buffer = core.scene_custom_attribute_buffer
        self.custom_attribute_layout = core.scene_custom_attribute_layout
        self.texture_buffer = core.scene_texture_buffer
        self.texture_binding_buffer = core.scene_texture_binding_buffer
        self.volume_header_buffer = core.scene_volume_header_buffer
        self.volume_scalar_buffer = core.scene_volume_scalar_buffer
        self.volume_transfer_buffer = core.scene_volume_transfer_buffer
        self.triangle_volume_buffer = core.scene_triangle_volume_buffer
        self.volume_empty_space_skipping = (
            core.scene_volume_empty_space_skipping
        )
        self.sampled_textures = tuple(core._sampled_textures[texture_start:])
        self.scene_sampled_textures = tuple(core.scene_sampled_textures)
        self.scene_sampled_volumes = tuple(core.scene_sampled_volumes)
        self.volume_signature = tuple(
            (id(volume), volume.shape, volume.visible)
            for volume in scene.volumes
        )
        self.volume_data_revisions = tuple(
            volume.data_revision for volume in scene.visible_volumes
        )
        self.volume_dirty_counts = tuple(
            len(volume.dirty_regions) for volume in scene.visible_volumes
        )
        self.volume_gpu_revisions = tuple(
            getattr(volume.gpu_source, "revision", None)
            for volume in scene.visible_volumes
        )
        self.blases = tuple(core.scene_blases)
        self.instances = tuple(core.scene_instances)
        self.instance_buffer = core.scene_instance_buffer
        self._structures = core._structures[structure_start:]
        self._buffers = core._buffers[buffer_start:]
        self._sampled_textures = core._sampled_textures[texture_start:]
        del core._structures[structure_start:]
        del core._buffers[buffer_start:]
        del core._sampled_textures[texture_start:]
        self.runtime.retain(self)
        # Packed primitive order is the same as the renderer's ID products.
        self.primitive_ids = np.array(scene.triangle_instance_ids(), copy=True)
        self.primitive_ids.flags.writeable = False

    def _update_content_signatures(self):
        # Scene's shading revision groups materials and lights. Separate content
        # signatures let application history invalidate either domain precisely.
        programs, default = self._core._ensure_scene_pipeline(self.scene)
        material = hashlib.sha256(self.scene.triangle_material_data(programs, default).tobytes())
        material.update(self.scene.texture_binding_data().tobytes())
        lighting = hashlib.sha256(self.scene.analytic_light_data().tobytes())
        lighting.update(self.scene.emissive_triangle_data().tobytes())
        for texture in self.scene.textures:
            pixels = texture.pixels.tobytes()
            material.update(pixels)
            # Includes environment maps; conservatively invalidate lighting for
            # texture changes without guessing which shader samples the atlas.
            lighting.update(pixels)
        self.content_signatures = MappingProxyType({
            "materials": material.hexdigest(), "lighting": lighting.hexdigest(),
        })

    @property
    def revisions(self):
        self.require_open()
        return MappingProxyType(dict(scene=self.scene_revision,
                                     geometry=self.geometry_revision,
                                     shading=self.shading_revision,
                                     transform=self.transform_revision))

    @property
    def bindings(self):
        """Borrow immutable handles until close; buffer packing is scene ABI v1."""
        self.require_open()
        names = ("vertex", "previous_vertex", "material", "light", "area_light",
                 "attribute", "custom_attribute", "texture", "texture_binding",
                 "volume_header", "volume_scalar", "volume_transfer", "triangle_volume")
        return MappingProxyType({name: getattr(self, name + "_buffer") for name in names})

    def resource(self, name):
        """Typed borrowed pass binding; never free its native handle directly."""
        from ...pipeline.vulkan import VulkanResource
        self.require_open()
        if name == "tlas":
            return VulkanResource(self, "acceleration_structure", self.tlas.handle)
        buffer = self.bindings[name]
        if buffer is None:
            raise ValueError(f"Scene does not have a {name} buffer")
        return VulkanResource(self, "buffer", buffer.buffer, buffer.size)

    def require_open(self):
        if self._core is None:
            raise RuntimeError("Vulkan scene resources are closed")
        self.runtime.require_open()

    def __enter__(self):
        self.require_open()
        return self

    def __exit__(self, *_exc):
        self.close()

    def close(self):
        if self._core is None:
            return
        if self._borrowers:
            raise RuntimeError("Unbind scene consumers before closing resident scene resources")
        vk.vkDeviceWaitIdle(self.runtime.device)
        self._core._release_resources(self._structures, self._buffers)
        self._core._release_sampled_textures(self._sampled_textures)
        self._structures.clear()
        self._buffers.clear()
        self._sampled_textures.clear()
        self._core = None
        self.runtime.release(self)


class VulkanSceneUploader:
    """Shared triangle/texture/volume upload implementation, without GI pipelines."""

    def __init__(self, runtime, *, config=None):
        runtime.require_open()
        self.runtime = runtime
        self.config = config or runtime.config
        if self.config.wavefront_native_textures and not runtime.config.wavefront_native_textures:
            raise ValueError("Runtime did not enable native textures")
        self._buffers = []
        self._structures = []
        self._sampled_textures = []
        self.last_timings = {}

    def __getattr__(self, name):
        return getattr(self.runtime, name)

    @property
    def native_textures_enabled(self):
        return self.config.wavefront_native_textures and self.runtime.native_textures_supported

    def _ensure_scene_pipeline(self, scene):
        from ...materials import builtin_material
        default = self.config.material_program or builtin_material
        return scene.material_programs(default), default

    @staticmethod
    def _material_attribute_layout(scene, programs, material_modifier=None):
        """Return the opt-in custom vertex ABI required by material programs."""
        requirements = {}
        ordered_names = []
        for program in programs:
            for name, components in program.required_attributes:
                previous = requirements.get(name)
                if previous is not None and previous != components:
                    raise ValueError(
                        f"attribute {name!r} has conflicting material declarations"
                    )
                if previous is None:
                    ordered_names.append(name)
                requirements[name] = components
        from ...scene import VertexAttributeLayout
        if not ordered_names:
            from ...materials import builtin_material
            return (
                None if (
                    all(program is builtin_material for program in programs)
                    and material_modifier is None
                )
                else VertexAttributeLayout(())
            )
        layout = VertexAttributeLayout.from_scene(scene, ordered_names)
        for name, components in layout.channels:
            if requirements[name] != components:
                raise ValueError(
                    f"mesh attribute {name!r} has {components} components; "
                    f"material requires {requirements[name]}"
                )
        return layout

    def _memory_type(self, bits, flags):
        for index in range(self.memory_properties.memoryTypeCount):
            memory_type = self.memory_properties.memoryTypes[index]
            if bits & (1 << index) and memory_type.propertyFlags & flags == flags:
                return index
        raise RuntimeError(f"No Vulkan memory type satisfies flags {flags:#x}")

    def _create_buffer(self, size, usage, memory_flags, data=None, device_address=False):
        buffer = vk.vkCreateBuffer(
            self.device,
            vk.VkBufferCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                size=size,
                usage=usage,
                sharingMode=vk.VK_SHARING_MODE_EXCLUSIVE,
            ),
            None,
        )
        requirements = vk.vkGetBufferMemoryRequirements(self.device, buffer)
        allocation_flags = None
        if device_address:
            allocation_flags = vk.VkMemoryAllocateFlagsInfo(
                sType=vk.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO,
                flags=MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT,
            )
        try:
            memory = vk.vkAllocateMemory(
                self.device,
                vk.VkMemoryAllocateInfo(
                    sType=vk.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                    pNext=allocation_flags,
                    allocationSize=requirements.size,
                    memoryTypeIndex=self._memory_type(
                        requirements.memoryTypeBits, memory_flags),
                ),
                None,
            )
        except vk.VkErrorOutOfDeviceMemory:
            vk.vkDestroyBuffer(self.device, buffer, None)
            raise
        vk.vkBindBufferMemory(self.device, buffer, memory, 0)
        result = Buffer(buffer, memory, size)
        self._buffers.append(result)
        if data is not None:
            payload = memoryview(data).cast("B") if not isinstance(data, bytes) else data
            mapped = vk.vkMapMemory(self.device, memory, 0, len(payload), 0)
            mapped[:] = payload
            vk.vkUnmapMemory(self.device, memory)
        return result

    def _create_exportable_buffer(self, size, usage, memory_flags):
        """Create a dedicated opaque-FD exportable buffer allocation."""
        external_info = vk.VkExternalMemoryBufferCreateInfo(
            handleTypes=vk.VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT,
        )
        buffer = vk.vkCreateBuffer(
            self.device,
            vk.VkBufferCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                pNext=external_info,
                size=size, usage=usage,
                sharingMode=vk.VK_SHARING_MODE_EXCLUSIVE,
            ), None,
        )
        requirements = vk.vkGetBufferMemoryRequirements(self.device, buffer)
        export_info = vk.VkExportMemoryAllocateInfo(
            handleTypes=vk.VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT,
        )
        dedicated_info = vk.VkMemoryDedicatedAllocateInfo(
            pNext=export_info, image=vk.VK_NULL_HANDLE, buffer=buffer,
        )
        try:
            memory = vk.vkAllocateMemory(
                self.device,
                vk.VkMemoryAllocateInfo(
                    sType=vk.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                    pNext=dedicated_info,
                    allocationSize=requirements.size,
                    memoryTypeIndex=self._memory_type(
                        requirements.memoryTypeBits, memory_flags,
                    ),
                ), None,
            )
        except Exception:
            vk.vkDestroyBuffer(self.device, buffer, None)
            raise
        vk.vkBindBufferMemory(self.device, buffer, memory, 0)
        result = Buffer(buffer, memory, int(requirements.size))
        self._buffers.append(result)
        return result

    def _create_uploaded_device_buffer(
        self, data, usage, *, device_address=False,
    ):
        """Stage immutable data into shader-fast device-local storage."""
        payload = np.ascontiguousarray(data)
        staging = self._create_buffer(
            payload.nbytes,
            vk.VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
            | vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            data=payload,
        )
        destination = self._create_buffer(
            payload.nbytes,
            usage | vk.VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            device_address=device_address,
        )
        self._single_use(lambda command: vk.vkCmdCopyBuffer(
            command, staging.buffer, destination.buffer, 1,
            [vk.VkBufferCopy(srcOffset=0, dstOffset=0, size=payload.nbytes)],
        ))
        vk.vkDestroyBuffer(self.device, staging.buffer, None)
        vk.vkFreeMemory(self.device, staging.memory, None)
        self._buffers.remove(staging)
        return destination

    def _update_device_buffers(self, updates):
        """Replace equal-sized device-local buffer contents in one submission."""
        prepared = []
        for destination, data in updates:
            payload = np.ascontiguousarray(data)
            if payload.nbytes != destination.size:
                raise ValueError("updated buffer data must retain its byte size")
            staging = self._create_buffer(
                payload.nbytes, vk.VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
                | vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                data=payload,
            )
            prepared.append((staging, destination, payload.nbytes))
        try:
            self._single_use(lambda command: [
                vk.vkCmdCopyBuffer(
                    command, staging.buffer, destination.buffer, 1,
                    [vk.VkBufferCopy(srcOffset=0, dstOffset=0, size=size)],
                )
                for staging, destination, size in prepared
            ])
        finally:
            for staging, _destination, _size in prepared:
                vk.vkDestroyBuffer(self.device, staging.buffer, None)
                vk.vkFreeMemory(self.device, staging.memory, None)
                self._buffers.remove(staging)

    def _update_device_buffer_regions(self, destination, regions):
        """Replace byte ranges of one device-local buffer in one submission."""
        prepared = []
        for offset, data in regions:
            payload = np.ascontiguousarray(data)
            offset = int(offset)
            if offset < 0 or offset + payload.nbytes > destination.size:
                raise ValueError("updated buffer region lies outside destination")
            staging = self._create_buffer(
                payload.nbytes, vk.VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
                | vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                data=payload,
            )
            prepared.append((staging, offset, payload.nbytes))
        try:
            self._single_use(lambda command: [
                vk.vkCmdCopyBuffer(
                    command, staging.buffer, destination.buffer, 1,
                    [vk.VkBufferCopy(
                        srcOffset=0, dstOffset=offset, size=size,
                    )],
                )
                for staging, offset, size in prepared
            ])
        finally:
            for staging, _offset, _size in prepared:
                vk.vkDestroyBuffer(self.device, staging.buffer, None)
                vk.vkFreeMemory(self.device, staging.memory, None)
                self._buffers.remove(staging)

    def _update_sampled_volume_regions(self, texture, volume, regions):
        """Upload z/y/x boxes into an existing shader-readable 3-D image."""
        prepared = []
        scalar_data = volume.data
        for offset, shape in regions:
            stop = tuple(start + size for start, size in zip(offset, shape))
            payload = np.ascontiguousarray(scalar_data[
                offset[0]:stop[0], offset[1]:stop[1], offset[2]:stop[2]
            ], np.float32)
            staging = self._create_buffer(
                payload.nbytes, vk.VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
                | vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                data=payload,
            )
            prepared.append((staging, offset, shape))
        subresource = vk.VkImageSubresourceRange(
            aspectMask=vk.VK_IMAGE_ASPECT_COLOR_BIT,
            baseMipLevel=0, levelCount=1, baseArrayLayer=0, layerCount=1,
        )

        def upload(command):
            vk.vkCmdPipelineBarrier(
                command, vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                vk.VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, None, 0, None, 1,
                [vk.VkImageMemoryBarrier(
                    sType=vk.VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                    srcAccessMask=vk.VK_ACCESS_SHADER_READ_BIT,
                    dstAccessMask=vk.VK_ACCESS_TRANSFER_WRITE_BIT,
                    oldLayout=vk.VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    newLayout=vk.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    srcQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
                    dstQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
                    image=texture.image, subresourceRange=subresource,
                )],
            )
            for staging, offset, shape in prepared:
                vk.vkCmdCopyBufferToImage(
                    command, staging.buffer, texture.image,
                    vk.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1,
                    [vk.VkBufferImageCopy(
                        bufferOffset=0, bufferRowLength=0, bufferImageHeight=0,
                        imageSubresource=vk.VkImageSubresourceLayers(
                            aspectMask=vk.VK_IMAGE_ASPECT_COLOR_BIT,
                            mipLevel=0, baseArrayLayer=0, layerCount=1,
                        ),
                        imageOffset=vk.VkOffset3D(
                            x=offset[2], y=offset[1], z=offset[0],
                        ),
                        imageExtent=vk.VkExtent3D(
                            width=shape[2], height=shape[1], depth=shape[0],
                        ),
                    )],
                )
            vk.vkCmdPipelineBarrier(
                command, vk.VK_PIPELINE_STAGE_TRANSFER_BIT,
                vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                0, None, 0, None, 1,
                [vk.VkImageMemoryBarrier(
                    sType=vk.VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                    srcAccessMask=vk.VK_ACCESS_TRANSFER_WRITE_BIT,
                    dstAccessMask=vk.VK_ACCESS_SHADER_READ_BIT,
                    oldLayout=vk.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    newLayout=vk.VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    srcQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
                    dstQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
                    image=texture.image, subresourceRange=subresource,
                )],
            )

        try:
            self._single_use(upload)
        finally:
            for staging, _offset, _shape in prepared:
                vk.vkDestroyBuffer(self.device, staging.buffer, None)
                vk.vkFreeMemory(self.device, staging.memory, None)
                self._buffers.remove(staging)

    def _update_sampled_volume_from_gpu(self, texture, source, shape):
        """Refresh a sampled volume directly from a same-device float buffer."""
        depth, height, width = map(int, shape)
        subresource = vk.VkImageSubresourceRange(
            aspectMask=vk.VK_IMAGE_ASPECT_COLOR_BIT,
            baseMipLevel=0, levelCount=1, baseArrayLayer=0, layerCount=1,
        )

        def upload(command):
            vk.vkCmdPipelineBarrier(
                command, vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                vk.VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                0, None, 0, None, 1, [vk.VkImageMemoryBarrier(
                    sType=vk.VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                    srcAccessMask=vk.VK_ACCESS_SHADER_READ_BIT,
                    dstAccessMask=vk.VK_ACCESS_TRANSFER_WRITE_BIT,
                    oldLayout=vk.VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    newLayout=vk.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    srcQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
                    dstQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
                    image=texture.image, subresourceRange=subresource,
                )],
            )
            vk.vkCmdCopyBufferToImage(
                command, source.buffer, texture.image,
                vk.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1,
                [vk.VkBufferImageCopy(
                    bufferOffset=0, bufferRowLength=0, bufferImageHeight=0,
                    imageSubresource=vk.VkImageSubresourceLayers(
                        aspectMask=vk.VK_IMAGE_ASPECT_COLOR_BIT,
                        mipLevel=0, baseArrayLayer=0, layerCount=1,
                    ),
                    imageExtent=vk.VkExtent3D(
                        width=width, height=height, depth=depth,
                    ),
                )],
            )
            vk.vkCmdPipelineBarrier(
                command, vk.VK_PIPELINE_STAGE_TRANSFER_BIT,
                vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                0, None, 0, None, 1, [vk.VkImageMemoryBarrier(
                    sType=vk.VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                    srcAccessMask=vk.VK_ACCESS_TRANSFER_WRITE_BIT,
                    dstAccessMask=vk.VK_ACCESS_SHADER_READ_BIT,
                    oldLayout=vk.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    newLayout=vk.VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    srcQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
                    dstQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
                    image=texture.image, subresourceRange=subresource,
                )],
            )

        self._single_use(upload)

    def _create_sampled_texture(self, levels, texture, image_format):
        """Upload a complete RGBA8 mip pyramid to an optimal sampled image."""
        height, width, _ = levels[0].shape
        image = vk.vkCreateImage(
            self.device,
            vk.VkImageCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
                imageType=vk.VK_IMAGE_TYPE_2D,
                format=image_format,
                extent=vk.VkExtent3D(width=width, height=height, depth=1),
                mipLevels=len(levels), arrayLayers=1,
                samples=vk.VK_SAMPLE_COUNT_1_BIT,
                tiling=vk.VK_IMAGE_TILING_OPTIMAL,
                usage=(vk.VK_IMAGE_USAGE_TRANSFER_DST_BIT
                       | vk.VK_IMAGE_USAGE_SAMPLED_BIT),
                sharingMode=vk.VK_SHARING_MODE_EXCLUSIVE,
                initialLayout=vk.VK_IMAGE_LAYOUT_UNDEFINED,
            ), None,
        )
        requirements = vk.vkGetImageMemoryRequirements(self.device, image)
        memory = vk.vkAllocateMemory(
            self.device,
            vk.VkMemoryAllocateInfo(
                sType=vk.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                allocationSize=requirements.size,
                memoryTypeIndex=self._memory_type(
                    requirements.memoryTypeBits,
                    vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                ),
            ), None,
        )
        vk.vkBindImageMemory(self.device, image, memory, 0)
        offsets = []
        chunks = []
        offset = 0
        for level in levels:
            payload = np.ascontiguousarray(level, dtype=np.uint8)
            offsets.append(offset)
            chunks.append(payload.reshape(-1))
            offset += payload.nbytes
        packed = np.concatenate(chunks)
        staging = self._create_buffer(
            packed.nbytes, vk.VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
            | vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            data=packed,
        )
        subresource = vk.VkImageSubresourceRange(
            aspectMask=vk.VK_IMAGE_ASPECT_COLOR_BIT,
            baseMipLevel=0, levelCount=len(levels),
            baseArrayLayer=0, layerCount=1,
        )
        regions = [vk.VkBufferImageCopy(
            bufferOffset=offsets[index],
            bufferRowLength=0, bufferImageHeight=0,
            imageSubresource=vk.VkImageSubresourceLayers(
                aspectMask=vk.VK_IMAGE_ASPECT_COLOR_BIT,
                mipLevel=index, baseArrayLayer=0, layerCount=1,
            ),
            imageOffset=vk.VkOffset3D(x=0, y=0, z=0),
            imageExtent=vk.VkExtent3D(
                width=level.shape[1], height=level.shape[0], depth=1
            ),
        ) for index, level in enumerate(levels)]

        def upload(command):
            vk.vkCmdPipelineBarrier(
                command, vk.VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                vk.VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, None, 0, None, 1,
                [vk.VkImageMemoryBarrier(
                    sType=vk.VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                    srcAccessMask=0,
                    dstAccessMask=vk.VK_ACCESS_TRANSFER_WRITE_BIT,
                    oldLayout=vk.VK_IMAGE_LAYOUT_UNDEFINED,
                    newLayout=vk.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    srcQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
                    dstQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
                    image=image, subresourceRange=subresource,
                )],
            )
            vk.vkCmdCopyBufferToImage(
                command, staging.buffer, image,
                vk.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                len(regions), regions,
            )
            vk.vkCmdPipelineBarrier(
                command, vk.VK_PIPELINE_STAGE_TRANSFER_BIT,
                vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                0, None, 0, None, 1,
                [vk.VkImageMemoryBarrier(
                    sType=vk.VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                    srcAccessMask=vk.VK_ACCESS_TRANSFER_WRITE_BIT,
                    dstAccessMask=vk.VK_ACCESS_SHADER_READ_BIT,
                    oldLayout=vk.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    newLayout=vk.VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    srcQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
                    dstQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
                    image=image, subresourceRange=subresource,
                )],
            )

        self._single_use(upload)
        vk.vkDestroyBuffer(self.device, staging.buffer, None)
        vk.vkFreeMemory(self.device, staging.memory, None)
        self._buffers.remove(staging)
        view = vk.vkCreateImageView(
            self.device,
            vk.VkImageViewCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                image=image, viewType=vk.VK_IMAGE_VIEW_TYPE_2D,
                format=image_format,
                subresourceRange=subresource,
            ), None,
        )
        address_modes = {
            "repeat": vk.VK_SAMPLER_ADDRESS_MODE_REPEAT,
            "clamp": vk.VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
            "mirror": vk.VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT,
        }
        filtering = (
            vk.VK_FILTER_LINEAR if texture.linear_filter
            else vk.VK_FILTER_NEAREST
        )
        sampler = vk.vkCreateSampler(
            self.device,
            vk.VkSamplerCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
                magFilter=filtering, minFilter=filtering,
                mipmapMode=(vk.VK_SAMPLER_MIPMAP_MODE_LINEAR
                            if texture.linear_filter
                            else vk.VK_SAMPLER_MIPMAP_MODE_NEAREST),
                addressModeU=address_modes[texture.wrap_s],
                addressModeV=address_modes[texture.wrap_t],
                addressModeW=vk.VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                minLod=0.0, maxLod=float(len(levels) - 1),
                maxAnisotropy=1.0,
            ), None,
        )
        result = SampledTexture(image, memory, view, sampler)
        self._sampled_textures.append(result)
        return result

    def _create_sampled_volume(self, data, gpu_source=None):
        """Upload one float32 scalar field to a linearly sampled 3D image."""
        payload = np.ascontiguousarray(data, dtype=np.float32)
        depth, height, width = payload.shape
        image = vk.vkCreateImage(
            self.device, vk.VkImageCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
                imageType=vk.VK_IMAGE_TYPE_3D,
                format=vk.VK_FORMAT_R32_SFLOAT,
                extent=vk.VkExtent3D(width=width, height=height, depth=depth),
                mipLevels=1, arrayLayers=1, samples=vk.VK_SAMPLE_COUNT_1_BIT,
                tiling=vk.VK_IMAGE_TILING_OPTIMAL,
                usage=(vk.VK_IMAGE_USAGE_TRANSFER_DST_BIT
                       | vk.VK_IMAGE_USAGE_SAMPLED_BIT),
                sharingMode=vk.VK_SHARING_MODE_EXCLUSIVE,
                initialLayout=vk.VK_IMAGE_LAYOUT_UNDEFINED,
            ), None,
        )
        requirements = vk.vkGetImageMemoryRequirements(self.device, image)
        memory = vk.vkAllocateMemory(
            self.device, vk.VkMemoryAllocateInfo(
                sType=vk.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                allocationSize=requirements.size,
                memoryTypeIndex=self._memory_type(
                    requirements.memoryTypeBits,
                    vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                ),
            ), None,
        )
        vk.vkBindImageMemory(self.device, image, memory, 0)
        resident = (
            gpu_source is not None
            and getattr(gpu_source, "device", None) == self.device
            and tuple(getattr(gpu_source, "shape", ())) == payload.shape
            and np.dtype(getattr(gpu_source, "dtype", None)) == np.dtype(np.float32)
            and int(getattr(gpu_source, "byte_size", -1)) == payload.nbytes
        )
        staging = None if resident else self._create_buffer(
            payload.nbytes, vk.VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
            | vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            data=payload,
        )
        source_buffer = gpu_source.buffer if resident else staging.buffer
        subresource = vk.VkImageSubresourceRange(
            aspectMask=vk.VK_IMAGE_ASPECT_COLOR_BIT,
            baseMipLevel=0, levelCount=1, baseArrayLayer=0, layerCount=1,
        )

        def upload(command):
            vk.vkCmdPipelineBarrier(
                command, vk.VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                vk.VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, None, 0, None, 1,
                [vk.VkImageMemoryBarrier(
                    sType=vk.VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                    srcAccessMask=0, dstAccessMask=vk.VK_ACCESS_TRANSFER_WRITE_BIT,
                    oldLayout=vk.VK_IMAGE_LAYOUT_UNDEFINED,
                    newLayout=vk.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    srcQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
                    dstQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
                    image=image, subresourceRange=subresource,
                )],
            )
            vk.vkCmdCopyBufferToImage(
                command, source_buffer, image,
                vk.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1,
                [vk.VkBufferImageCopy(
                    bufferOffset=0, bufferRowLength=0, bufferImageHeight=0,
                    imageSubresource=vk.VkImageSubresourceLayers(
                        aspectMask=vk.VK_IMAGE_ASPECT_COLOR_BIT,
                        mipLevel=0, baseArrayLayer=0, layerCount=1,
                    ),
                    imageOffset=vk.VkOffset3D(x=0, y=0, z=0),
                    imageExtent=vk.VkExtent3D(
                        width=width, height=height, depth=depth,
                    ),
                )],
            )
            vk.vkCmdPipelineBarrier(
                command, vk.VK_PIPELINE_STAGE_TRANSFER_BIT,
                vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                0, None, 0, None, 1,
                [vk.VkImageMemoryBarrier(
                    sType=vk.VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                    srcAccessMask=vk.VK_ACCESS_TRANSFER_WRITE_BIT,
                    dstAccessMask=vk.VK_ACCESS_SHADER_READ_BIT,
                    oldLayout=vk.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    newLayout=vk.VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    srcQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
                    dstQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
                    image=image, subresourceRange=subresource,
                )],
            )

        self._single_use(upload)
        if staging is not None:
            vk.vkDestroyBuffer(self.device, staging.buffer, None)
            vk.vkFreeMemory(self.device, staging.memory, None)
            self._buffers.remove(staging)
        view = vk.vkCreateImageView(
            self.device, vk.VkImageViewCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                image=image, viewType=vk.VK_IMAGE_VIEW_TYPE_3D,
                format=vk.VK_FORMAT_R32_SFLOAT,
                subresourceRange=subresource,
            ), None,
        )
        sampler = vk.vkCreateSampler(
            self.device, vk.VkSamplerCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
                magFilter=vk.VK_FILTER_LINEAR, minFilter=vk.VK_FILTER_LINEAR,
                mipmapMode=vk.VK_SAMPLER_MIPMAP_MODE_NEAREST,
                addressModeU=vk.VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                addressModeV=vk.VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                addressModeW=vk.VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                minLod=0.0, maxLod=0.0, maxAnisotropy=1.0,
            ), None,
        )
        result = SampledTexture(image, memory, view, sampler)
        self._sampled_textures.append(result)
        return result

    def _release_sampled_textures(self, textures):
        for texture in reversed(textures):
            vk.vkDestroySampler(self.device, texture.sampler, None)
            vk.vkDestroyImageView(self.device, texture.view, None)
            vk.vkDestroyImage(self.device, texture.image, None)
            vk.vkFreeMemory(self.device, texture.memory, None)

    def _buffer_address(self, buffer):
        info = vk.VkBufferDeviceAddressInfo(
            sType=vk.VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
            buffer=buffer.buffer,
        )
        return self._raw_buffer_address(self.device, vk.ffi.addressof(info))

    def _as_address(self, structure):
        info = vk.VkAccelerationStructureDeviceAddressInfoKHR(
            sType=vk.VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR,
            accelerationStructure=structure.handle,
        )
        return self._raw_as_address(self.device, vk.ffi.addressof(info))

    def _single_use(self, record):
        command = vk.vkAllocateCommandBuffers(
            self.device,
            vk.VkCommandBufferAllocateInfo(
                sType=vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                commandPool=self.command_pool,
                level=vk.VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                commandBufferCount=1,
            ),
        )[0]
        vk.vkBeginCommandBuffer(
            command,
            vk.VkCommandBufferBeginInfo(
                sType=vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                flags=vk.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
            ),
        )
        record(command)
        vk.vkEndCommandBuffer(command)
        vk.vkQueueSubmit(
            self.queue,
            1,
            [
                vk.VkSubmitInfo(
                    sType=vk.VK_STRUCTURE_TYPE_SUBMIT_INFO,
                    commandBufferCount=1,
                    pCommandBuffers=[command],
                )
            ],
            vk.VK_NULL_HANDLE,
        )
        vk.vkQueueWaitIdle(self.queue)
        vk.vkFreeCommandBuffers(self.device, self.command_pool, 1, [command])

    def _make_as(
        self, geometry, primitive_count, structure_type, *, allow_update=False,
    ):
        flags = vk.VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR
        if allow_update:
            flags |= vk.VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR
        build = vk.VkAccelerationStructureBuildGeometryInfoKHR(
            sType=vk.VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR,
            type=structure_type,
            flags=flags,
            mode=vk.VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR,
            geometryCount=1,
            pGeometries=[geometry],
        )
        sizes = vk.VkAccelerationStructureBuildSizesInfoKHR(
            sType=vk.VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR
        )
        self.get_as_sizes(
            self.device,
            vk.VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
            build,
            [primitive_count],
            sizes,
        )
        storage = self._create_buffer(
            sizes.accelerationStructureSize,
            vk.VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR,
            vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        )
        handle = self.create_as(
            self.device,
            vk.VkAccelerationStructureCreateInfoKHR(
                sType=vk.VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR,
                buffer=storage.buffer,
                size=sizes.accelerationStructureSize,
                type=structure_type,
            ),
            None,
        )
        scratch = self._create_buffer(
            max(sizes.buildScratchSize, sizes.updateScratchSize),
            vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
            | BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
            vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            device_address=True,
        )
        build.dstAccelerationStructure = handle
        build.scratchData.deviceAddress = self._buffer_address(scratch)
        range_info = vk.VkAccelerationStructureBuildRangeInfoKHR(
            primitiveCount=primitive_count,
            primitiveOffset=0,
            firstVertex=0,
            transformOffset=0,
        )
        range_pointer = vk.ffi.addressof(range_info)
        ranges = vk.ffi.new(
            "VkAccelerationStructureBuildRangeInfoKHR*[]", [range_pointer]
        )
        self._single_use(lambda command: self.build_as(command, 1, [build], ranges))
        result = AccelerationStructure(handle, storage, scratch)
        self._structures.append(result)
        return result

    def _scene_instance_bytes(self, instances):
        return b"".join(
            struct.pack(
                "<12fIIQ",
                *np.asarray(item.mesh.transform.matrix[:3], np.float32).reshape(-1),
                (int(item.visibility_mask) << 24) | int(item.triangle_offset),
                0,
                self._as_address(item.blas.structure),
            )
            for item in instances
        )

    @staticmethod
    def _mesh_blas_vertices(mesh):
        positions = mesh.vertices[mesh.indices].reshape((-1, 3))
        return np.ascontiguousarray(
            np.column_stack((
                positions, np.ones(len(positions), dtype=np.float32),
            )),
            dtype=np.float32,
        )

    def _refit_scene_blases(self, entries):
        """Update equal-topology BLAS bounds without reallocating them."""
        builds = []
        ranges = []
        range_storage = []
        for item in entries:
            mesh = item.mesh
            triangle_data = vk.VkAccelerationStructureGeometryTrianglesDataKHR(
                sType=(
                    vk.VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR
                ),
                vertexFormat=vk.VK_FORMAT_R32G32B32_SFLOAT,
                vertexData=vk.VkDeviceOrHostAddressConstKHR(
                    deviceAddress=self._buffer_address(item.vertex_buffer)
                ),
                vertexStride=16,
                maxVertex=len(mesh.indices) * 3 - 1,
                indexType=vk.VK_INDEX_TYPE_NONE_KHR,
            )
            geometry = vk.VkAccelerationStructureGeometryKHR(
                sType=vk.VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR,
                geometryType=vk.VK_GEOMETRY_TYPE_TRIANGLES_KHR,
                geometry=vk.VkAccelerationStructureGeometryDataKHR(
                    triangles=triangle_data
                ),
                flags=vk.VK_GEOMETRY_OPAQUE_BIT_KHR,
            )
            build = vk.VkAccelerationStructureBuildGeometryInfoKHR(
                sType=(
                    vk.VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR
                ),
                type=vk.VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR,
                flags=(
                    vk.VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR
                    | vk.VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR
                ),
                mode=vk.VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR,
                srcAccelerationStructure=item.structure.handle,
                dstAccelerationStructure=item.structure.handle,
                geometryCount=1,
                pGeometries=[geometry],
            )
            build.scratchData.deviceAddress = self._buffer_address(
                item.structure.scratch
            )
            range_info = vk.VkAccelerationStructureBuildRangeInfoKHR(
                primitiveCount=len(mesh.indices), primitiveOffset=0,
                firstVertex=0, transformOffset=0,
            )
            range_storage.append(range_info)
            range_pointer = vk.ffi.addressof(range_info)
            ranges.append(range_pointer)
            builds.append(build)
        if builds:
            self._single_use(
                lambda command: self.build_as(
                    command, len(builds), builds, ranges
                )
            )

    def _tlas_geometry(self, instance_buffer):
        instance_data = vk.VkAccelerationStructureGeometryInstancesDataKHR(
            sType=(
                vk.VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR
            ),
            arrayOfPointers=vk.VK_FALSE,
            data=vk.VkDeviceOrHostAddressConstKHR(
                deviceAddress=self._buffer_address(instance_buffer)
            ),
        )
        return vk.VkAccelerationStructureGeometryKHR(
            sType=vk.VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR,
            geometryType=vk.VK_GEOMETRY_TYPE_INSTANCES_KHR,
            geometry=vk.VkAccelerationStructureGeometryDataKHR(
                instances=instance_data
            ),
        )

    def _rebuild_scene_tlas(self, tlas, instance_buffer, instance_count):
        """Rebuild an equal-sized TLAS after instance transforms change."""
        geometry = self._tlas_geometry(instance_buffer)
        build = vk.VkAccelerationStructureBuildGeometryInfoKHR(
            sType=(
                vk.VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR
            ),
            type=vk.VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR,
            flags=vk.VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR,
            mode=vk.VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR,
            dstAccelerationStructure=tlas.handle,
            geometryCount=1,
            pGeometries=[geometry],
        )
        build.scratchData.deviceAddress = self._buffer_address(tlas.scratch)
        range_info = vk.VkAccelerationStructureBuildRangeInfoKHR(
            primitiveCount=instance_count,
            primitiveOffset=0, firstVertex=0, transformOffset=0,
        )
        range_pointer = vk.ffi.addressof(range_info)
        ranges = vk.ffi.new(
            "VkAccelerationStructureBuildRangeInfoKHR*[]", [range_pointer]
        )
        self._single_use(
            lambda command: self.build_as(command, 1, [build], ranges)
        )

    def _build_scene(self, scene):
        stage_start = time.perf_counter()
        programs, default_program = self._ensure_scene_pipeline(scene)
        custom_attribute_layout = self._material_attribute_layout(
            scene, programs, self.config.material_modifier
        )
        triangles = scene.render_triangles()
        if not len(triangles):
            raise ValueError("Vulkan ray-query rendering requires at least one triangle")
        positions = triangles.reshape((-1, 3))
        vertices = np.ascontiguousarray(
            np.column_stack((positions, np.ones(len(positions), dtype=np.float32))),
            dtype=np.float32,
        )
        vertex_buffer = self._create_uploaded_device_buffer(
            vertices,
            vk.VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR
            | BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
            | vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            device_address=True,
        )
        self.scene_vertex_buffer = vertex_buffer
        self.scene_previous_vertex_buffer = self._create_uploaded_device_buffer(
            vertices, vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        )
        material_data = scene.triangle_material_data(programs, default_program)
        self.scene_material_buffer = self._create_uploaded_device_buffer(
            material_data,
            vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        )
        light_data = scene.analytic_light_data()
        self.scene_light_buffer = self._create_uploaded_device_buffer(
            light_data,
            vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        )
        area_light_data = scene.emissive_triangle_data()
        self.scene_area_light_buffer = self._create_uploaded_device_buffer(
            area_light_data,
            vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        )
        attribute_data = scene.triangle_attribute_data()
        self.scene_attribute_buffer = self._create_uploaded_device_buffer(
            attribute_data,
            vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        )
        self.scene_custom_attribute_layout = custom_attribute_layout
        self.scene_custom_attribute_buffer = None
        if custom_attribute_layout is not None:
            custom_attribute_data = custom_attribute_layout.pack(scene)
            if custom_attribute_data.nbytes == 0:
                custom_attribute_data = np.zeros(4, np.float32)
            self.scene_custom_attribute_buffer = (
                self._create_uploaded_device_buffer(
                    custom_attribute_data,
                    vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                )
            )
        # Native shaders only consult word zero for bounds checking. Avoid
        # building and uploading a duplicate packed mip pyramid in that mode.
        texture_data = (
            np.asarray([len(scene.textures)], dtype=np.uint32)
            if self.native_textures_enabled
            else scene.texture_data()
        )
        if self.config.wavefront_device_local_textures:
            self.scene_texture_buffer = self._create_uploaded_device_buffer(
                texture_data, vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
            )
        else:
            self.scene_texture_buffer = self._create_buffer(
                texture_data.nbytes,
                vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
                | vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                data=texture_data,
            )
        texture_binding_data = scene.texture_binding_data()
        self.scene_texture_binding_buffer = self._create_uploaded_device_buffer(
            texture_binding_data,
            vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        )
        from ...volume import pack_volumes
        volume_headers, volume_scalars, volume_transfers = pack_volumes(
            scene.visible_volumes,
            empty_space_skipping=self.config.volume_empty_space_skipping,
        )
        self.scene_volume_empty_space_skipping = bool(
            np.any(volume_headers["acceleration_parameters"][:, 1:] > 0)
        )
        self.scene_volume_header_buffer = self._create_uploaded_device_buffer(
            volume_headers, vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        )
        self.scene_volume_scalar_buffer = self._create_uploaded_device_buffer(
            volume_scalars, vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        )
        self.scene_volume_transfer_buffer = self._create_uploaded_device_buffer(
            volume_transfers, vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        )
        self.scene_triangle_volume_buffer = self._create_uploaded_device_buffer(
            scene.triangle_volume_indices(),
            vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        )
        if len(scene.visible_volumes) > MAX_NATIVE_VOLUMES:
            raise ValueError(
                f"Vulkan backend supports at most {MAX_NATIVE_VOLUMES} visible volumes"
            )
        self.scene_sampled_volumes = [
            self._create_sampled_volume(volume.data, volume.gpu_source)
            for volume in scene.visible_volumes
        ] or [self._create_sampled_volume(np.zeros((2, 2, 2), np.float32))]
        self.scene_sampled_textures = []
        if self.native_textures_enabled:
            if len(scene.textures) > MAX_NATIVE_TEXTURES:
                raise ValueError(
                    f"Native texture backend supports at most {MAX_NATIVE_TEXTURES} "
                    f"textures; scene contains {len(scene.textures)}"
                )
            textures = scene.textures
            if not textures:
                from ...scene import Texture
                textures = (Texture(np.full((1, 1, 4), 255, np.uint8)),)
            for texture in textures:
                self.scene_sampled_textures.extend((
                    self._create_sampled_texture(
                        scene._texture_mips(texture.pixels, srgb=True), texture,
                        vk.VK_FORMAT_R8G8B8A8_SRGB,
                    ),
                    self._create_sampled_texture(
                        scene._texture_mips(texture.pixels, srgb=False), texture,
                        vk.VK_FORMAT_R8G8B8A8_UNORM,
                    ),
                ))
        self.last_timings["scene_upload_ms"] = (
            time.perf_counter() - stage_start
        ) * 1000.0
        stage_start = time.perf_counter()
        blases = []
        instances = []
        blas_by_geometry = {}
        triangle_offset = 0
        for mesh in scene.render_meshes:
            mesh_triangles = mesh.vertices[mesh.indices]
            if not len(mesh_triangles):
                continue
            if triangle_offset + len(mesh_triangles) > (1 << 24):
                raise ValueError(
                    "scene triangle offsets exceed Vulkan's 24-bit instance "
                    "custom index"
                )
            geometry_source = mesh.resource or mesh
            blas_entry = blas_by_geometry.get(id(geometry_source))
            if blas_entry is None:
                object_vertices = self._mesh_blas_vertices(geometry_source)
                blas_vertices = self._create_uploaded_device_buffer(
                    object_vertices,
                    vk.VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR
                    | BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                    device_address=True,
                )
                triangle_data = vk.VkAccelerationStructureGeometryTrianglesDataKHR(
                    sType=(
                        vk.VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR
                    ),
                    vertexFormat=vk.VK_FORMAT_R32G32B32_SFLOAT,
                    vertexData=vk.VkDeviceOrHostAddressConstKHR(
                        deviceAddress=self._buffer_address(blas_vertices)
                    ),
                    vertexStride=16,
                    maxVertex=len(object_vertices) - 1,
                    indexType=vk.VK_INDEX_TYPE_NONE_KHR,
                )
                geometry = vk.VkAccelerationStructureGeometryKHR(
                    sType=vk.VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR,
                    geometryType=vk.VK_GEOMETRY_TYPE_TRIANGLES_KHR,
                    geometry=vk.VkAccelerationStructureGeometryDataKHR(
                        triangles=triangle_data
                    ),
                    flags=vk.VK_GEOMETRY_OPAQUE_BIT_KHR,
                )
                blas = self._make_as(
                    geometry, len(mesh_triangles),
                    vk.VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR,
                    allow_update=geometry_source.deformable,
                )
                blas_entry = SceneBlas(blas, geometry_source, blas_vertices)
                blas_by_geometry[id(geometry_source)] = blas_entry
                blases.append(blas_entry)
            instances.append(SceneTlasInstance(
                mesh, blas_entry, triangle_offset
            ))
            triangle_offset += len(mesh_triangles)
        self.scene_blases = blases
        self.scene_instances = instances
        self.last_timings["blas_count"] = len(blases)
        self.last_timings["instance_count"] = len(instances)
        self.last_timings["shared_blas_savings"] = max(
            0, len(instances) - len(blases)
        )
        self.last_timings["blas_ms"] = (time.perf_counter() - stage_start) * 1000.0

        stage_start = time.perf_counter()
        instance_bytes = self._scene_instance_bytes(instances)
        instance_buffer = self._create_buffer(
            len(instance_bytes),
            vk.VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR
            | BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
            vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            data=instance_bytes,
            device_address=True,
        )
        self.scene_instance_buffer = instance_buffer
        tlas_geometry = self._tlas_geometry(instance_buffer)
        tlas = self._make_as(
            tlas_geometry, len(instances),
            vk.VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR
        )
        self.last_timings["tlas_ms"] = (time.perf_counter() - stage_start) * 1000.0
        return tlas

    def _release_resources(self, structures, buffers):
        for structure in reversed(structures):
            self.destroy_as(self.device, structure.handle, None)
        for buffer in reversed(buffers):
            vk.vkDestroyBuffer(self.device, buffer.buffer, None)
            vk.vkFreeMemory(self.device, buffer.memory, None)

    def _image_barrier(self, image, old_layout, new_layout, src_access, dst_access):
        return vk.VkImageMemoryBarrier(
            sType=vk.VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            srcAccessMask=src_access,
            dstAccessMask=dst_access,
            oldLayout=old_layout,
            newLayout=new_layout,
            srcQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
            dstQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
            image=image,
            subresourceRange=vk.VkImageSubresourceRange(
                aspectMask=vk.VK_IMAGE_ASPECT_COLOR_BIT,
                baseMipLevel=0,
                levelCount=1,
                baseArrayLayer=0,
                layerCount=1,
            ),
        )
