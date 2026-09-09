"""Reconstruction kernel with application-owned image/camera bindings."""

from importlib.resources import files
from operator import index
import vulkan as vk

from ..pipeline.reconstruction import ReconstructionSettings, RECONSTRUCT_PUSH_SIZE
from ..pipeline.graph import VulkanOperation
from ..pipeline.vulkan import VulkanPass, VulkanResource, VulkanResourceUse
from .kernel import VulkanKernel


class VulkanReconstruction:
    """Borrow explicit reconstruction resources; own immutable pipeline/bindings.

    outputs is one image or 1-8 images, padded to eight descriptors. Camera
    forward.w selects the descriptor at execution; graph users must supply the
    matching output_index to operation. Single-output binding repeats the image.
    FSR2 is an upstream producer: this stage encodes its full-resolution HDR.
    """

    def __init__(
        self,
        runtime,
        *,
        hdr,
        position,
        normal,
        previous_color,
        previous_position,
        previous_normal,
        history_color,
        outputs,
        previous_camera,
        current_camera,
        material,
    ):
        with runtime.lock:
            runtime.require_open()
            self.runtime = runtime
            outputs = (
                tuple(outputs) if isinstance(outputs, (tuple, list)) else (outputs,)
            )
            if not 1 <= len(outputs) <= 8:
                raise ValueError("Reconstruction requires 1-8 outputs")
            self.outputs = outputs + (outputs[-1],) * (8 - len(outputs))
            self.output_extent = (outputs[0].width, outputs[0].height)
            self.closed = False
            self.completion = None
            self.hdr = hdr
            images = (
                hdr,
                position,
                normal,
                previous_color,
                previous_position,
                previous_normal,
                history_color,
                material,
            )
            formats = (
                vk.VK_FORMAT_R16G16B16A16_SFLOAT,
                vk.VK_FORMAT_R32_SFLOAT,
                vk.VK_FORMAT_R32_UINT,
                vk.VK_FORMAT_B10G11R11_UFLOAT_PACK32,
                vk.VK_FORMAT_R32_SFLOAT,
                vk.VK_FORMAT_R32_UINT,
                vk.VK_FORMAT_B10G11R11_UFLOAT_PACK32,
                vk.VK_FORMAT_R32_UINT,
            )
            for image, fmt in zip(images, formats):
                self._image(image, fmt)
            output_format = outputs[0].format
            if output_format not in (
                vk.VK_FORMAT_R8G8B8A8_UNORM,
                vk.VK_FORMAT_B8G8R8A8_UNORM,
            ):
                raise ValueError("Reconstruction output must be RGBA8 or BGRA8 UNORM")
            if (
                output_format == vk.VK_FORMAT_B8G8R8A8_UNORM
                and not runtime.formatless_storage_write_supported
            ):
                raise ValueError("BGRA output requires formatless storage writes")
            for output in outputs:
                self._image(output, output_format)
                if (output.width, output.height) != self.output_extent:
                    raise ValueError("Reconstruction output extents must match")
            reads = {image.image for image in (*images[:6], material)}
            writes = {image.image for image in (history_color, *outputs)}
            if reads & writes or history_color.image in {o.image for o in outputs}:
                raise ValueError(
                    "Reconstruction writes must not alias inputs or each other"
                )
            self.bindings = {
                i: VulkanResource.image(image) for i, image in enumerate(images[:7])
            }
            self.bindings[10] = VulkanResource.image(material)
            for binding, camera in ((8, previous_camera), (9, current_camera)):
                camera.require_open()
                if camera.runtime is not runtime or camera.byte_size < 64:
                    raise ValueError(
                        "Camera binding requires same-runtime storage of at least 64 bytes"
                    )
                self.bindings[binding] = VulkanResource.buffer(camera)
            shader = (
                "wavefront_reconstruct_bgra.comp.spv"
                if output_format == vk.VK_FORMAT_B8G8R8A8_UNORM
                else "wavefront_reconstruct.comp.spv"
            )
            self.kernel = VulkanKernel(
                runtime,
                files("ordinarylight.shaders").joinpath(shader).read_bytes(),
                self.bindings,
                image_arrays={7: tuple(VulkanResource.image(o) for o in self.outputs)},
                push_constant_size=RECONSTRUCT_PUSH_SIZE,
            )

    def _image(self, image, format):
        image.require_open()
        if (
            image.runtime is not self.runtime
            or image.format != format
            or not image.usage & vk.VK_IMAGE_USAGE_STORAGE_BIT
        ):
            raise ValueError(
                "Reconstruction image format/runtime/storage usage mismatch"
            )

    def require_open(self):
        if self.closed:
            raise RuntimeError("Reconstruction stage is closed")
        self.kernel.require_open()

    def operation(
        self,
        *,
        extent=None,
        settings=None,
        history_valid=False,
        effects=(),
        output_index=0,
        after=(),
        external_output=False,
    ):
        """Graph work with explicit hazards and completion tracking.

        external_output=True is for a parent pass managing output synchronization
        (e.g. a cached swapchain-image-array dispatch). That parent must retain and
        synchronize the dynamically selected output. Ordinary graph users should
        leave it false and match output_index to current_camera.forward.w.
        """
        self.require_open()
        extent = (self.hdr.width, self.hdr.height) if extent is None else tuple(extent)
        settings = ReconstructionSettings() if settings is None else settings
        effects = tuple(effects)
        constants = settings.pack(extent, history_valid=history_valid, effects=effects)
        if any(effect.kind for effect in effects):
            material = self.bindings[10].owner
            if material.width < extent[0] or material.height < extent[1]:
                raise ValueError(
                    "Effect material image is smaller than the active extent"
                )
        if extent[0] > self.hdr.width or extent[1] > self.hdr.height:
            raise ValueError("Reconstruction extent exceeds HDR input")
        if settings.temporal_enabled or settings.diffuse_filter:
            for binding in (1, 2):
                image = self.bindings[binding].owner
                if image.width < extent[0] or image.height < extent[1]:
                    raise ValueError(
                        "Reconstruction guides are smaller than the active extent"
                    )
        if settings.temporal_enabled:
            required = (3, 6, 4, 5) if history_valid else (6,)
            for binding in required:
                image = self.bindings[binding].owner
                expected = self.output_extent if binding in (3, 6) else extent
                if image.width < expected[0] or image.height < expected[1]:
                    raise ValueError(
                        "Reconstruction history is smaller than the active extent"
                    )
        if settings.upscale_filter == "fsr2" and extent != self.output_extent:
            raise ValueError(
                "FSR2 reconstruction requires upstream full-resolution HDR"
            )
        output_index = index(output_index)
        if not 0 <= output_index < 8:
            raise ValueError("Reconstruction output index must be between 0 and 7")
        uses = {}
        bindings = dict(self.bindings)
        if not external_output:
            bindings[7] = VulkanResource.image(self.outputs[output_index])
        for binding, resource in bindings.items():
            access = (
                vk.VK_ACCESS_SHADER_WRITE_BIT
                if binding in (6, 7)
                else vk.VK_ACCESS_SHADER_READ_BIT
            )
            key = (resource.kind, resource.handle)
            old = uses.get(key)
            uses[key] = VulkanResourceUse(
                resource,
                vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                access | (old.access if old else 0),
                vk.VK_IMAGE_LAYOUT_GENERAL if resource.kind == "image" else None,
            )
        width, height = self.output_extent
        after = tuple(after)
        return VulkanOperation(
            [
                VulkanPass(
                    "reconstruct",
                    tuple(uses.values()),
                    lambda command: self.kernel.bind(command, constants),
                    ((width + 7) // 8, (height + 7) // 8, 1),
                )
            ],
            validate=self.require_open,
            dependencies=lambda: after
            + ((self.completion,) if self.completion is not None else ()),
            submitted=lambda completion: setattr(self, "completion", completion),
        )

    def close(self):
        with self.runtime.lock:
            if self.closed:
                return
            if self.completion is not None:
                self.completion.wait()
            self.kernel.close()
            self.closed = True

    def __enter__(self):
        self.require_open()
        return self

    def __exit__(self, *_exc):
        self.close()
