"""Temporal ReLAX with explicit history ownership and graph submissions."""

from importlib.resources import files
import vulkan as vk

from ..denoising.temporal import temporal_constants
from ..pipeline.graph import VulkanOperation
from ..pipeline.vulkan import VulkanPass, VulkanResource, VulkanResourceUse
from .kernel import VulkanKernel

_RGBA = vk.VK_FORMAT_R16G16B16A16_SFLOAT
_FLOAT = vk.VK_FORMAT_R32_SFLOAT
_UINT = vk.VK_FORMAT_R32_UINT


def _image(runtime, image, format, extent):
    image.require_open()
    if image.runtime is not runtime or image.format != format:
        raise ValueError("Temporal image format/runtime mismatch")
    if not image.usage & vk.VK_IMAGE_USAGE_STORAGE_BIT:
        raise ValueError("Temporal images require storage usage")
    if (image.width, image.height) != extent:
        raise ValueError("Temporal image extent mismatch")


class VulkanRelaxHistory:
    """Own temporal lobe radiance/length; borrow guides for that history frame.

    Guides must remain paired with these outputs until all consumers finish.
    ``valid`` becomes true only after graph submission succeeds. Reset invalidates
    reuse but preserves the completion dependency on any in-flight work.
    Supplying images=(diffuse, specular, diffuse_length, specular_length) borrows
    caller-owned history storage instead of allocating it.
    """

    def __init__(
        self, runtime, *, normal_roughness, view_z, material, identity, images=None
    ):
        with runtime.lock:
            runtime.require_open()
            self.runtime = runtime
            self.extent = (view_z.width, view_z.height)
            self.guides = (normal_roughness, view_z, material, identity)
            for image, fmt in zip(self.guides, (_RGBA, _FLOAT, _UINT, _UINT)):
                _image(runtime, image, fmt, self.extent)
            self.closed = False
            self.valid = False
            self.completion = None
            self.images = []
            self._owns_images = images is None
            if images is not None:
                images = tuple(images)
                if len(images) != 4:
                    raise ValueError("History requires four radiance/length images")
                for image, fmt in zip(images, (_RGBA, _RGBA, _FLOAT, _FLOAT)):
                    _image(runtime, image, fmt, self.extent)
                if len({image.image for image in (*images, *self.guides)}) != 8:
                    raise ValueError(
                        "History images must not alias each other or guides"
                    )
                self.images.extend(images)
            runtime.retain(self)
            try:
                if self._owns_images:
                    for fmt in (_RGBA, _RGBA, _FLOAT, _FLOAT):
                        self.images.append(runtime.image(*self.extent, format=fmt))
                (
                    self.diffuse,
                    self.specular,
                    self.diffuse_length,
                    self.specular_length,
                ) = self.images
            except Exception:
                self.close()
                raise

    def require_open(self):
        self.runtime.require_open()
        if self.closed:
            raise RuntimeError("Temporal history is closed")
        for image in (*self.guides, *self.images):
            image.require_open()

    def reset(self):
        with self.runtime.lock:
            self.require_open()
            self.valid = False

    def close(self):
        with self.runtime.lock:
            if self.closed:
                return
            # Check all borrowers before destroying any images.
            if self._owns_images and any(image._borrowers for image in self.images):
                raise RuntimeError("Close temporal history consumers before history")
            if self.completion is not None:
                self.completion.wait()
            if self._owns_images:
                for image in reversed(self.images):
                    image.close()
            self.closed = True
            self.runtime.release(self)

    def __enter__(self):
        self.require_open()
        return self

    def __exit__(self, *_exc):
        self.close()


class VulkanRelaxTemporal:
    """Bind one frame-ring direction, with caller-produced current signals.

    Output history's guides describe the current frame. Previous and output
    histories must be distinct and have equal extents. Current motion is RGBA16F:
    xy is previous-minus-current pixel motion, z expected previous depth.
    Owns immutable valid/reset uniform buffers and kernels. No per-frame host
    upload or implicit device-idle wait is needed to change history validity.
    An optional policy_buffer borrows a caller-synchronized uniform allocation
    with the temporal_constants ABI; its contents control extent and reset.
    """

    def __init__(
        self,
        runtime,
        *,
        diffuse,
        specular,
        motion,
        previous,
        output,
        policy_buffer=None,
        **policy,
    ):
        with runtime.lock:
            runtime.require_open()
            previous.require_open()
            output.require_open()
            if (
                previous is output
                or previous.runtime is not runtime
                or output.runtime is not runtime
            ):
                raise ValueError(
                    "Temporal histories must be distinct on the same runtime"
                )
            if previous.extent != output.extent:
                raise ValueError(
                    "Temporal history extent mismatch; recreate after resize"
                )
            width, height = output.extent
            constants = [
                temporal_constants(width, height, valid, **policy)
                for valid in (False, True)
            ]
            for image in (diffuse, specular, motion):
                _image(runtime, image, _RGBA, output.extent)
            reads = (
                *previous.images,
                *previous.guides,
                *output.guides,
                diffuse,
                specular,
                motion,
            )
            if {image.image for image in output.images} & {
                image.image for image in reads
            }:
                raise ValueError("Temporal output images must not alias inputs")
            self.runtime, self.previous, self.output = runtime, previous, output
            self.closed = False
            self.completion = None
            self.buffers, self.kernels, self.bindings = [], [], []
            self._owns_buffers = policy_buffer is None
            if policy_buffer is not None:
                policy_buffer.require_open()
                if (
                    policy_buffer.runtime is not runtime
                    or policy_buffer.byte_size < 32
                    or not policy_buffer.usage & vk.VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT
                ):
                    raise ValueError(
                        "Temporal policy requires a same-runtime 32-byte uniform buffer"
                    )
                if policy:
                    raise ValueError("Pass a policy buffer or policy values, not both")
                self.buffers.append(policy_buffer)
            runtime.retain(self)
            try:
                if self._owns_buffers:
                    for data in constants:
                        self.buffers.append(runtime.buffer(len(data), data=data))
                source = (
                    files("ordinarylight.shaders")
                    .joinpath("denoiser_relax_temporal.comp.spv")
                    .read_bytes()
                )
                for lobe, radiance in enumerate((diffuse, specular)):
                    normal, depth, material, identity = output.guides
                    old_normal, old_depth, old_material, old_identity = previous.guides
                    images = [
                        radiance,
                        normal,
                        depth,
                        motion,
                        material,
                        previous.images[lobe],
                        old_normal,
                        old_depth,
                        old_material,
                        previous.images[lobe + 2],
                        output.images[lobe],
                        output.images[lobe + 2],
                    ]
                    bindings = {
                        i: VulkanResource.image(image) for i, image in enumerate(images)
                    }
                    bindings[13] = VulkanResource.image(identity)
                    bindings[14] = VulkanResource.image(old_identity)
                    self.bindings.append(bindings)
                    for buffer in self.buffers:
                        self.kernels.append(
                            VulkanKernel(
                                runtime,
                                source,
                                {**bindings, 12: VulkanResource.uniform_buffer(buffer)},
                            )
                        )
            except Exception:
                self.close()
                raise

    def require_open(self):
        self.runtime.require_open()
        if self.closed:
            raise RuntimeError("Temporal stage is closed")
        self.previous.require_open()
        self.output.require_open()
        for kernel in self.kernels:
            kernel.require_open()

    def operation(self, *, reset=False, after=(), extent=None):
        """Create reusable graph work. Frame history advances on submission only.

        Submit successive temporal frames separately. Queue ordering handles
        in-flight previous frames. Declare signal/guide producers in the graph
        or supply their same-runtime completions through after.
        """
        self.require_open()
        after = tuple(after)
        if reset and not self._owns_buffers:
            raise ValueError("Set reset in the externally managed policy buffer")
        width, height = self.output.extent if extent is None else tuple(extent)
        temporal_constants(width, height, False)
        if width > self.output.extent[0] or height > self.output.extent[1]:
            raise ValueError("Active temporal extent exceeds prepared extent")
        if self._owns_buffers and (width, height) != self.output.extent:
            raise ValueError(
                "Active temporal extent requires an external policy buffer"
            )
        selection = [0]
        passes = []
        for lobe, bindings in enumerate(self.bindings):
            uses = {}
            for binding, resource in bindings.items():
                access = (
                    vk.VK_ACCESS_SHADER_WRITE_BIT
                    if binding in (10, 11)
                    else vk.VK_ACCESS_SHADER_READ_BIT
                )
                key = resource.handle
                prior = uses.get(key)
                uses[key] = VulkanResourceUse(
                    resource,
                    vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                    access | (prior.access if prior else 0),
                    vk.VK_IMAGE_LAYOUT_GENERAL,
                )
            # Both immutable policy buffers are declared: the recorder selects one.
            uniforms = [
                VulkanResourceUse(
                    VulkanResource.uniform_buffer(buffer),
                    vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                    vk.VK_ACCESS_UNIFORM_READ_BIT,
                )
                for buffer in self.buffers
            ]

            def record(command, lobe=lobe):
                self.kernels[lobe * len(self.buffers) + selection[0]].bind(command)

            passes.append(
                VulkanPass(
                    f"temporal_{lobe}",
                    (*uses.values(), *uniforms),
                    record,
                    ((width + 7) // 8, (height + 7) // 8, 1),
                )
            )

        def prepare(context):
            selection[0] = (
                int(self.previous.valid and not reset) if self._owns_buffers else 0
            )

        def dependencies():
            return after + tuple(
                c
                for c in (
                    self.completion,
                    self.previous.completion,
                    self.output.completion,
                )
                if c is not None
            )

        def submitted(completion):
            self.completion = completion
            self.output.completion = completion
            self.output.valid = True

        return VulkanOperation(
            passes,
            validate=self.require_open,
            prepare=prepare,
            dependencies=dependencies,
            submitted=submitted,
        )

    def close(self):
        with self.runtime.lock:
            if self.closed:
                return
            if self.completion is not None:
                self.completion.wait()
            for kernel in reversed(self.kernels):
                kernel.close()
            if self._owns_buffers:
                for buffer in reversed(self.buffers):
                    buffer.close()
            self.closed = True
            self.runtime.release(self)

    def __enter__(self):
        self.require_open()
        return self

    def __exit__(self, *_exc):
        self.close()
