"""Independent ReLAX spatial filtering and HDR composition on VulkanRuntime."""

from importlib.resources import files
from dataclasses import replace

import vulkan as vk

from ..denoising.spatial import atrous_constants, compose_constants, spatial_settings
from ..pipeline.graph import VulkanOperation
from ..pipeline.vulkan import VulkanPass, VulkanResource, VulkanResourceUse
from .kernel import VulkanKernel


class VulkanRelaxSpatial:
    """Prepared spatial denoiser; no scene, presenter, camera or temporal history.

    All six image arguments are distinct, caller-owned storage images on runtime.
    Diffuse/specular are linear RGBA16F (alpha carries signal metadata), guides
    are RGBA16F normal/roughness, R32F view-space depth, and R32_UINT material IDs.
    Output is initialized RGBA16F HDR: zero-depth background is preserved.
    Owns immutable descriptor/pipeline bindings. By default it owns four scratch
    images; scratch=(diffuse_a, diffuse_b, specular_a, specular_b) borrows those
    allocations instead. Borrowed scratch must outlive all submitted operations.
    """

    def __init__(
        self,
        runtime,
        *,
        diffuse,
        specular,
        normal_roughness,
        view_z,
        material,
        output,
        extent=None,
        iterations=3,
        color_weight=4.0,
        scratch=None,
    ):
        with runtime.lock:
            runtime.require_open()
            width, height = (
                extent if extent is not None else (output.width, output.height)
            )
            self.width, self.height, self.iterations, self.color_weight = (
                spatial_settings(width, height, iterations, color_weight)
            )
            images = (diffuse, specular, normal_roughness, view_z, material, output)
            formats = (vk.VK_FORMAT_R16G16B16A16_SFLOAT,) * 3 + (
                vk.VK_FORMAT_R32_SFLOAT,
                vk.VK_FORMAT_R32_UINT,
                vk.VK_FORMAT_R16G16B16A16_SFLOAT,
            )
            for image, image_format in zip(images, formats):
                image.require_open()
                if image.runtime is not runtime or image.format != image_format:
                    raise ValueError(
                        "ReLAX images must use the required format and runtime"
                    )
                if not image.usage & vk.VK_IMAGE_USAGE_STORAGE_BIT:
                    raise ValueError("ReLAX images require storage usage")
                if image.width < self.width or image.height < self.height:
                    raise ValueError("ReLAX image is smaller than the active extent")
            if len({image.image for image in images}) != len(images):
                raise ValueError("ReLAX bindings must not alias")
            self.runtime, self.output = runtime, output
            self.inputs = images
            self.closed = False
            self.completion = None
            self.scratch, self.kernels, self.passes = [], [], []
            self._owns_scratch = scratch is None
            if scratch is not None:
                scratch = tuple(scratch)
                if len(scratch) != 4:
                    raise ValueError("Spatial scratch requires four images")
                for image in scratch:
                    image.require_open()
                    if (
                        image.runtime is not runtime
                        or image.format != vk.VK_FORMAT_R16G16B16A16_SFLOAT
                        or not image.usage & vk.VK_IMAGE_USAGE_STORAGE_BIT
                        or image.width < self.width
                        or image.height < self.height
                    ):
                        raise ValueError("Invalid spatial scratch image")
                if (
                    len({image.image for image in (*images, *scratch)})
                    != len(images) + 4
                ):
                    raise ValueError("Spatial scratch must not alias bindings")
                self.scratch.extend(scratch)
            runtime.retain(self)
            try:
                if self._owns_scratch:
                    for _ in range(4):
                        self.scratch.append(
                            runtime.image(
                                self.width,
                                self.height,
                                format=vk.VK_FORMAT_R16G16B16A16_SFLOAT,
                            )
                        )
                atrous = (
                    files("ordinarylight.shaders")
                    .joinpath("denoiser_relax_atrous.comp.spv")
                    .read_bytes()
                )
                compose = (
                    files("ordinarylight.shaders")
                    .joinpath("denoiser_relax_compose.comp.spv")
                    .read_bytes()
                )
                current = [diffuse, specular]
                for iteration in range(self.iterations):
                    for lobe in range(2):
                        target = self.scratch[lobe * 2 + iteration % 2]
                        self._add_pass(
                            f"atrous_{iteration}_{lobe}",
                            atrous,
                            [current[lobe], normal_roughness, view_z, material, target],
                            4,
                            atrous_constants(
                                self.width,
                                self.height,
                                iteration,
                                lobe,
                                self.color_weight,
                            ),
                        )
                        current[lobe] = target
                self.filtered_diffuse, self.filtered_specular = current
                self._add_pass(
                    "compose",
                    compose,
                    [*current, view_z, output],
                    3,
                    compose_constants(self.width, self.height),
                    preserve=True,
                )
            except Exception:
                self.close()
                raise

    def _add_pass(
        self, name, spirv, images, output_binding, constants, *, preserve=False
    ):
        resources = {i: VulkanResource.image(image) for i, image in enumerate(images)}
        kernel = VulkanKernel(
            self.runtime, spirv, resources, push_constant_size=len(constants)
        )
        self.kernels.append(kernel)
        uses = tuple(
            VulkanResourceUse(
                resource,
                vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                (
                    (vk.VK_ACCESS_SHADER_READ_BIT if preserve else 0)
                    | vk.VK_ACCESS_SHADER_WRITE_BIT
                )
                if i == output_binding
                else vk.VK_ACCESS_SHADER_READ_BIT,
                vk.VK_IMAGE_LAYOUT_GENERAL,
            )
            for i, resource in resources.items()
        )
        self.passes.append(
            VulkanPass(
                name,
                uses,
                lambda command: kernel.bind(command, constants),
                ((self.width + 7) // 8, (self.height + 7) // 8, 1),
            )
        )

    def require_open(self):
        self.runtime.require_open()
        if self.closed:
            raise RuntimeError("ReLAX spatial stage is closed")
        for kernel in self.kernels:
            kernel.require_open()

    def operation(self, *, after=(), extent=None):
        """Return graph work with explicit image hazards and completion tracking.

        Caller supplies producer ordering, either graph edges/resource versions
        or prior same-runtime completions. Repeated execution is serial on the
        runtime queue; use distinct stages for independently in-flight scratch.
        """
        return self._operation(self._passes_for_extent(extent), after)

    def filter_operation(self, *, after=(), extent=None):
        """Filter lobes only; results are filtered_diffuse and filtered_specular."""
        return self._operation(self._passes_for_extent(extent)[:-1], after)

    def compose_operation(self, *, after=(), extent=None):
        """Compose filtered lobes into output; order after filter_operation."""
        return self._operation(self._passes_for_extent(extent)[-1:], after)

    def _passes_for_extent(self, extent):
        if extent is None:
            return self.passes
        width, height, _, _ = spatial_settings(
            *extent, self.iterations, self.color_weight
        )
        if width > self.width or height > self.height:
            raise ValueError("Active spatial extent exceeds prepared extent")
        result = []
        for index, (stage, kernel) in enumerate(zip(self.passes, self.kernels)):
            constants = (
                compose_constants(width, height)
                if index == len(self.passes) - 1
                else atrous_constants(
                    width, height, index // 2, index % 2, self.color_weight
                )
            )
            result.append(
                replace(
                    stage,
                    record=lambda command,
                    kernel=kernel,
                    constants=constants: kernel.bind(command, constants),
                    workgroups=((width + 7) // 8, (height + 7) // 8, 1),
                )
            )
        return result

    def _operation(self, passes, after):
        self.require_open()
        after = tuple(after)
        return VulkanOperation(
            passes,
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
            for kernel in reversed(self.kernels):
                kernel.close()
            if self._owns_scratch:
                for image in reversed(self.scratch):
                    image.close()
            self.closed = True
            self.runtime.release(self)

    def __enter__(self):
        self.require_open()
        return self

    def __exit__(self, *_exc):
        self.close()
