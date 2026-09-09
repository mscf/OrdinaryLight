"""Resource-backed FSR2 preparation and temporal upscaling."""

from importlib.resources import files
import struct
import math
from operator import index
import vulkan as vk

from ._fsr2 import Fsr2Context
from .kernel import VulkanKernel
from ..pipeline.graph import VulkanOperation
from ..pipeline.vulkan import VulkanPass, VulkanResource, VulkanResourceUse


class VulkanFsr2:
    """One serial temporal stream with explicit per-slot input bindings.

    Each input mapping contains hdr, view_z, motion and normal_roughness images.
    Inputs may be larger than extent. Outputs and preparation images are owned
    by this component. Create a fresh operation per frame and submit it before
    recording the next frame; command replay and multi-frame graphs are unsupported.
    """

    def __init__(self, runtime, *, inputs, extent, output_extent):
        self.runtime = runtime
        self.closed = False
        self.context = None
        self.frames, self.kernels, self.retained = [], [], set()
        self.completion = None
        with runtime.lock:
            runtime.require_open()
            self.extent = Fsr2Context._extent(extent)
            self.output_extent = Fsr2Context._extent(output_extent)
            self.inputs = tuple(dict(frame) for frame in inputs)
            if not self.inputs:
                raise ValueError("FSR2 requires at least one input slot")
            formats = dict(
                hdr=vk.VK_FORMAT_R16G16B16A16_SFLOAT,
                view_z=vk.VK_FORMAT_R32_SFLOAT,
                motion=vk.VK_FORMAT_R16G16B16A16_SFLOAT,
                normal_roughness=vk.VK_FORMAT_R16G16B16A16_SFLOAT,
            )
            for frame in self.inputs:
                for name, fmt in formats.items():
                    image = frame[name]
                    image.require_open()
                    usage = (
                        vk.VK_IMAGE_USAGE_SAMPLED_BIT
                        if name == "hdr"
                        else vk.VK_IMAGE_USAGE_STORAGE_BIT
                    )
                    if (
                        image.runtime is not runtime
                        or image.format != fmt
                        or not image.usage & usage
                        or image.width < self.extent[0]
                        or image.height < self.extent[1]
                    ):
                        raise ValueError(
                            f"Invalid FSR2 {name} format, extent, usage or runtime"
                        )
            runtime.retain(self)
            try:
                self.context = Fsr2Context(
                    runtime.physical_device,
                    runtime.device,
                    self.extent,
                    self.output_extent,
                )
                for frame in self.inputs:
                    for image in frame.values():
                        if hasattr(image, "retain") and image not in self.retained:
                            image.retain(self)
                            self.retained.add(image)
                    images = []
                    self.frames.append(images)
                    for fmt in (
                        vk.VK_FORMAT_R32_SFLOAT,
                        vk.VK_FORMAT_R16G16_SFLOAT,
                        vk.VK_FORMAT_R8_UNORM,
                        vk.VK_FORMAT_R16G16B16A16_SFLOAT,
                    ):
                        size = self.output_extent if len(images) == 3 else self.extent
                        images.append(runtime.image(*size, format=fmt))
                    bindings = [
                        frame[n] for n in ("view_z", "motion", "normal_roughness")
                    ]
                    bindings += images[:3]
                    self.kernels.append(
                        VulkanKernel(
                            runtime,
                            files("ordinarylight.shaders")
                            .joinpath("fsr2_prepare.comp.spv")
                            .read_bytes(),
                            {
                                i: VulkanResource.image(v)
                                for i, v in enumerate(bindings)
                            },
                            push_constant_size=16,
                        )
                    )
            except Exception:
                self.close()
                raise
        self.outputs = tuple(frame[3] for frame in self.frames)

    def require_open(self):
        self.runtime.require_open()
        if self.closed:
            raise RuntimeError("FSR2 stage is closed")
        for frame in self.inputs:
            for image in frame.values():
                image.require_open()

    def jitter(self, sequence):
        self.require_open()
        return self.context.jitter(sequence)

    def operation(self, *, slot=0, jitter, fov_y, dt_ms, reset=False, after=()):
        """Prepare reversed finite depth/motion/reactivity and upscale linear HDR.

        fov_y is in radians; dt_ms is explicit simulation/display frame time.
        Motion uses the native guide convention: pixel displacement including
        current jitter, and previous view depth in z. Depth range is .1–10000.
        """
        self.require_open()
        slot = index(slot)
        if not 0 <= slot < len(self.frames):
            raise ValueError("Invalid FSR2 slot")
        jitter = tuple(jitter)
        if (
            len(jitter) != 2
            or not all(math.isfinite(v) for v in jitter)
            or not math.isfinite(fov_y)
            or not 0 < fov_y < math.pi
            or not math.isfinite(dt_ms)
            or not 0 < dt_ms <= 1000
        ):
            raise ValueError("Invalid FSR2 jitter, field of view or frame time")
        constants = struct.pack("2f2I", *jitter, *self.extent)
        frame, images, kernel = self.inputs[slot], self.frames[slot], self.kernels[slot]
        dispatch_images = [frame["hdr"], *images]
        token = None
        recorded = False

        def uses(reads, writes):
            merged = {}
            for image, access in [(v, vk.VK_ACCESS_SHADER_READ_BIT) for v in reads] + [
                (v, vk.VK_ACCESS_SHADER_WRITE_BIT) for v in writes
            ]:
                resource = VulkanResource.image(image)
                old = merged.get(image.image)
                merged[image.image] = VulkanResourceUse(
                    resource,
                    vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                    access | (old.access if old else 0),
                    vk.VK_IMAGE_LAYOUT_GENERAL,
                )
            return tuple(merged.values())

        def prepare(command):
            nonlocal recorded
            if recorded:
                raise RuntimeError(
                    "FSR2 operations cannot be replayed; create a new operation"
                )
            recorded = True
            kernel.bind(command, constants)

        def dispatch(command):
            nonlocal token
            token = self.context.record(
                command,
                [image.image for image in dispatch_images],
                [image.view for image in dispatch_images],
                jitter=jitter,
                fov_y=fov_y,
                dt_ms=dt_ms,
                reset=reset,
            )

        def submitted(completion):
            self.context.submitted(token)
            self.completion = completion

        def prepare_graph(context):
            if self in context:
                raise ValueError(
                    "Only one frame per FSR2 context is allowed in a graph"
                )
            context[self] = True

        after = tuple(after)
        return VulkanOperation(
            [
                VulkanPass(
                    "fsr2.prepare",
                    uses(
                        [frame[n] for n in ("view_z", "motion", "normal_roughness")],
                        images[:3],
                    ),
                    prepare,
                    ((self.extent[0] + 7) // 8, (self.extent[1] + 7) // 8, 1),
                ),
                VulkanPass(
                    "fsr2.upscale", uses(dispatch_images[:4], images[3:]), dispatch
                ),
            ],
            validate=self.require_open,
            prepare=prepare_graph,
            dependencies=lambda: after
            + ((self.completion,) if self.completion is not None else ()),
            submitted=submitted,
        )

    def close(self):
        with self.runtime.lock:
            if self.closed:
                return
            # Refuse destruction while downstream kernels still borrow outputs.
            for frame in self.frames:
                for image in frame:
                    if image._borrowers - set(self.kernels):
                        raise RuntimeError(
                            "Close FSR2 output borrowers before the stage"
                        )
            if self.completion is not None:
                self.completion.wait()
            for kernel in reversed(self.kernels):
                kernel.close()
            if self.context is not None:
                self.context.close()
            for frame in self.frames:
                for image in frame:
                    image.close()
            for image in self.retained:
                image.release(self)
            self.runtime.release(self)
            self.closed = True

    def __enter__(self):
        self.require_open()
        return self

    def __exit__(self, *_exc):
        self.close()
