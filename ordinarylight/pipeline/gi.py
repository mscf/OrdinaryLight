"""Composition boundary for native GI command recording.

Stages borrow prepared backend resources. They record on the caller's command
buffer; allocation, submission, presentation and history retirement belong to
that backend. Logical dependencies do not replace Vulkan memory barriers.
"""

from dataclasses import dataclass, field
from types import MappingProxyType

from ._core import RenderPipeline, RenderStage


@dataclass(eq=False)
class GiImage:
    """Borrowed storage image, compatible with ``VulkanResource.image``.

    ``width``/``height`` describe allocation capacity, not the active region.
    The allocation must remain alive through every submitted consumer. A view
    cannot be retained across resize, replacement, or renderer close. It grants
    no ownership and has no close method.
    """

    runtime: object
    image: object
    view: object
    width: int
    height: int
    format: int
    layout: int
    usage: int
    _validate: object = field(repr=False)

    def require_open(self):
        self._validate()


@dataclass(frozen=True)
class GiBuffer:
    """Borrowed GPU buffer; lifetime matches the native allocation generation."""

    runtime: object
    buffer: object
    byte_size: int
    usage: int
    _validate: object = field(repr=False, compare=False)

    def require_open(self):
        self._validate()


@dataclass(frozen=True)
class GiFrame:
    """Per-recording metadata; history flags describe eligible previous-frame input.

    Resource mappings are borrowed, read-only views valid only during recording.
    Do not retain a frame across resize, resource replacement or presenter close.
    """

    command: object
    device: object
    slot: int
    render_extent: tuple[int, int]
    output_extent: tuple[int, int]
    resources: object
    previous_resources: object
    restir_history_valid: bool = False
    denoiser_history_valid: bool = False
    reconstruction_history_valid: bool = False
    images: object = field(default_factory=dict)
    buffers: object = field(default_factory=dict)
    _recordings: list = field(default_factory=list, repr=False, compare=False)

    def __post_init__(self):
        for name in ("render_extent", "output_extent"):
            extent = tuple(getattr(self, name))
            if len(extent) != 2 or any(
                not isinstance(n, int) or n <= 0 for n in extent
            ):
                raise ValueError(f"{name} must contain two positive integers")
            object.__setattr__(self, name, extent)
        for name in ("resources", "previous_resources", "images", "buffers"):
            object.__setattr__(self, name, MappingProxyType(dict(getattr(self, name))))

    def process_hdr(self, pipeline, stage):
        """Insert a stage after lighting/denoising and before any upscaling.

        The stage reads/writes ``images['hdr']`` in linear scene radiance.
        It must record its own barriers and preserve GENERAL layout. The same
        upstream transport and history are used with and without this stage.
        """
        target = "gi.upscale" if "gi.upscale" in pipeline.stage_names else "gi.reconstruct"
        if target in pipeline.stage_names:
            return pipeline.insert_before(target, stage)
        return RenderPipeline((*pipeline.stages, stage), pipeline.initial_resources)

    def record_graph(self, graph):
        """Record a prepared application graph into this native GI frame.

        All resources and prior completions must use this frame's runtime/queue.
        External semaphore waits/signals are not supported by this boundary.
        Operation prepare callbacks must not allocate, submit, or wait. Native
        submission publishes completion even when recorded commands are reused.
        Borrowed GI images must finish in GENERAL layout.
        """
        from .graph import CompiledVulkanGraph
        if not isinstance(graph, CompiledVulkanGraph):
            raise TypeError("Expected a compiled VulkanGraph")
        runtime = self.images["hdr"].runtime
        recording = graph.prepare_recording(runtime)
        if recording.wait_semaphores or recording.signal_semaphores:
            raise ValueError("Native GI subgraphs cannot import semaphore operations")
        if any(getattr(d, "runtime", None) is not runtime for d in recording.dependencies):
            raise ValueError("GI subgraph dependencies must use the native runtime queue")
        # The graph supplies entry/internal dependencies; restore the native
        # boundary explicitly because its downstream consumer is outside graph.
        import vulkan as vk
        for node in graph.nodes:
            for stage in node.operation.passes:
                for use in stage.uses:
                    if isinstance(use.resource.owner, GiImage) and use.layout != vk.VK_IMAGE_LAYOUT_GENERAL:
                        raise ValueError("Borrowed GI images must remain in GENERAL layout")
        recording.record(self.command)
        vk.vkCmdPipelineBarrier(
            self.command, vk.VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
            vk.VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0,
            1, [vk.VkMemoryBarrier(
                srcAccessMask=vk.VK_ACCESS_MEMORY_WRITE_BIT,
                dstAccessMask=vk.VK_ACCESS_MEMORY_READ_BIT | vk.VK_ACCESS_MEMORY_WRITE_BIT,
            )], 0, None, 0, None,
        )
        self._recordings.append(recording)


def create_gi_pipeline(
    *, trace, reconstruct=None, indirect=None, denoise=None, upscale=None,
    snapshot_hdr=None,
):
    """Compose prepared recorders, each accepting a context with a ``frame``.

    Returned RenderStage objects can be reused in another RenderPipeline. Trace
    includes primary/secondary transport, ReSTIR and signal preparation. Denoise
    includes temporal accumulation, spatial filtering and HDR composition.
    Reconstruction includes spatial upscaling (if selected), tone mapping and
    display encoding. Temporal upscaling is a distinct optional HDR stage.
    Omit reconstruction to compose a lighting-only pipeline. This changes only
    recording; the caller must still own preparation, submission and retirement.
    """
    stages = [
        RenderStage(
            "gi.trace",
            reads={"gi.scene", "gi.camera", "gi.history"},
            writes={"gi.hdr", "gi.guides", "gi.reservoirs"},
            recorder=trace,
        )
    ]
    if indirect is not None:
        stages.append(
            RenderStage(
                "gi.indirect",
                reads={"gi.hdr", "gi.guides", "gi.reservoirs"},
                writes={"gi.hdr", "gi.reservoirs"},
                recorder=indirect,
            )
        )
    if snapshot_hdr is not None:
        stages.append(RenderStage(
            "gi.snapshot_hdr", reads={"gi.hdr"}, writes={"gi.raw_hdr"},
            recorder=snapshot_hdr,
        ))
    if denoise is not None:
        stages.append(
            RenderStage(
                "gi.denoise",
                reads={"gi.hdr", "gi.guides", "gi.history"},
                writes={"gi.hdr", "gi.denoiser_history"},
                recorder=denoise,
            )
        )
    if upscale is not None:
        stages.append(
            RenderStage(
                "gi.upscale",
                reads={"gi.hdr", "gi.guides", "gi.history"},
                writes={"gi.upscaled_hdr"},
                recorder=upscale,
            )
        )
    if reconstruct is not None:
        stages.append(RenderStage(
            "gi.reconstruct",
            reads={
                "gi.upscaled_hdr" if upscale is not None else "gi.hdr",
                "gi.guides",
                "gi.history",
            },
            writes={"gi.display", "gi.reconstruction_history"},
            recorder=reconstruct,
        ))
    return RenderPipeline(
        stages, initial_resources={"gi.scene", "gi.camera", "gi.history"}
    )


def record_gi_pipeline(pipeline, frame, *, builder=None, required_outputs=("gi.display",)):
    """Record a default or application-composed pipeline without submitting.

    A builder receives (default_pipeline, frame) and returns a RenderPipeline.
    Custom stages own their Vulkan synchronization. The backend owns history;
    builders must preserve stages needed by the active backend configuration.
    """
    if builder is not None:
        pipeline = builder(pipeline, frame)
    if not isinstance(pipeline, RenderPipeline):
        raise TypeError("GI pipeline builder must return a RenderPipeline")
    missing = set(required_outputs) - pipeline.output_resources
    if missing:
        raise ValueError("GI pipeline must produce " + ", ".join(sorted(missing)))
    pipeline.record({"frame": frame})
