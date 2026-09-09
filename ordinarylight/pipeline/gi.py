"""Composition boundary for native GI command recording.

Stages borrow prepared backend resources. They record on the caller's command
buffer; allocation, submission, presentation and history retirement belong to
that backend. Logical dependencies do not replace Vulkan memory barriers.
"""

from dataclasses import dataclass
from types import MappingProxyType

from ._core import RenderPipeline, RenderStage


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

    def __post_init__(self):
        for name in ("render_extent", "output_extent"):
            extent = tuple(getattr(self, name))
            if len(extent) != 2 or any(
                not isinstance(n, int) or n <= 0 for n in extent
            ):
                raise ValueError(f"{name} must contain two positive integers")
            object.__setattr__(self, name, extent)
        for name in ("resources", "previous_resources"):
            object.__setattr__(self, name, MappingProxyType(dict(getattr(self, name))))


def create_gi_pipeline(
    *, trace, reconstruct, indirect=None, denoise=None, upscale=None
):
    """Compose prepared recorders, each accepting a context with a ``frame``.

    Returned RenderStage objects can be reused in another RenderPipeline. Trace
    includes primary/secondary transport, ReSTIR and signal preparation. Denoise
    includes temporal accumulation, spatial filtering and HDR composition.
    Reconstruction includes spatial upscaling (if selected), tone mapping and
    display encoding. Temporal upscaling is a distinct optional HDR stage.
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
    stages.append(
        RenderStage(
            "gi.reconstruct",
            reads={
                "gi.upscaled_hdr" if upscale is not None else "gi.hdr",
                "gi.guides",
                "gi.history",
            },
            writes={"gi.display", "gi.reconstruction_history"},
            recorder=reconstruct,
        )
    )
    return RenderPipeline(
        stages, initial_resources={"gi.scene", "gi.camera", "gi.history"}
    )


def record_gi_pipeline(pipeline, frame, *, builder=None):
    """Record a default or application-composed pipeline without submitting.

    A builder receives (default_pipeline, frame) and returns a RenderPipeline.
    Custom stages own their Vulkan synchronization. The backend owns history;
    builders must preserve stages needed by the active backend configuration.
    """
    if builder is not None:
        pipeline = builder(pipeline, frame)
    if not isinstance(pipeline, RenderPipeline):
        raise TypeError("GI pipeline builder must return a RenderPipeline")
    if "gi.display" not in pipeline.output_resources:
        raise ValueError("GI presentation pipeline must produce gi.display")
    pipeline.record({"frame": frame})
