"""Portable CPU renderer implementing the high-level renderer contract."""

from dataclasses import dataclass

from .path_tracer import ReferencePathTracer
from ..base import RendererImplementation, RendererImplementationInfo


@dataclass(frozen=True)
class ReferenceConfig:
    samples_per_pixel: int = 1
    max_bounces: int = 3
    seed: int = 1

    def __post_init__(self):
        if not 1 <= int(self.samples_per_pixel) <= 4096:
            raise ValueError("samples_per_pixel must be between 1 and 4096")
        if not 1 <= int(self.max_bounces) <= 64:
            raise ValueError("max_bounces must be between 1 and 64")
        if int(self.seed) < 0:
            raise ValueError("seed cannot be negative")


class CpuReferenceRenderer(RendererImplementation):
    """Deterministic CPU renderer for portability and correctness checks."""

    available_outputs = ("color",)
    implementation = RendererImplementationInfo(
        name="cpu-reference", family="reference", graphics_api="cpu",
    )

    def __init__(self, config=None, **options):
        if config is not None and options:
            raise TypeError("pass config or reference options, not both")
        self.config = config or ReferenceConfig(**options)
        if not isinstance(self.config, ReferenceConfig):
            raise TypeError("config must be a ReferenceConfig")
        self.device = "CPU reference"
        self.last_timings = {}
        self._closed = False

    @property
    def capabilities(self):
        return {
            "renderer": "cpu-reference",
            "outputs": self.available_outputs,
            "features": frozenset({
                "offscreen_rendering", "instancing", "volumes",
                "volume_scattering",
            }),
            "limits": {
                "max_bounces": 64,
                "max_samples_per_pixel": 4096,
            },
            "device": self.device,
        }

    def render_frame(
        self, scene, camera, width, height, *, samples=None, frame_index=0,
    ):
        if self._closed:
            raise RuntimeError("reference renderer is closed")
        sample_count = self.config.samples_per_pixel if samples is None else int(samples)
        tracer = ReferencePathTracer(seed=self.config.seed + int(frame_index))
        return tracer.render_hdr(
            scene, camera, width, height, samples=sample_count,
            max_bounces=self.config.max_bounces,
        )

    # Retained as a low-level compatibility spelling during the 0.x series.
    render_wavefront = render_frame

    def close(self):
        self._closed = True


__all__ = ["CpuReferenceRenderer", "ReferenceConfig"]
