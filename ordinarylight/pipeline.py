"""Composable render-pipeline scheduling primitives.

Stages declare logical resource dependencies independently of a graphics API.
Backends attach record callbacks that encode their commands into a backend
context. This keeps scheduling and dependency validation reusable while Vulkan
objects remain private to the Vulkan backend.
"""

from dataclasses import dataclass, field
from typing import Callable, Iterable, Mapping, Any


@dataclass(frozen=True)
class RenderStage:
    """One ordered renderer operation with declared logical resource use."""

    name: str
    reads: frozenset[str] = field(default_factory=frozenset)
    writes: frozenset[str] = field(default_factory=frozenset)
    recorder: Callable[[Mapping[str, Any]], None] | None = field(
        default=None, compare=False, repr=False
    )

    def __post_init__(self):
        if not self.name or not self.name.strip():
            raise ValueError("render stage name cannot be empty")
        object.__setattr__(self, "reads", frozenset(self.reads))
        object.__setattr__(self, "writes", frozenset(self.writes))
        if self.recorder is not None and not callable(self.recorder):
            raise TypeError("render stage recorder must be callable")

    def record(self, context):
        if self.recorder is None:
            raise RuntimeError(f"render stage '{self.name}' has no backend recorder")
        self.recorder(context)


class RenderPipeline:
    """Validated, immutable ordered collection of render stages."""

    def __init__(self, stages: Iterable[RenderStage], initial_resources=()):
        self.stages = tuple(stages)
        self.initial_resources = frozenset(initial_resources)
        self._validate()

    def _validate(self):
        available = set(self.initial_resources)
        names = set()
        for stage in self.stages:
            if not isinstance(stage, RenderStage):
                raise TypeError("pipeline stages must be RenderStage objects")
            if stage.name in names:
                raise ValueError(f"duplicate render stage name: {stage.name}")
            missing = stage.reads - available
            if missing:
                resources = ", ".join(sorted(missing))
                raise ValueError(
                    f"render stage '{stage.name}' reads unavailable resources: {resources}"
                )
            names.add(stage.name)
            available.update(stage.writes)
        self.output_resources = frozenset(available)

    @property
    def stage_names(self):
        return tuple(stage.name for stage in self.stages)

    def record(self, context):
        for stage in self.stages:
            stage.record(context)

    def insert_before(self, target, stage):
        """Return a new pipeline with ``stage`` inserted before ``target``."""
        try:
            index = self.stage_names.index(target)
        except ValueError as error:
            raise KeyError(f"unknown render stage: {target}") from error
        return RenderPipeline(
            self.stages[:index] + (stage,) + self.stages[index:],
            self.initial_resources,
        )
