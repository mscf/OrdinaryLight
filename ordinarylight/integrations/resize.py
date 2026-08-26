"""Window-resize policies that do not depend on a windowing backend."""

from dataclasses import dataclass, field
import math


@dataclass
class ResizeRecreationGate:
    """Delay expensive surface recreation until an extent stops changing."""

    settle_seconds: float = 0.15
    _pending_extent: tuple[int, int] | None = field(default=None, init=False)
    _changed_at: float = field(default=0.0, init=False)

    def __post_init__(self):
        if not math.isfinite(self.settle_seconds) or self.settle_seconds < 0.0:
            raise ValueError("settle_seconds must be finite and non-negative")

    @property
    def pending_extent(self):
        return self._pending_extent

    def should_render(self, extent, now, *, resources_allocated):
        """Return true when rendering cannot trigger a transient recreation."""
        extent = tuple(int(value) for value in extent)
        if len(extent) != 2 or extent[0] < 1 or extent[1] < 1:
            raise ValueError("extent must contain two positive dimensions")
        now = float(now)
        if not math.isfinite(now):
            raise ValueError("now must be finite")
        if not resources_allocated:
            self._pending_extent = extent
            self._changed_at = now - self.settle_seconds
            return True
        if extent != self._pending_extent:
            self._pending_extent = extent
            self._changed_at = now
        return now - self._changed_at >= self.settle_seconds

