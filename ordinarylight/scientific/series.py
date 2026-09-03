"""Time-series and multi-channel scalar-field datasets."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .scalar_field import ScalarField3D


@dataclass(slots=True)
class ScalarFieldSeries:
    """A canonical ``(time, channel, z, y, x)`` scientific dataset.

    Use :meth:`from_array` to adapt arrays whose explicit axis order is one of
    ``zyx``, ``tzyx``, ``czyx``, or ``tczyx``. Exact frames remain views of the
    source allocation whenever NumPy can represent the requested transpose as
    a view, including memory-mapped arrays.
    """

    data: np.ndarray = field(repr=False)
    times: np.ndarray
    channels: tuple[str, ...]
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0)
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0)
    direction: np.ndarray = field(default_factory=lambda: np.eye(3), repr=False)
    channel_units: tuple[str | None, ...] | None = None
    time_unit: str | None = None
    name: str | None = None
    metadata: dict = field(default_factory=dict, repr=False)
    revision: int = field(default=0, init=False)
    _updates: list = field(default_factory=list, init=False, repr=False)

    def __post_init__(self):
        data = np.asarray(self.data)
        if data.ndim != 5 or any(size < 1 for size in data.shape[:2]) or any(
            size < 2 for size in data.shape[2:]
        ):
            raise ValueError("data must have shape (time, channel, z, y, x) with spatial dimensions >= 2")
        if not np.issubdtype(data.dtype, np.number):
            raise TypeError("data must have a numeric dtype")
        times = np.asarray(self.times, np.float64)
        if times.shape != (data.shape[0],) or not np.isfinite(times).all():
            raise ValueError("times must contain one finite value per frame")
        if len(times) > 1 and np.any(np.diff(times) <= 0.0):
            raise ValueError("times must be strictly increasing")
        channels = tuple(self.channels)
        if len(channels) != data.shape[1] or any(
            not isinstance(value, str) or not value for value in channels
        ):
            raise ValueError("channels must contain one nonempty name per channel")
        if len(set(channels)) != len(channels):
            raise ValueError("channel names must be unique")
        units = self.channel_units
        if units is None:
            units = (None,) * len(channels)
        units = tuple(units)
        if len(units) != len(channels) or any(
            value is not None and not isinstance(value, str) for value in units
        ):
            raise ValueError("channel_units must contain one string or None per channel")
        if self.time_unit is not None and not isinstance(self.time_unit, str):
            raise TypeError("time_unit must be a string or None")
        if not hasattr(self.metadata, "items"):
            raise TypeError("metadata must be a mapping")
        # Reuse ScalarField3D's coordinate validation without copying data.
        validated = ScalarField3D(
            data[0, 0], spacing=self.spacing, origin=self.origin,
            direction=self.direction,
        )
        self.data = data
        self.times = times
        self.channels = channels
        self.channel_units = units
        self.spacing = validated.spacing
        self.origin = validated.origin
        self.direction = validated.direction
        self.metadata = dict(self.metadata)

    @classmethod
    def from_array(
        cls, data, *, axis_order, times=None, channels=None,
        spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0), direction=None,
        channel_units=None, time_unit=None, name=None, metadata=None,
    ):
        """Adapt an explicitly labeled NumPy array into canonical axis order."""
        axis_order = str(axis_order).lower()
        if len(axis_order) != len(set(axis_order)) or set(axis_order) not in (
            set("zyx"), set("tzyx"), set("czyx"), set("tczyx"),
        ):
            raise ValueError("axis_order must be zyx, tzyx, czyx, or tczyx")
        array = np.asarray(data)
        if array.ndim != len(axis_order):
            raise ValueError("axis_order length must match data dimensions")
        present = list(axis_order)
        for missing in ("t", "c"):
            if missing not in present:
                array = np.expand_dims(array, axis=0)
                present.insert(0, missing)
        array = np.transpose(array, tuple(present.index(axis) for axis in "tczyx"))
        frame_count, channel_count = array.shape[:2]
        if times is None:
            times = np.arange(frame_count, dtype=np.float64)
        if channels is None:
            channels = tuple(f"channel_{index}" for index in range(channel_count))
        return cls(
            array, times, tuple(channels), spacing=spacing, origin=origin,
            direction=np.eye(3) if direction is None else direction,
            channel_units=channel_units, time_unit=time_unit, name=name,
            metadata={} if metadata is None else metadata,
        )

    def channel_index(self, channel):
        if isinstance(channel, str):
            try:
                return self.channels.index(channel)
            except ValueError as error:
                raise KeyError(f"unknown channel {channel!r}") from error
        if isinstance(channel, bool):
            raise TypeError("channel must be a name or integer")
        index = int(channel)
        if index != channel or not 0 <= index < len(self.channels):
            raise IndexError("channel index is outside the dataset")
        return index

    def frame(self, time_index, channel=0):
        """Return an exact frame as a coordinate-aware, zero-copy field view."""
        if isinstance(time_index, bool):
            raise TypeError("time_index must be an integer")
        time_index = int(time_index)
        if not 0 <= time_index < len(self.times):
            raise IndexError("time index is outside the dataset")
        channel_index = self.channel_index(channel)
        return ScalarField3D(
            self.data[time_index, channel_index], spacing=self.spacing,
            origin=self.origin, direction=self.direction,
            unit=self.channel_units[channel_index],
            name=f"{self.name + ':' if self.name else ''}{self.channels[channel_index]}",
            metadata={
                **self.metadata, "series_name": self.name,
                "time_index": time_index, "time": float(self.times[time_index]),
                "time_unit": self.time_unit, "channel_index": channel_index,
                "channel": self.channels[channel_index],
            },
        )

    def at_time(self, time, channel=0, *, interpolation="nearest"):
        """Select or linearly interpolate a field at a physical time."""
        time = float(time)
        if not np.isfinite(time):
            raise ValueError("time must be finite")
        if interpolation not in {"nearest", "linear"}:
            raise ValueError("interpolation must be 'nearest' or 'linear'")
        if time < self.times[0] or time > self.times[-1]:
            raise ValueError("time is outside the dataset")
        upper = int(np.searchsorted(self.times, time, side="left"))
        if upper == 0 or self.times[upper] == time or interpolation == "nearest":
            if interpolation == "nearest" and upper not in (0, len(self.times)):
                lower = upper - 1
                upper = lower if time - self.times[lower] <= self.times[upper] - time else upper
            return self.frame(min(upper, len(self.times) - 1), channel)
        lower = upper - 1
        channel_index = self.channel_index(channel)
        weight = (time - self.times[lower]) / (self.times[upper] - self.times[lower])
        values = np.asarray(
            self.data[lower, channel_index] * (1.0 - weight)
            + self.data[upper, channel_index] * weight,
            np.result_type(self.data.dtype, np.float32),
        )
        result = ScalarField3D(
            values, spacing=self.spacing, origin=self.origin,
            direction=self.direction, unit=self.channel_units[channel_index],
            name=f"{self.name + ':' if self.name else ''}{self.channels[channel_index]}",
            metadata={**self.metadata, "series_name": self.name, "time": time,
                      "time_unit": self.time_unit, "channel_index": channel_index,
                      "channel": self.channels[channel_index],
                      "interpolation": "linear", "source_frames": [lower, upper],
                      "weight": float(weight)},
        )
        return result

    def update(self, time_index, channel, offset, values):
        """Update one frame/channel z/y/x region and record dataset revision."""
        values = np.asarray(values)
        field_view = self.frame(time_index, channel)
        field_view.update(offset, values)
        channel_index = self.channel_index(channel)
        self.revision += 1
        update = (self.revision, int(time_index), channel_index,
                  tuple(int(value) for value in offset), tuple(values.shape))
        self._updates.append(update)
        return self.revision

    def updates_since(self, revision):
        revision = int(revision)
        if revision < 0 or revision > self.revision:
            raise ValueError("revision is outside this dataset's history")
        return tuple(update for update in self._updates if update[0] > revision)

    def snapshot(self):
        return {
            "kind": "scalar_field_series", "axis_order": "tczyx",
            "shape": list(self.data.shape), "dtype": str(self.data.dtype),
            "times": self.times.tolist(), "time_unit": self.time_unit,
            "channels": list(self.channels), "channel_units": list(self.channel_units),
            "spacing_xyz": list(self.spacing), "origin_xyz": list(self.origin),
            "direction": self.direction.tolist(), "name": self.name,
            "revision": self.revision, "metadata": dict(self.metadata),
        }
