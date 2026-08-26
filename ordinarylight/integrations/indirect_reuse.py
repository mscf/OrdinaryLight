"""Reference contracts for composable indirect-light sample reuse.

Indirect reuse cannot share the direct-light reservoir ABI: reconnecting a
secondary vertex requires its position, normal, incident radiance, proposal
density, and source target.  This module fixes the CPU oracle and memory plan
before the corresponding Vulkan storage and shaders are enabled.
"""

from dataclasses import dataclass
import math
import struct


@dataclass(frozen=True)
class IndirectReservoirPlan:
    """Validated storage plan for double-buffered indirect reservoirs.

    The 24-byte compact ABI reserves eight bytes for camera-relative FP16
    position plus proposal density, four for an octahedral normal, four for
    packed incident radiance, four for half weight/target, and four for the
    validity/sample-count header.
    """

    output_width: int
    output_height: int
    scale: float = 0.5
    bytes_per_reservoir: int = 24
    bytes_per_seed: int = 4
    history_frames: int = 2
    budget_mib: float = 128.0

    def __post_init__(self):
        if self.output_width < 1 or self.output_height < 1:
            raise ValueError("output dimensions must be positive")
        if not 0.25 <= self.scale <= 1.0:
            raise ValueError("indirect reservoir scale must be in [0.25, 1.0]")
        if self.bytes_per_reservoir < 24:
            raise ValueError(
                "indirect reservoirs require at least 24 bytes per pixel"
            )
        if self.bytes_per_seed < 4:
            raise ValueError("indirect seeds require at least 4 bytes per pixel")
        if self.history_frames not in (1, 2):
            raise ValueError("history_frames must be 1 or 2")
        if not math.isfinite(self.budget_mib) or self.budget_mib <= 0.0:
            raise ValueError("budget_mib must be finite and positive")
        if self.estimated_mib > self.budget_mib:
            raise MemoryError(
                f"indirect reservoir plan requires {self.estimated_mib:.1f} "
                f"MiB, exceeding its {self.budget_mib:.1f} MiB budget"
            )

    @property
    def width(self) -> int:
        return max(1, math.ceil(self.output_width * self.scale))

    @property
    def height(self) -> int:
        return max(1, math.ceil(self.output_height * self.scale))

    @property
    def reservoir_count(self) -> int:
        return self.width * self.height

    @property
    def estimated_bytes(self) -> int:
        return (
            self.reservoir_count
            * (self.bytes_per_reservoir + self.bytes_per_seed)
            * self.history_frames
        )

    @property
    def estimated_mib(self) -> float:
        return self.estimated_bytes / (1024.0 * 1024.0)


@dataclass(frozen=True)
class IndirectLightSample:
    """A reconnectable secondary-vertex candidate."""

    target: float
    proposal_pdf: float
    radiance: tuple[float, float, float]
    secondary_position: tuple[float, float, float]
    secondary_normal: tuple[float, float, float]

    def __post_init__(self):
        if not math.isfinite(self.target) or self.target < 0.0:
            raise ValueError("target must be finite and non-negative")
        if not math.isfinite(self.proposal_pdf) or self.proposal_pdf <= 0.0:
            raise ValueError("proposal_pdf must be finite and positive")
        if len(self.radiance) != 3 or any(
            not math.isfinite(value) or value < 0.0 for value in self.radiance
        ):
            raise ValueError("radiance must contain three finite non-negative values")
        if len(self.secondary_position) != 3 or any(
            not math.isfinite(value) for value in self.secondary_position
        ):
            raise ValueError("secondary_position must contain three finite values")
        if len(self.secondary_normal) != 3 or any(
            not math.isfinite(value) for value in self.secondary_normal
        ) or sum(value * value for value in self.secondary_normal) <= 0.0:
            raise ValueError("secondary_normal must be a finite non-zero vector")


@dataclass
class IndirectLightReservoir:
    """Weighted reservoir oracle with reconnection-Jacobian reuse."""

    sample: IndirectLightSample | None = None
    weight_sum: float = 0.0
    sample_count: int = 0

    def update(
        self,
        sample: IndirectLightSample,
        weight: float,
        random_value: float,
        *,
        represented_samples: int = 1,
    ) -> bool:
        if not math.isfinite(weight) or weight < 0.0:
            raise ValueError("weight must be finite and non-negative")
        if not 0.0 <= random_value < 1.0:
            raise ValueError("random_value must be in [0, 1)")
        if represented_samples < 1:
            raise ValueError("represented_samples must be positive")
        self.weight_sum += weight
        self.sample_count += represented_samples
        selected = weight > 0.0 and (
            self.sample is None or random_value * self.weight_sum < weight
        )
        if selected:
            self.sample = sample
        return selected

    def merge_reconnected(
        self,
        source: "IndirectLightReservoir",
        target_at_current_surface: float,
        reconnection_jacobian: float,
        random_value: float,
    ) -> bool:
        """Merge a source after reevaluation and geometric reconnection."""
        for name, value in (
            ("target_at_current_surface", target_at_current_surface),
            ("reconnection_jacobian", reconnection_jacobian),
        ):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if source.sample is None or source.sample_count == 0:
            return False
        source_target = source.sample.target
        reuse_weight = 0.0
        if source_target > 0.0:
            reuse_weight = (
                target_at_current_surface
                * source.weight_sum
                * reconnection_jacobian
                / source_target
            )
        reused = IndirectLightSample(
            target_at_current_surface,
            source.sample.proposal_pdf,
            source.sample.radiance,
            source.sample.secondary_position,
            source.sample.secondary_normal,
        )
        return self.update(
            reused,
            reuse_weight,
            random_value,
            represented_samples=source.sample_count,
        )

    def limit_history(self, max_samples: int) -> bool:
        """Bound represented history while preserving reservoir normalization."""
        if not 1 <= max_samples <= 127:
            raise ValueError("max_samples must be between 1 and 127")
        if self.sample_count <= max_samples:
            return False
        scale = max_samples / self.sample_count
        self.weight_sum *= scale
        self.sample_count = max_samples
        return True

    @property
    def normalization(self) -> float:
        if self.sample is None or self.sample_count == 0 or self.sample.target <= 0.0:
            return 0.0
        return self.weight_sum / (self.sample_count * self.sample.target)


@dataclass(frozen=True)
class PackedIndirectReservoir:
    """Decoded representation of one 24-byte reservoir record."""

    sample: IndirectLightSample | None
    weight_sum: float
    sample_count: int


_PACKED_RESERVOIR = struct.Struct("<4eII2eI")
assert _PACKED_RESERVOIR.size == 24


def _encode_octahedral(normal):
    length = math.sqrt(sum(value * value for value in normal))
    x, y, z = (value / length for value in normal)
    inverse_l1 = 1.0 / (abs(x) + abs(y) + abs(z))
    x, y, z = x * inverse_l1, y * inverse_l1, z * inverse_l1
    if z < 0.0:
        old_x = x
        x = math.copysign(1.0 - abs(y), old_x)
        y = math.copysign(1.0 - abs(old_x), y)
    encoded_x = round((x * 0.5 + 0.5) * 65535.0)
    encoded_y = round((y * 0.5 + 0.5) * 65535.0)
    return max(0, min(encoded_x, 65535)) | (
        max(0, min(encoded_y, 65535)) << 16
    )


def _decode_octahedral(packed):
    x = ((packed & 0xFFFF) / 65535.0) * 2.0 - 1.0
    y = (((packed >> 16) & 0xFFFF) / 65535.0) * 2.0 - 1.0
    z = 1.0 - abs(x) - abs(y)
    if z < 0.0:
        old_x = x
        x = math.copysign(1.0 - abs(y), old_x)
        y = math.copysign(1.0 - abs(old_x), y)
    length = math.sqrt(x * x + y * y + z * z)
    return x / length, y / length, z / length


def _pack_rgb9e5(color):
    red, green, blue = (
        min(max(float(value), 0.0), 65408.0) for value in color
    )
    maximum = max(red, green, blue)
    if maximum < 2.0 ** -16:
        return 0
    exponent = max(-16, math.floor(math.log2(maximum))) + 1
    exponent = min(exponent, 16)
    scale = 2.0 ** (exponent - 9)
    mantissas = [round(value / scale) for value in (red, green, blue)]
    if max(mantissas) > 511 and exponent < 16:
        exponent += 1
        scale *= 2.0
        mantissas = [round(value / scale) for value in (red, green, blue)]
    mantissas = [max(0, min(value, 511)) for value in mantissas]
    return (
        mantissas[0]
        | (mantissas[1] << 9)
        | (mantissas[2] << 18)
        | ((exponent + 15) << 27)
    )


def _unpack_rgb9e5(packed):
    exponent = ((packed >> 27) & 0x1F) - 15
    scale = 2.0 ** (exponent - 9)
    return (
        (packed & 0x1FF) * scale,
        ((packed >> 9) & 0x1FF) * scale,
        ((packed >> 18) & 0x1FF) * scale,
    )


def pack_indirect_reservoir(
    reservoir: IndirectLightReservoir,
    camera_origin=(0.0, 0.0, 0.0),
) -> bytes:
    """Pack one reservoir using the future Vulkan 24-byte ABI."""
    if reservoir.sample is None or reservoir.sample_count == 0:
        return bytes(_PACKED_RESERVOIR.size)
    sample = reservoir.sample
    relative_position = tuple(
        sample.secondary_position[index] - camera_origin[index]
        for index in range(3)
    )
    header = 0x80000000 | min(reservoir.sample_count, 127)
    try:
        return _PACKED_RESERVOIR.pack(
            *relative_position,
            min(max(sample.proposal_pdf, 2.0 ** -24), 65504.0),
            _encode_octahedral(sample.secondary_normal),
            _pack_rgb9e5(sample.radiance),
            min(max(reservoir.weight_sum, 0.0), 65504.0),
            min(max(sample.target, 0.0), 65504.0),
            header,
        )
    except (OverflowError, struct.error) as error:
        raise ValueError(
            "secondary position exceeds the compact FP16 camera-relative range"
        ) from error


def unpack_indirect_reservoir(
    packed: bytes,
    camera_origin=(0.0, 0.0, 0.0),
) -> PackedIndirectReservoir:
    """Decode one compact reservoir for ABI and error-bound tests."""
    if len(packed) != _PACKED_RESERVOIR.size:
        raise ValueError("packed indirect reservoir must contain exactly 24 bytes")
    x, y, z, proposal_pdf, normal, radiance, weight, target, header = (
        _PACKED_RESERVOIR.unpack(packed)
    )
    sample_count = header & 0x7F
    if (header & 0x80000000) == 0 or sample_count == 0:
        return PackedIndirectReservoir(None, 0.0, 0)
    position = tuple(
        value + camera_origin[index]
        for index, value in enumerate((x, y, z))
    )
    sample = IndirectLightSample(
        target,
        proposal_pdf,
        _unpack_rgb9e5(radiance),
        position,
        _decode_octahedral(normal),
    )
    return PackedIndirectReservoir(sample, weight, sample_count)
