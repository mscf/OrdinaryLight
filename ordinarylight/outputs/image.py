"""Backend-neutral image output conversion."""

import numpy as np


def to_sdr(image, *, exposure=1.0, tone_mapping="aces", alpha=False):
    """Convert a linear HDR RGB(A) array to an sRGB ``uint8`` image.

    ``tone_mapping`` may be ``"aces"``, ``"reinhard"``, or ``"clip"``.
    Existing uint8 input is copied without applying another transfer curve.
    """
    source = np.asarray(image)
    if source.ndim != 3 or source.shape[2] not in (3, 4):
        raise ValueError("image must have shape (height, width, 3 or 4)")
    if source.dtype == np.uint8:
        result = np.array(source, copy=True, order="C")
    else:
        if not np.issubdtype(source.dtype, np.floating):
            raise TypeError("image must use a floating or uint8 dtype")
        if not np.isfinite(exposure) or exposure < 0.0:
            raise ValueError("exposure must be finite and non-negative")
        rgb = np.maximum(source[..., :3].astype(np.float32), 0.0) * exposure
        if tone_mapping == "aces":
            rgb = np.clip(
                rgb * (2.51 * rgb + 0.03)
                / np.maximum(rgb * (2.43 * rgb + 0.59) + 0.14, 1e-8),
                0.0, 1.0,
            )
        elif tone_mapping == "reinhard":
            rgb = rgb / (1.0 + rgb)
        elif tone_mapping == "clip":
            rgb = np.clip(rgb, 0.0, 1.0)
        else:
            raise ValueError("tone_mapping must be aces, reinhard, or clip")
        low = rgb <= 0.0031308
        rgb = np.where(low, rgb * 12.92, 1.055 * rgb ** (1.0 / 2.4) - 0.055)
        channels = [np.clip(rgb * 255.0 + 0.5, 0, 255).astype(np.uint8)]
        if source.shape[2] == 4:
            channels.append(np.clip(
                source[..., 3:4] * 255.0 + 0.5, 0, 255
            ).astype(np.uint8))
        result = np.concatenate(channels, axis=2) if len(channels) == 2 else channels[0]
    if alpha and result.shape[2] == 3:
        result = np.concatenate((
            result, np.full((*result.shape[:2], 1), 255, np.uint8)
        ), axis=2)
    elif not alpha and result.shape[2] == 4:
        result = result[..., :3].copy()
    return np.ascontiguousarray(result)
