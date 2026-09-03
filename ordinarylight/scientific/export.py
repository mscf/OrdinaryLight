"""Reproducible scientific image export and verification."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path
import tempfile

import numpy as np
from PIL import Image

from ..outputs import to_sdr
from .clipping import ClipRegion
from .scalar_field import ScalarField3D
from .transfer import TransferFunction


SCIENTIFIC_EXPORT_SCHEMA = "ordinarylight.scientific-image/1"


def _canonical_value(value):
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, (float, np.floating)):
        value = float(value)
        if np.isfinite(value):
            return value
        return {"__float__": "NaN" if np.isnan(value) else ("Infinity" if value > 0 else "-Infinity")}
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.ndarray):
        return _canonical_value(value.tolist())
    if is_dataclass(value):
        return _canonical_value(asdict(value))
    if hasattr(value, "items"):
        return {str(key): _canonical_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_canonical_value(item) for item in value]
    raise TypeError(f"cannot serialize reproducibility value of type {type(value).__name__}")


def _component_snapshot(value):
    if value is None:
        return None
    if hasattr(value, "snapshot") and callable(value.snapshot):
        payload = value.snapshot()
    elif is_dataclass(value):
        payload = asdict(value)
    elif hasattr(value, "items"):
        payload = dict(value)
    elif hasattr(value, "__dict__"):
        payload = {
            key: item for key, item in vars(value).items()
            if not key.startswith("_")
        }
    else:
        raise TypeError(f"cannot snapshot component of type {type(value).__name__}")
    return {"type": f"{type(value).__module__}.{type(value).__qualname__}",
            "state": _canonical_value(payload)}


def scalar_field_sha256(field):
    """Hash field layout, coordinates, units, and exact array bytes."""
    if not isinstance(field, ScalarField3D):
        raise TypeError("field must be a ScalarField3D")
    snapshot = field.snapshot()
    identity = {
        key: snapshot[key] for key in (
            "kind", "axis_order", "shape", "dtype", "spacing_xyz",
            "origin_xyz", "direction", "unit",
        )
    }
    digest = sha256(json.dumps(
        _canonical_value(identity), sort_keys=True, separators=(",", ":"),
    ).encode("utf-8"))
    array = np.ascontiguousarray(field.data)
    byte_view = memoryview(array).cast("B")
    chunk_size = 8 * 1024 * 1024
    for start in range(0, len(byte_view), chunk_size):
        digest.update(byte_view[start:start + chunk_size])
    return digest.hexdigest()


def _png_bytes(pixels):
    buffer = BytesIO()
    mode = "RGBA" if pixels.shape[2] == 4 else "RGB"
    Image.fromarray(pixels, mode=mode).save(
        buffer, format="PNG", optimize=False, compress_level=9,
    )
    return buffer.getvalue()


def _atomic_write(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent,
    )
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def export_scientific_image(
    path, image, *, field, scene, camera, transfer_function,
    clipping=None, renderer=None, seed=None, exposure=1.0,
    tone_mapping="clip", alpha=False, extra_metadata=None, overwrite=False,
):
    """Write a PNG and canonical ``.json`` reproducibility sidecar.

    The returned mapping is exactly the document written beside the image.
    Existing targets are protected unless ``overwrite=True``.
    """
    path = Path(path)
    if path.suffix.lower() != ".png":
        raise ValueError("scientific image export currently requires a .png path")
    sidecar = path.with_suffix(path.suffix + ".json")
    if not overwrite and (path.exists() or sidecar.exists()):
        raise FileExistsError("image or metadata sidecar already exists")
    if not isinstance(field, ScalarField3D):
        raise TypeError("field must be a ScalarField3D")
    if not isinstance(transfer_function, TransferFunction):
        raise TypeError("transfer_function must be a TransferFunction")
    if clipping is not None and not isinstance(clipping, ClipRegion):
        raise TypeError("clipping must be a ClipRegion or None")
    exposure = float(exposure)
    if not np.isfinite(exposure) or exposure < 0.0:
        raise ValueError("exposure must be finite and non-negative")
    if tone_mapping not in {"aces", "reinhard", "clip"}:
        raise ValueError("tone_mapping must be aces, reinhard, or clip")
    if not isinstance(alpha, bool):
        raise TypeError("alpha must be a bool")
    if seed is not None and (isinstance(seed, bool) or int(seed) != seed):
        raise ValueError("seed must be an integer or None")
    pixels = to_sdr(
        image, exposure=exposure, tone_mapping=tone_mapping, alpha=alpha,
    )
    encoded = _png_bytes(pixels)
    document = {
        "schema": SCIENTIFIC_EXPORT_SCHEMA,
        "image": {
            "filename": path.name, "format": "PNG", "mode": "RGBA" if alpha else "RGB",
            "width": int(pixels.shape[1]), "height": int(pixels.shape[0]),
            "sha256": sha256(encoded).hexdigest(),
            "pixel_sha256": sha256(memoryview(pixels).cast("B")).hexdigest(),
        },
        "field": {
            "sha256": scalar_field_sha256(field),
            "description": _canonical_value(field.snapshot()),
        },
        "camera": _component_snapshot(camera),
        "scene": _component_snapshot(scene),
        "transfer_function": _canonical_value(transfer_function.snapshot(field.data)),
        "clipping": None if clipping is None else _canonical_value(clipping.snapshot()),
        "renderer": _component_snapshot(renderer),
        "random_seed": None if seed is None else int(seed),
        "color_management": {
            "input": "linear", "output": "sRGB", "exposure": float(exposure),
            "tone_mapping": tone_mapping,
        },
        "extra": _canonical_value({} if extra_metadata is None else extra_metadata),
    }
    metadata_bytes = (
        json.dumps(document, sort_keys=True, indent=2, allow_nan=False)
        + "\n"
    ).encode("utf-8")
    _atomic_write(path, encoded)
    _atomic_write(sidecar, metadata_bytes)
    return document


def verify_scientific_export(path, *, field=None):
    """Verify encoded-image, decoded-pixel, and optional source-field hashes."""
    path = Path(path)
    sidecar = path.with_suffix(path.suffix + ".json")
    document = json.loads(sidecar.read_text(encoding="utf-8"))
    if document.get("schema") != SCIENTIFIC_EXPORT_SCHEMA:
        raise ValueError("unsupported scientific export schema")
    encoded = path.read_bytes()
    if sha256(encoded).hexdigest() != document["image"]["sha256"]:
        raise ValueError("encoded image checksum does not match metadata")
    pixels = np.asarray(Image.open(BytesIO(encoded)))
    if sha256(memoryview(np.ascontiguousarray(pixels)).cast("B")).hexdigest() != document["image"]["pixel_sha256"]:
        raise ValueError("decoded pixel checksum does not match metadata")
    if field is not None and scalar_field_sha256(field) != document["field"]["sha256"]:
        raise ValueError("source field checksum does not match metadata")
    return document
