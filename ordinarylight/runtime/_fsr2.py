"""Low-level FSR2 context; callers own synchronization and external images."""

import ctypes as ct
import math
import os
from pathlib import Path
from operator import index
import vulkan as vk


def handle(value):
    return int(vk.ffi.cast("uintptr_t", value))


def library():
    path = Path(
        os.environ.get(
            "ORDINARYLIGHT_FSR2_LIBRARY",
            Path(__file__).resolve().parents[2]
            / ".tools/fsr2/libordinarylight_fsr2.so",
        )
    )
    if not path.is_file():
        raise RuntimeError(
            "FSR 2 requires the optional native bridge; run .venv/bin/python scripts/build_fsr2.py or set ORDINARYLIGHT_FSR2_LIBRARY"
        )
    lib = ct.CDLL(str(path))
    if lib.ol_fsr2_version() != 1:
        raise RuntimeError(
            "Incompatible FSR 2 bridge ABI; rebuild scripts/build_fsr2.py"
        )
    lib.ol_fsr2_create.argtypes = [
        ct.c_void_p,
        ct.c_void_p,
        ct.c_uint,
        ct.c_uint,
        ct.POINTER(ct.c_int),
    ]
    lib.ol_fsr2_create.restype = ct.c_void_p
    lib.ol_fsr2_destroy.argtypes = [ct.c_void_p]
    lib.ol_fsr2_jitter.argtypes = [
        ct.c_int,
        ct.c_int,
        ct.c_int,
        ct.POINTER(ct.c_float),
        ct.POINTER(ct.c_float),
    ]
    lib.ol_fsr2_dispatch.argtypes = [
        ct.c_void_p,
        ct.c_void_p,
        ct.POINTER(ct.c_uint64),
        ct.POINTER(ct.c_uint64),
        ct.c_uint,
        ct.c_uint,
        ct.c_float,
        ct.c_float,
        ct.c_float,
        ct.c_float,
        ct.c_int,
    ]
    return lib


class Fsr2Context:
    """One serial temporal stream on an application-owned Vulkan device.

    Recordings must be submitted in order; command replay is unsupported. The
    caller keeps images alive and waits for GPU completion before close(). An
    abandoned recording forces the next dispatch to reset SDK history.
    """

    def __init__(self, physical_device, device, extent, output):
        self.extent = self._extent(extent)
        self.output = self._extent(output)
        if any(a > b for a, b in zip(self.extent, self.output)):
            raise ValueError("FSR2 render extent exceeds output extent")
        self.lib = library()
        error = ct.c_int()
        self.context = self.lib.ol_fsr2_create(
            handle(physical_device), handle(device), *self.output, ct.byref(error)
        )
        if not self.context:
            raise RuntimeError(f"FSR 2 context creation failed: {error.value}")
        self.history_valid = False
        self.pending = None

    @staticmethod
    def _extent(value):
        value = tuple(index(v) for v in value)
        if len(value) != 2 or any(v <= 0 or v > 0x7FFFFFFF for v in value):
            raise ValueError("FSR2 requires two positive extent dimensions")
        return value

    def require_open(self):
        if not self.context:
            raise RuntimeError("FSR2 context is closed")

    def jitter(self, sequence):
        self.require_open()
        sequence = index(sequence)
        if not 0 <= sequence <= 0x7FFFFFFF:
            raise ValueError("FSR2 sequence must fit a nonnegative signed integer")
        x, y = ct.c_float(), ct.c_float()
        self.lib.ol_fsr2_jitter(
            sequence, self.extent[0], self.output[0], ct.byref(x), ct.byref(y)
        )
        # Match the native camera's packed half coordinates.
        import numpy as np

        return tuple(
            float(v) - 0.5
            for v in np.asarray([x.value + 0.5, y.value + 0.5], dtype=np.float16)
        )

    def record(self, command, images, views, *, jitter, fov_y, dt_ms, reset=False):
        self.require_open()
        images, views, jitter = tuple(images), tuple(views), tuple(jitter)
        if len(images) != 5 or len(views) != 5:
            raise ValueError("FSR2 requires HDR, depth, motion, reactive and output")
        if (
            len(jitter) != 2
            or not all(math.isfinite(v) for v in jitter)
            or not math.isfinite(fov_y)
            or not 0 < fov_y < math.pi
            or not math.isfinite(dt_ms)
            or not 0 < dt_ms <= 1000
        ):
            raise ValueError("Invalid FSR2 jitter, field of view or frame time")
        reset = reset or not self.history_valid or self.pending is not None
        # Dispatch mutates SDK state even before the command reaches the queue.
        # Invalidate before calling it so errors/abandoned commands force reset.
        self.history_valid = False
        self.pending = None
        error = self.lib.ol_fsr2_dispatch(
            self.context,
            handle(command),
            (ct.c_uint64 * 5)(*[handle(i) for i in images]),
            (ct.c_uint64 * 5)(*[handle(v) for v in views]),
            *self.extent,
            *jitter,
            fov_y,
            dt_ms,
            int(reset),
        )
        if error:
            raise RuntimeError(f"FSR 2 dispatch failed: {error}")
        self.pending = object()
        return self.pending

    def submitted(self, token):
        self.require_open()
        if token is None or token is not self.pending:
            raise RuntimeError("FSR2 submission token is stale or already published")
        self.history_valid = True
        self.pending = None

    def close(self):
        if self.context:
            self.lib.ol_fsr2_destroy(self.context)
            self.context = None
        self.history_valid = False
        self.pending = None
