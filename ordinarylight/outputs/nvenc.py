"""CUDA/Vulkan zero-copy NVENC output.

This module is optional. Importing :mod:`ordinarylight` never imports CUDA or
PyNvVideoCodec; those dependencies are loaded only when this writer is built.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

from ..gpu import GpuFrame, VulkanBufferMetadata


def _cuda_modules():
    try:
        from cuda.bindings import runtime as cudart
        import PyNvVideoCodec as nvc
    except ImportError as error:
        raise RuntimeError(
            "GPU video output requires ordinarylight[video-gpu] "
            "(cuda-bindings and PyNvVideoCodec)"
        ) from error
    return cudart, nvc


def _check(result, operation):
    values = result if isinstance(result, tuple) else (result,)
    status = values[0]
    if int(status) != 0:
        raise RuntimeError(f"{operation} failed with CUDA status {status}")
    if len(values) == 1:
        return None
    if len(values) == 2:
        return values[1]
    return values[1:]


def _device_pointer(value):
    try:
        return int(value)
    except TypeError:
        return int(value.value)


@dataclass
class _CudaSlot:
    identity: tuple
    external_memory: object
    pointer: object
    ready_semaphore: object
    release_semaphore: object

    def close(self, cudart):
        _check(cudart.cudaFree(self.pointer), "cudaFree(external buffer)")
        _check(
            cudart.cudaDestroyExternalSemaphore(self.ready_semaphore),
            "cudaDestroyExternalSemaphore(ready)",
        )
        _check(
            cudart.cudaDestroyExternalSemaphore(self.release_semaphore),
            "cudaDestroyExternalSemaphore(release)",
        )
        _check(
            cudart.cudaDestroyExternalMemory(self.external_memory),
            "cudaDestroyExternalMemory",
        )


class _CudaArrayPlane:
    def __init__(self, interface):
        self.__cuda_array_interface__ = interface


class _Nv12CudaFrame:
    """PyNvVideoCodec input object backed by imported Vulkan memory."""

    def __init__(self, pointer, metadata):
        self._pointer = _device_pointer(pointer)
        self._metadata = metadata
        self.frameSize = int(metadata.width * metadata.height * 3 // 2)

    def cuda(self):
        metadata = self._metadata
        return [
            _CudaArrayPlane({
                "shape": (metadata.height, metadata.width, 1),
                "strides": (metadata.pitch, 1, 1),
                "typestr": "|u1",
                "data": (self._pointer + metadata.y_offset, False),
                "version": 3,
            }),
            _CudaArrayPlane({
                "shape": (metadata.height // 2, metadata.width // 2, 2),
                "strides": (metadata.pitch, 2, 1),
                "typestr": "|u1",
                "data": (self._pointer + metadata.uv_offset, False),
                "version": 3,
            }),
        ]


class NvencVideoWriter:
    """Write GPU-resident Ordinary Light frames as H.264 elementary stream.

    Rendering, tone mapping, NV12 conversion, and encoding remain on the GPU.
    Vulkan and CUDA coordinate through external binary semaphores; no NumPy
    array, host staging copy, or CPU color conversion occurs per frame.
    """

    def __init__(
        self, path, size, *, fps=30, bitrate=None, preset="P4", device=0,
        codec="h264",
    ):
        self._cudart, self._nvc = _cuda_modules()
        self.width, self.height = map(int, size)
        if self.width <= 0 or self.height <= 0:
            raise ValueError("video size must be positive")
        if self.width % 4 or self.height % 2:
            raise ValueError(
                "NV12 video width must be divisible by 4 and height must be even"
            )
        self.destination = path
        self._owns_file = not hasattr(path, "write")
        self._file = Path(path).open("wb") if self._owns_file else path
        self._slots = {}
        self._closed = False
        self.device = int(device)
        self.frame_count = 0
        self.byte_count = 0
        _check(self._cudart.cudaSetDevice(self.device), "cudaSetDevice")
        properties = _check(
            self._cudart.cudaGetDeviceProperties(self.device),
            "cudaGetDeviceProperties",
        )
        self._device_uuid = bytes(properties.uuid.bytes).hex()
        self._stream = _check(
            self._cudart.cudaStreamCreate(), "cudaStreamCreate"
        )
        options = {
            "codec": str(codec), "fps": str(fps), "preset": str(preset),
        }
        if bitrate is not None:
            options["bitrate"] = str(bitrate)
        self._encoder = self._nvc.CreateEncoder(
            self.width, self.height, "NV12", False,
            cudastream=_device_pointer(self._stream), **options,
        )

    def _import_semaphore(self, descriptor):
        desc = self._cudart.cudaExternalSemaphoreHandleDesc()
        desc.type = (
            self._cudart.cudaExternalSemaphoreHandleType
            .cudaExternalSemaphoreHandleTypeOpaqueFd
        )
        desc.handle.fd = int(descriptor)
        try:
            return _check(
                self._cudart.cudaImportExternalSemaphore(desc),
                "cudaImportExternalSemaphore",
            )
        except Exception:
            os.close(descriptor)
            raise

    def _import_slot(self, frame, metadata, identity):
        memory_fd = frame.export_memory_fd()
        memory_desc = self._cudart.cudaExternalMemoryHandleDesc()
        memory_desc.type = (
            self._cudart.cudaExternalMemoryHandleType
            .cudaExternalMemoryHandleTypeOpaqueFd
        )
        memory_desc.handle.fd = int(memory_fd)
        memory_desc.size = int(metadata.memory_size)
        memory_desc.flags = int(self._cudart.cudaExternalMemoryDedicated)
        try:
            external_memory = _check(
                self._cudart.cudaImportExternalMemory(memory_desc),
                "cudaImportExternalMemory",
            )
        except Exception:
            os.close(memory_fd)
            raise
        buffer_desc = self._cudart.cudaExternalMemoryBufferDesc()
        buffer_desc.offset = int(metadata.memory_offset)
        buffer_desc.size = int(metadata.memory_size)
        buffer_desc.flags = 0
        pointer = None
        ready_semaphore = None
        release_semaphore = None
        try:
            pointer = _check(
                self._cudart.cudaExternalMemoryGetMappedBuffer(
                    external_memory, buffer_desc
                ),
                "cudaExternalMemoryGetMappedBuffer",
            )
            ready_semaphore = self._import_semaphore(
                frame.export_ready_semaphore_fd()
            )
            release_semaphore = self._import_semaphore(
                frame.export_release_semaphore_fd()
            )
            return _CudaSlot(
                identity=identity,
                external_memory=external_memory,
                pointer=pointer,
                ready_semaphore=ready_semaphore,
                release_semaphore=release_semaphore,
            )
        except Exception:
            if pointer is not None:
                self._cudart.cudaFree(pointer)
            if ready_semaphore is not None:
                self._cudart.cudaDestroyExternalSemaphore(ready_semaphore)
            if release_semaphore is not None:
                self._cudart.cudaDestroyExternalSemaphore(release_semaphore)
            self._cudart.cudaDestroyExternalMemory(external_memory)
            raise

    def _slot(self, frame, metadata):
        index = int(frame.attributes["frame_slot"])
        identity = (
            metadata.device_uuid, metadata.buffer_handle,
            metadata.memory_handle, metadata.memory_size,
        )
        slot = self._slots.get(index)
        if slot is not None and slot.identity != identity:
            _check(
                self._cudart.cudaStreamSynchronize(self._stream),
                "cudaStreamSynchronize",
            )
            slot.close(self._cudart)
            slot = None
        if slot is None:
            slot = self._import_slot(frame, metadata, identity)
            self._slots[index] = slot
        return slot

    def write(self, frame: GpuFrame):
        """Encode one ``pixel_format='nv12'`` GPU frame and return byte count."""
        if self._closed:
            raise RuntimeError("video writer is closed")
        if not isinstance(frame, GpuFrame):
            raise TypeError("frame must be an ordinarylight.GpuFrame")
        metadata = frame.metadata
        if not isinstance(metadata, VulkanBufferMetadata) or metadata.format != "NV12":
            raise ValueError("NvencVideoWriter requires Vulkan NV12 output")
        if (metadata.width, metadata.height) != (self.width, self.height):
            raise ValueError("GPU frame dimensions do not match the encoder")
        if metadata.device_uuid.lower() != self._device_uuid:
            raise RuntimeError(
                "Vulkan output and CUDA encoder use different physical GPUs"
            )
        try:
            slot = self._slot(frame, metadata)
        except Exception:
            frame.close()
            raise
        wait_params = self._cudart.cudaExternalSemaphoreWaitParams()
        _check(
            self._cudart.cudaWaitExternalSemaphoresAsync(
                [slot.ready_semaphore], [wait_params], 1, self._stream,
            ),
            "cudaWaitExternalSemaphoresAsync",
        )
        try:
            packets = self._encoder.Encode(
                _Nv12CudaFrame(slot.pointer, metadata)
            )
        finally:
            signal_params = self._cudart.cudaExternalSemaphoreSignalParams()
            _check(
                self._cudart.cudaSignalExternalSemaphoresAsync(
                    [slot.release_semaphore], [signal_params], 1, self._stream,
                ),
                "cudaSignalExternalSemaphoresAsync",
            )
            frame.mark_external_release_scheduled()
            frame.close()
        byte_count = 0
        for packet in packets:
            data = packet["data"] if isinstance(packet, dict) else packet
            self._file.write(data)
            byte_count += len(data)
        self.frame_count += 1
        self.byte_count += byte_count
        return byte_count

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            for packet in self._encoder.EndEncode():
                data = packet["data"] if isinstance(packet, dict) else packet
                self._file.write(data)
                self.byte_count += len(data)
            _check(
                self._cudart.cudaStreamSynchronize(self._stream),
                "cudaStreamSynchronize",
            )
            for slot in self._slots.values():
                slot.close(self._cudart)
            self._slots.clear()
            _check(
                self._cudart.cudaStreamDestroy(self._stream),
                "cudaStreamDestroy",
            )
        finally:
            if self._owns_file:
                self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()


__all__ = ["NvencVideoWriter"]
