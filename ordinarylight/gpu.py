"""Backend-neutral ownership contract for GPU-resident render products."""

from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from types import MappingProxyType


@dataclass(frozen=True)
class VulkanImageMetadata:
    """Description required to import an external Vulkan image allocation.

    File descriptors are intentionally not stored in this value.  They are
    created on demand by :class:`GpuFrame`, and ownership transfers to the
    caller when an export method succeeds.
    """

    width: int
    height: int
    format: str
    format_value: int
    layout: str
    memory_size: int
    memory_offset: int
    dedicated_allocation: bool
    device_uuid: str
    image_handle: int
    memory_handle: int
    device_handle: int
    physical_device_handle: int
    completion_fence_handle: int
    queue_family_index: int
    handle_type: str = "opaque_fd"


@dataclass(frozen=True)
class VulkanBufferMetadata:
    """Description of an externally importable pitch-linear video buffer."""

    width: int
    height: int
    format: str
    pitch: int
    y_offset: int
    uv_offset: int
    memory_size: int
    memory_offset: int
    dedicated_allocation: bool
    device_uuid: str
    buffer_handle: int
    memory_handle: int
    device_handle: int
    physical_device_handle: int
    completion_fence_handle: int
    queue_family_index: int
    color_matrix: str = "bt709"
    color_range: str = "limited"
    handle_type: str = "opaque_fd"


class GpuFrame:
    """A rendered image that remains resident on a graphics device.

    The frame owns a backend slot until :meth:`close` is called. Exported file
    descriptors use transfer ownership: after importing one into CUDA, the
    CUDA import owns that descriptor. If an import fails, the caller must close
    the descriptor with :func:`os.close`.

    A frame must not be closed while another API is still accessing its image.
    Synchronize the consuming CUDA stream (or signal a future release
    semaphore) before releasing it.
    """

    def __init__(
        self, *, api, metadata, export_memory_fd, export_ready_semaphore_fd,
        wait, close, export_release_semaphore_fd=None, attributes=None,
    ):
        self.api = str(api)
        self.metadata = metadata
        self.attributes = MappingProxyType(dict(attributes or {}))
        self._export_memory_fd = export_memory_fd
        self._export_ready_semaphore_fd = export_ready_semaphore_fd
        self._export_release_semaphore_fd = export_release_semaphore_fd
        self._wait = wait
        self._close = close
        self._memory_exported = False
        self._semaphore_exported = False
        self._release_semaphore_exported = False
        self._external_release_scheduled = False
        self._closed = False
        self._lock = Lock()

    @property
    def closed(self):
        return self._closed

    def _export_once(self, kind, callback):
        with self._lock:
            if self._closed:
                raise RuntimeError("GPU frame is closed")
            attribute = f"_{kind}_exported"
            if getattr(self, attribute):
                raise RuntimeError(f"{kind.replace('_', ' ')} was already exported")
            descriptor = int(callback())
            if descriptor < 0:
                raise RuntimeError(f"backend returned an invalid {kind} descriptor")
            setattr(self, attribute, True)
            return descriptor

    def export_memory_fd(self):
        """Return an opaque-memory FD whose ownership transfers to the caller."""
        return self._export_once("memory", self._export_memory_fd)

    def export_ready_semaphore_fd(self):
        """Return an opaque binary-semaphore FD, transferring ownership."""
        return self._export_once(
            "semaphore", self._export_ready_semaphore_fd
        )

    def export_release_semaphore_fd(self):
        """Export the semaphore the consumer signals after using this frame.

        A consumer that exports this semaphore may call :meth:`close`
        immediately after queueing its GPU signal. Vulkan waits for that signal
        before reusing the frame slot, without synchronizing either API on the
        CPU.
        """
        if self._export_release_semaphore_fd is None:
            raise RuntimeError("this GPU frame has no release semaphore")
        return self._export_once(
            "release_semaphore", self._export_release_semaphore_fd
        )

    def mark_external_release_scheduled(self):
        """Record that a cached interop semaphore will release this frame.

        Most consumers export synchronization handles once and cache their
        imported API objects for every subsequent use of the same frame slot.
        Such consumers call this after queueing the cached release signal.
        """
        with self._lock:
            if self._closed:
                raise RuntimeError("GPU frame is closed")
            self._external_release_scheduled = True

    def wait(self, timeout=None):
        """Wait for the Vulkan render submission to finish.

        This is intended for diagnostics and non-CUDA consumers. A zero-copy
        CUDA path should import and wait on the exported semaphore instead.
        """
        with self._lock:
            if self._closed:
                raise RuntimeError("GPU frame is closed")
        return bool(self._wait(timeout))

    def close(self):
        """Release the backend frame slot after all consumers have finished."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._close(self._external_release_scheduled)

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()


__all__ = ["GpuFrame", "VulkanBufferMetadata", "VulkanImageMetadata"]
