"""Optional FFmpeg-backed compressed video output."""

from pathlib import Path
import shutil
import subprocess
import threading

import numpy as np


class FFmpegVideoWriter:
    """Incrementally encode RGB/RGBA uint8 frames without a GUI.

    ``destination`` may be a path or a binary file-like object. File-like
    outputs receive a fragmented Matroska stream as bytes while frames are
    written, which is suitable for notebook widgets and server transports.
    """

    def __init__(
        self, destination, size, *, fps=30.0, codec="libx264", quality=18,
        ffmpeg="ffmpeg", container=None,
    ):
        width, height = (int(value) for value in size)
        if width < 1 or height < 1:
            raise ValueError("size must contain positive width and height")
        if not np.isfinite(fps) or fps <= 0.0:
            raise ValueError("fps must be finite and positive")
        if not isinstance(codec, str) or not codec:
            raise ValueError("codec must be a non-empty string")
        if not 0 <= int(quality) <= 63:
            raise ValueError("quality must be between 0 and 63")
        if shutil.which(ffmpeg) is None:
            raise RuntimeError(f"FFmpeg executable not found: {ffmpeg!r}")
        self.destination = destination
        self.size = (width, height)
        self.fps = float(fps)
        self.codec = codec
        self.quality = int(quality)
        self.ffmpeg = ffmpeg
        self.container = container
        self._process = None
        self._reader = None
        self._reader_error = None
        self.frame_count = 0

    @staticmethod
    def available(ffmpeg="ffmpeg"):
        """Return whether the requested FFmpeg executable is discoverable."""
        return shutil.which(ffmpeg) is not None

    def _command(self, channels):
        width, height = self.size
        pixel_format = "rgba" if channels == 4 else "rgb24"
        command = [
            self.ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
            "-f", "rawvideo", "-pixel_format", pixel_format,
            "-video_size", f"{width}x{height}", "-framerate", str(self.fps),
            "-i", "pipe:0", "-an", "-c:v", self.codec,
            "-crf", str(self.quality), "-pix_fmt", "yuv420p",
        ]
        if hasattr(self.destination, "write"):
            command.extend(("-f", self.container or "matroska", "pipe:1"))
        else:
            if self.container:
                command.extend(("-f", self.container))
            command.append(str(Path(self.destination)))
        return command

    def _drain(self, process):
        try:
            while True:
                chunk = process.stdout.read(65536)
                if not chunk:
                    break
                self.destination.write(chunk)
        except BaseException as exc:  # propagated by close()
            self._reader_error = exc

    def write(self, frame):
        """Encode one contiguous RGB or RGBA uint8 frame."""
        array = np.asarray(frame)
        width, height = self.size
        if array.shape not in ((height, width, 3), (height, width, 4)):
            raise ValueError(
                f"frame must have shape ({height}, {width}, 3 or 4)"
            )
        if array.dtype != np.uint8:
            raise TypeError("video frames must be uint8; use outputs.to_sdr()")
        if self._process is None:
            streaming = hasattr(self.destination, "write")
            self._process = subprocess.Popen(
                self._command(array.shape[2]), stdin=subprocess.PIPE,
                stdout=subprocess.PIPE if streaming else subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
            if streaming:
                self._reader = threading.Thread(
                    target=self._drain, args=(self._process,), daemon=True
                )
                self._reader.start()
        try:
            self._process.stdin.write(np.ascontiguousarray(array).tobytes())
        except BrokenPipeError as exc:
            detail = self._process.stderr.read().decode("utf-8", "replace")
            raise RuntimeError(f"FFmpeg encoding failed: {detail.strip()}") from exc
        self.frame_count += 1
        return self

    def close(self):
        """Flush the stream and raise if encoding failed."""
        if self._process is None:
            return
        process, self._process = self._process, None
        process.stdin.close()
        if self._reader is not None:
            self._reader.join()
            self._reader = None
        return_code = process.wait()
        detail = process.stderr.read().decode("utf-8", "replace")
        if self._reader_error is not None:
            error, self._reader_error = self._reader_error, None
            raise RuntimeError("video destination rejected encoded bytes") from error
        if return_code:
            raise RuntimeError(f"FFmpeg encoding failed: {detail.strip()}")

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception, traceback):
        self.close()
