"""Low-copy V4L2 MJPEG camera source for Jetson deployments."""

from __future__ import annotations

import logging
import shutil
import subprocess
import threading
import time

from ..contracts import ImageFrame


logger = logging.getLogger(__name__)


class MjpegStreamParser:
    """Extract complete JPEG images from an arbitrary byte stream."""

    _SOI = b"\xff\xd8"
    _EOI = b"\xff\xd9"

    def __init__(self, max_buffer_bytes: int = 16 * 1024 * 1024) -> None:
        if max_buffer_bytes < 4:
            raise ValueError("max_buffer_bytes must be at least four")
        self._buffer = bytearray()
        self._max_buffer_bytes = max_buffer_bytes

    def feed(self, chunk: bytes) -> list[bytes]:
        self._buffer.extend(chunk)
        frames: list[bytes] = []
        while True:
            start = self._buffer.find(self._SOI)
            if start < 0:
                if len(self._buffer) > 1:
                    self._buffer[:] = self._buffer[-1:]
                break
            end = self._buffer.find(self._EOI, start + 2)
            if end < 0:
                if start:
                    del self._buffer[:start]
                break
            end += 2
            frames.append(bytes(self._buffer[start:end]))
            del self._buffer[:end]
        if len(self._buffer) > self._max_buffer_bytes:
            logger.warning("Discarding oversized incomplete MJPEG frame")
            self._buffer.clear()
        return frames


class LatestFrameCamera:
    """Continuously retain only the newest encoded MJPEG frame."""

    def __init__(
        self,
        *,
        device: str,
        width: int = 640,
        height: int = 480,
        fps: int = 5,
        ffmpeg: str = "ffmpeg",
        reconnect_seconds: float = 1.0,
    ) -> None:
        if width < 1 or height < 1 or fps < 1:
            raise ValueError("camera width, height, and fps must be positive")
        self._device = device
        self._width = width
        self._height = height
        self._fps = fps
        self._ffmpeg = ffmpeg
        self._reconnect_seconds = reconnect_seconds
        self._stopped = threading.Event()
        self._frame_ready = threading.Event()
        self._lock = threading.Lock()
        self._frame: ImageFrame | None = None
        self._process: subprocess.Popen[bytes] | None = None
        self._thread: threading.Thread | None = None

    def start(self, timeout: float = 5.0) -> None:
        if self._thread is not None:
            raise RuntimeError("camera is already started")
        if timeout <= 0:
            raise ValueError("camera start timeout must be positive")
        if shutil.which(self._ffmpeg) is None:
            raise RuntimeError(f"FFmpeg executable not found: {self._ffmpeg!r}")
        self._thread = threading.Thread(
            target=self._run,
            name="star-camera-latest-frame",
            daemon=True,
        )
        self._thread.start()
        if not self._frame_ready.wait(timeout):
            self.close()
            raise RuntimeError(
                f"No frame received from camera {self._device!r} within {timeout:.1f}s"
            )

    def snapshot(self, max_age_seconds: float = 2.0) -> ImageFrame:
        if max_age_seconds <= 0:
            raise ValueError("max_age_seconds must be positive")
        with self._lock:
            frame = self._frame
        if frame is None:
            raise RuntimeError(f"Camera {self._device!r} has no frame")
        age = time.monotonic() - frame.captured_monotonic
        if age > max_age_seconds:
            raise RuntimeError(
                f"Latest camera frame is stale: age={age:.2f}s "
                f"limit={max_age_seconds:.2f}s"
            )
        return frame

    def close(self) -> None:
        self._stopped.set()
        with self._lock:
            process = self._process
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2.0)
        if self._thread is not None:
            self._thread.join(timeout=3.0)

    def command(self) -> list[str]:
        """Return the zero-reencode capture command (also useful for diagnostics)."""

        return [
            self._ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "v4l2",
            "-input_format",
            "mjpeg",
            "-framerate",
            str(self._fps),
            "-video_size",
            f"{self._width}x{self._height}",
            "-i",
            self._device,
            "-c:v",
            "copy",
            "-f",
            "image2pipe",
            "pipe:1",
        ]

    def _run(self) -> None:
        while not self._stopped.is_set():
            parser = MjpegStreamParser()
            process: subprocess.Popen[bytes] | None = None
            try:
                process = subprocess.Popen(
                    self.command(),
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    bufsize=0,
                )
                with self._lock:
                    self._process = process
                assert process.stdout is not None
                while not self._stopped.is_set():
                    chunk = process.stdout.read(64 * 1024)
                    if not chunk:
                        break
                    for jpeg in parser.feed(chunk):
                        with self._lock:
                            self._frame = ImageFrame(jpeg, time.monotonic())
                        self._frame_ready.set()
            except OSError as exc:
                if not self._stopped.is_set():
                    logger.warning("Camera process failed: %s", exc)
            finally:
                if process is not None and process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=1.0)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=1.0)
                with self._lock:
                    if self._process is process:
                        self._process = None
            if not self._stopped.wait(self._reconnect_seconds):
                logger.warning("Camera stream stopped; reconnecting to %r", self._device)
