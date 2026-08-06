"""Bounded, non-blocking microphone capture."""

from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Any

import numpy as np

from .config import AudioConfig
from .contracts import AudioChunk

logger = logging.getLogger(__name__)


class SoundDeviceSource:
    """Capture 16 kHz mono PCM without doing work in PortAudio's callback."""

    def __init__(
        self,
        *,
        settings: AudioConfig,
        sounddevice_module: Any | None = None,
    ) -> None:
        self.sample_rate = settings.sample_rate
        self.block_samples = round(settings.sample_rate * settings.block_ms / 1000)
        self.device = settings.device
        self._heartbeat_timeout = settings.heartbeat_timeout_seconds
        self._reconnect_initial = settings.reconnect_initial_seconds
        self._reconnect_max = settings.reconnect_max_seconds
        capacity = max(2, round(settings.queue_seconds * 1000 / settings.block_ms))
        self._queue: queue.Queue[AudioChunk] = queue.Queue(maxsize=capacity)
        self._sd = sounddevice_module
        self._stream = None
        self._lock = threading.Lock()
        self._closed = True
        self._last_callback_monotonic = 0.0
        self._next_reconnect_monotonic = 0.0
        self._reconnect_delay = self._reconnect_initial
        self._outage_active = False
        self.dropped_chunks = 0
        self.callback_errors = 0
        self.reconnections = 0
        self.reconnect_failures = 0
        self.discontinuity_count = 0

    @property
    def queue_size(self) -> int:
        return self._queue.qsize()

    @property
    def queue_capacity(self) -> int:
        return self._queue.maxsize

    def start(self) -> None:
        with self._lock:
            if self._stream is not None:
                return
            if self._sd is None:
                import sounddevice as sd

                self._sd = sd
            self._closed = False
            try:
                self._stream = self._open_stream()
            except Exception:
                self._closed = True
                raise
            self._last_callback_monotonic = time.monotonic()
            self._next_reconnect_monotonic = 0.0
            self._reconnect_delay = self._reconnect_initial
            self._outage_active = False
            logger.info(
                "Microphone started: configured_device=%r portaudio_device=%r "
                "sample_rate=%d block=%d capacity=%d",
                self.device,
                self._resolved_device_name(),
                self.sample_rate,
                self.block_samples,
                self._queue.maxsize,
            )

    def read(self, timeout: float = 0.2) -> AudioChunk | None:
        self._maintain_stream()
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            self._maintain_stream()
            return None

    def close(self) -> None:
        with self._lock:
            self._closed = True
            stream, self._stream = self._stream, None
            self._outage_active = False
        self._dispose_stream(stream)
        self._drain_queue()

    def _callback(self, indata, _frames, _time_info, status) -> None:
        if status:
            logger.warning("PortAudio status: %s", status)
        try:
            samples = np.asarray(indata[:, 0], dtype=np.float32).copy()
            if samples.size == 0:
                logger.warning("PortAudio callback returned an empty input block")
                return
            chunk = AudioChunk(
                samples=samples,
                sample_rate=self.sample_rate,
                captured_monotonic_ns=time.monotonic_ns(),
            )
        except Exception:  # noqa: BLE001
            self.callback_errors += 1
            logger.exception("Audio callback failed")
            return

        self._last_callback_monotonic = time.monotonic()
        try:
            self._queue.put_nowait(chunk)
        except queue.Full:
            self.dropped_chunks += 1
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait(chunk)
            except queue.Full:
                self.dropped_chunks += 1

    def _open_stream(self):
        stream = self._sd.InputStream(
            samplerate=self.sample_rate,
            blocksize=self.block_samples,
            channels=1,
            dtype="float32",
            device=self.device,
            callback=self._callback,
        )
        try:
            stream.start()
        except Exception:
            self._dispose_stream(stream)
            raise
        return stream

    def _maintain_stream(self) -> None:
        now = time.monotonic()
        with self._lock:
            if self._closed:
                return
            reason = self._unhealthy_reason(now)
            if reason is None:
                return
            if not self._outage_active:
                self._outage_active = True
                self.discontinuity_count += 1
                self._drain_queue()
                logger.error("Microphone stream lost: %s", reason)
            if now < self._next_reconnect_monotonic:
                return

            old_stream, self._stream = self._stream, None
            self._dispose_stream(old_stream)
            try:
                self._stream = self._open_stream()
            except Exception:  # noqa: BLE001
                self.reconnect_failures += 1
                delay = self._reconnect_delay
                self._next_reconnect_monotonic = now + delay
                self._reconnect_delay = min(delay * 2, self._reconnect_max)
                logger.exception(
                    "Microphone reconnect failed; retrying in %.1fs", delay
                )
                return

            self._last_callback_monotonic = time.monotonic()
            self._next_reconnect_monotonic = 0.0
            self._reconnect_delay = self._reconnect_initial
            self._outage_active = False
            self.reconnections += 1
            logger.info(
                "Microphone reconnected: configured_device=%r portaudio_device=%r",
                self.device,
                self._resolved_device_name(),
            )

    def _resolved_device_name(self) -> str:
        query_devices = getattr(self._sd, "query_devices", None)
        if query_devices is None:
            return str(self.device)
        try:
            device = query_devices(self.device, "input")
        except Exception:  # noqa: BLE001
            logger.debug("Unable to resolve microphone device name", exc_info=True)
            return str(self.device)
        if isinstance(device, dict):
            return str(device.get("name", self.device))
        try:
            return str(device["name"])
        except (KeyError, TypeError):
            return str(self.device)

    def _unhealthy_reason(self, now: float) -> str | None:
        if self._stream is None:
            return "stream is unavailable"
        try:
            if not bool(getattr(self._stream, "active", True)):
                return "stream is inactive"
        except Exception as exc:  # noqa: BLE001
            return f"stream state check failed: {exc}"
        silence = now - self._last_callback_monotonic
        if silence > self._heartbeat_timeout:
            return f"no audio callback for {silence:.1f}s"
        return None

    @staticmethod
    def _dispose_stream(stream) -> None:
        if stream is None:
            return
        try:
            abort = getattr(stream, "abort", None)
            if abort is not None:
                abort()
            else:
                stream.stop()
        except Exception:  # noqa: BLE001
            logger.debug("Ignoring error while stopping microphone stream", exc_info=True)
        try:
            stream.close()
        except Exception:  # noqa: BLE001
            logger.debug("Ignoring error while closing microphone stream", exc_info=True)

    def _drain_queue(self) -> None:
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                return
