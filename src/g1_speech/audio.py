"""Bounded, non-blocking microphone capture."""

from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Any

import numpy as np

from .contracts import AudioChunk

logger = logging.getLogger(__name__)


class SoundDeviceSource:
    """Capture 16 kHz mono PCM without doing work in PortAudio's callback."""

    def __init__(
        self,
        *,
        sample_rate: int = 16000,
        block_ms: int = 100,
        device: int | str | None = None,
        queue_seconds: float = 5.0,
        sounddevice_module: Any | None = None,
    ) -> None:
        self.sample_rate = sample_rate
        self.block_samples = round(sample_rate * block_ms / 1000)
        self.device = device
        capacity = max(2, round(queue_seconds * 1000 / block_ms))
        self._queue: queue.Queue[AudioChunk] = queue.Queue(maxsize=capacity)
        self._sd = sounddevice_module
        self._stream = None
        self._lock = threading.Lock()
        self.dropped_chunks = 0

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

            def callback(indata, _frames, _time_info, status) -> None:
                if status:
                    logger.warning("PortAudio status: %s", status)
                chunk = AudioChunk(
                    samples=np.asarray(indata[:, 0], dtype=np.float32).copy(),
                    sample_rate=self.sample_rate,
                    captured_monotonic_ns=time.monotonic_ns(),
                )
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

            self._stream = self._sd.InputStream(
                samplerate=self.sample_rate,
                blocksize=self.block_samples,
                channels=1,
                dtype="float32",
                device=self.device,
                callback=callback,
            )
            self._stream.start()
            logger.info(
                "Microphone started: device=%r sample_rate=%d block=%d capacity=%d",
                self.device,
                self.sample_rate,
                self.block_samples,
                self._queue.maxsize,
            )

    def read(self, timeout: float = 0.2) -> AudioChunk | None:
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def close(self) -> None:
        with self._lock:
            stream, self._stream = self._stream, None
        if stream is not None:
            try:
                stream.stop()
            finally:
                stream.close()
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
