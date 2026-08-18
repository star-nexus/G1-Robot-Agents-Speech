"""Interruptible low-latency ALSA playback with an AEC render-reference tee."""

from __future__ import annotations

import logging
import re
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from .input import AudioInputUnavailable, read_alsa_card_numbers
from .processing import AudioProcessor, Pcm16Resampler
from ..config import AudioOutputConfig

logger = logging.getLogger(__name__)

_ALSA_HARDWARE_NAME = re.compile(r"\(hw:(\d+),(\d+)\)\s*$")


class _PcmRingBuffer:
    """Fixed-capacity mono PCM16 ring in the hardware output clock domain."""

    def __init__(self, capacity_samples: int) -> None:
        if capacity_samples < 1:
            raise ValueError("ring buffer capacity must be positive")
        self._samples = np.empty(capacity_samples, dtype="<i2")
        self._read = 0
        self._write = 0
        self.size = 0

    @property
    def capacity(self) -> int:
        return int(self._samples.size)

    @property
    def writable(self) -> int:
        return self.capacity - self.size

    def write(self, samples: np.ndarray) -> int:
        count = min(int(samples.size), self.writable)
        if count == 0:
            return 0
        first = min(count, self.capacity - self._write)
        self._samples[self._write : self._write + first] = samples[:first]
        remaining = count - first
        if remaining:
            self._samples[:remaining] = samples[first : first + remaining]
        self._write = (self._write + count) % self.capacity
        self.size += count
        return count

    def read(self, count: int) -> np.ndarray:
        count = min(count, self.size)
        result = np.empty(count, dtype="<i2")
        if count == 0:
            return result
        first = min(count, self.capacity - self._read)
        result[:first] = self._samples[self._read : self._read + first]
        remaining = count - first
        if remaining:
            result[first:] = self._samples[:remaining]
        self._read = (self._read + count) % self.capacity
        self.size -= count
        return result

    def clear(self) -> None:
        self._read = 0
        self._write = 0
        self.size = 0


def resolve_alsa_output_device(
    sounddevice_module: Any,
    *,
    card_id: str | None,
    pcm_device: int,
    proc_root: Path = Path("/proc/asound"),
) -> tuple[int, str]:
    candidates: list[tuple[int, str, int, int]] = []
    for index, device in enumerate(sounddevice_module.query_devices()):
        if int(device.get("max_output_channels", 0)) < 1:
            continue
        name = str(device.get("name", ""))
        match = _ALSA_HARDWARE_NAME.search(name)
        if match is not None:
            candidates.append((index, name, int(match.group(1)), int(match.group(2))))

    if card_id is None:
        matching = [item for item in candidates if item[3] == pcm_device]
        if len(matching) != 1:
            names = ", ".join(item[1] for item in matching) or "none"
            raise AudioInputUnavailable(
                "audio_output.alsa_card is unset and hardware output auto-detection "
                f"is ambiguous (candidates: {names})"
            )
        index, name, _card_number, _device_number = matching[0]
        return index, name

    cards = read_alsa_card_numbers(proc_root)
    if card_id not in cards:
        available = ", ".join(sorted(cards)) or "none"
        raise AudioInputUnavailable(
            f"ALSA output card ID {card_id!r} is unavailable (available: {available})"
        )
    card_number = cards[card_id]
    for index, name, candidate_card, candidate_device in candidates:
        if candidate_card == card_number and candidate_device == pcm_device:
            return index, name
    raise AudioInputUnavailable(
        f"ALSA output {card_id!r} maps to hw:{card_number},{pcm_device}, but "
        "PortAudio cannot see that hardware output"
    )


class AlsaOutputPlayer:
    """Continuous bounded PCM timeline consumed by a dedicated ALSA thread."""

    def __init__(
        self,
        settings: AudioOutputConfig,
        *,
        render_sink: AudioProcessor,
        sounddevice_module: Any | None = None,
        proc_asound_root: Path = Path("/proc/asound"),
    ) -> None:
        self._settings = settings
        self._render_sink = render_sink
        self._sd = sounddevice_module
        self._proc_root = proc_asound_root
        self._stream: Any | None = None
        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)
        self._source_rate: int | None = None
        self._resampler: Pcm16Resampler | None = None
        capacity = max(1, round(settings.sample_rate * settings.buffer_seconds))
        self._ring = _PcmRingBuffer(capacity)
        self._block_samples = settings.sample_rate * settings.block_ms // 1000
        self._thread: threading.Thread | None = None
        self._stop = False
        self._drain_requested = False
        self._inflight_samples = 0
        self._abort_epoch = 0
        self._aborted = False
        self.active_device: str | None = None
        self.actual_latency_seconds: float | None = None
        self.blocks_written = 0
        self.bytes_written = 0
        self.interruptions = 0
        self.backpressure_waits = 0
        self.peak_buffered_samples = 0
        self.playback_errors = 0

    def start(self) -> None:
        with self._lock:
            if self._stream is not None:
                return
            if self._sd is None:
                import sounddevice as sd

                self._sd = sd
            index, name = resolve_alsa_output_device(
                self._sd,
                card_id=self._settings.alsa_card,
                pcm_device=self._settings.alsa_device,
                proc_root=self._proc_root,
            )
            stream = self._sd.RawOutputStream(
                device=index,
                samplerate=self._settings.sample_rate,
                blocksize=self._block_samples,
                channels=self._settings.channels,
                dtype=self._settings.dtype,
                latency=self._settings.latency,
            )
            stream.start()
            self._stream = stream
            self._stop = False
            self._thread = threading.Thread(
                target=self._playback_loop,
                name="speech-pcm-playback",
                daemon=True,
            )
            self._thread.start()
            self.active_device = name
            latency = getattr(stream, "latency", None)
            self.actual_latency_seconds = float(latency) if latency is not None else None
            logger.info(
                "Speaker started: backend=alsa device=%r latency_ms=%s "
                "output=%dHz/%dch/%s block_ms=%d",
                name,
                "unknown" if latency is None else f"{float(latency) * 1000:.1f}",
                self._settings.sample_rate,
                self._settings.channels,
                self._settings.dtype,
                self._settings.block_ms,
            )

    def enqueue(
        self,
        pcm16_mono: bytes,
        source_sample_rate: int,
        *,
        cancel: threading.Event | None = None,
    ) -> bool:
        """Append PCM to the timeline, blocking only when bounded look-ahead is full."""
        if not pcm16_mono:
            return True
        with self._condition:
            if self._stream is None or self._stop:
                raise RuntimeError("audio output is not started")
            if cancel is not None and cancel.is_set():
                return False
            if self._source_rate != source_sample_rate:
                self._source_rate = source_sample_rate
                self._resampler = Pcm16Resampler(
                    source_sample_rate,
                    self._settings.sample_rate,
                )
            assert self._resampler is not None
            converted = self._resampler.process(pcm16_mono)
            samples = np.frombuffer(converted, dtype="<i2")
            self._aborted = False
            self._drain_requested = False

        offset = 0
        while offset < samples.size:
            with self._condition:
                if self._stop or self._aborted:
                    return False
                if cancel is not None and cancel.is_set():
                    return False
                written = self._ring.write(samples[offset:])
                if written:
                    offset += written
                    self.peak_buffered_samples = max(
                        self.peak_buffered_samples,
                        self._ring.size,
                    )
                    self._condition.notify_all()
                    continue
                self.backpressure_waits += 1
                self._condition.wait(timeout=0.05)
        return True

    def wait_until_idle(
        self,
        *,
        cancel: threading.Event | None = None,
        timeout: float | None = None,
    ) -> bool:
        """Flush a partial final block and wait until the PCM timeline is consumed."""
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._condition:
            self._drain_requested = True
            self._condition.notify_all()
            while self._ring.size or self._inflight_samples:
                if cancel is not None and cancel.is_set():
                    return False
                if deadline is not None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return False
                    self._condition.wait(timeout=min(remaining, 0.05))
                else:
                    self._condition.wait(timeout=0.05)
            self._drain_requested = False
            return True

    def abort(self) -> None:
        with self._condition:
            self._aborted = True
            self._ring.clear()
            self._drain_requested = False
            self._abort_epoch += 1
            stream = self._stream
            self.interruptions += 1
            self._condition.notify_all()
        if stream is not None:
            try:
                stream.abort()
            except Exception:  # noqa: BLE001
                logger.exception("ALSA playback abort failed")

    def close(self) -> None:
        with self._condition:
            stream = self._stream
            thread = self._thread
            self._stop = True
            self._ring.clear()
            self._abort_epoch += 1
            self._condition.notify_all()
        if stream is not None:
            try:
                stream.abort()
            except Exception:  # noqa: BLE001
                logger.exception("ALSA playback close abort failed")
        if thread is not None:
            thread.join(timeout=2.0)
            if thread.is_alive():
                logger.warning("ALSA playback thread did not stop within 2s")
        with self._condition:
            self._thread = None
            self._stream = None
            self._inflight_samples = 0
        if stream is not None:
            stream.close()

    def metrics(self) -> dict[str, int | float | str | None]:
        with self._condition:
            buffered_samples = self._ring.size + self._inflight_samples
        return {
            "audio_output_device": self.active_device,
            "audio_output_latency_ms": (
                None
                if self.actual_latency_seconds is None
                else round(self.actual_latency_seconds * 1000, 3)
            ),
            "audio_output_blocks": self.blocks_written,
            "audio_output_bytes": self.bytes_written,
            "audio_output_interruptions": self.interruptions,
            "audio_output_buffered_ms": round(
                buffered_samples * 1000 / self._settings.sample_rate,
                3,
            ),
            "audio_output_peak_buffered_ms": round(
                self.peak_buffered_samples * 1000 / self._settings.sample_rate,
                3,
            ),
            "audio_output_buffer_capacity_ms": round(
                self._ring.capacity * 1000 / self._settings.sample_rate,
                3,
            ),
            "audio_output_backpressure_waits": self.backpressure_waits,
            "audio_output_playback_errors": self.playback_errors,
        }

    def _playback_loop(self) -> None:
        while True:
            with self._condition:
                while True:
                    if self._stop:
                        return
                    available = self._ring.size
                    if available >= self._block_samples:
                        count = self._block_samples
                        break
                    if self._drain_requested and available:
                        count = available
                        break
                    self._condition.wait()
                block = self._ring.read(count)
                epoch = self._abort_epoch
                self._inflight_samples = count
                self._condition.notify_all()

            if count < self._block_samples:
                block = np.pad(block, (0, self._block_samples - count))
            pcm = block.astype("<i2", copy=False).tobytes()
            output, reference = self._prepare_block(pcm)

            try:
                with self._condition:
                    if epoch != self._abort_epoch or self._stop:
                        continue
                    stream = self._stream
                    if stream is None:
                        return
                    if not getattr(stream, "active", True):
                        stream.start()
                self._render_sink.push_render(reference, self._settings.sample_rate)
                stream.write(output)
                self.blocks_written += 1
                self.bytes_written += len(output)
            except Exception:  # noqa: BLE001
                with self._condition:
                    interrupted = epoch != self._abort_epoch or self._stop
                if not interrupted:
                    self.playback_errors += 1
                    logger.exception("ALSA continuous playback failed")
            finally:
                with self._condition:
                    self._inflight_samples = 0
                    if not self._ring.size:
                        self._drain_requested = False
                    self._condition.notify_all()

    def _prepare_block(self, pcm: bytes) -> tuple[bytes, np.ndarray]:
        mono = np.frombuffer(pcm, dtype="<i2").astype(np.float32)
        mono *= self._settings.volume
        mono_i16 = np.rint(np.clip(mono, -32768, 32767)).astype("<i2")
        reference = mono_i16.astype(np.float32) / 32768.0
        if self._settings.channels == 1:
            return mono_i16.tobytes(), reference
        stereo = np.repeat(mono_i16[:, None], self._settings.channels, axis=1)
        return stereo.astype("<i2", copy=False).tobytes(), reference
