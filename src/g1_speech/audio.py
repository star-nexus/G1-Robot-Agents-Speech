"""Bounded, non-blocking microphone capture."""

from __future__ import annotations

import logging
import queue
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .config import AudioConfig
from .contracts import AudioChunk

logger = logging.getLogger(__name__)

_ALSA_HARDWARE_NAME = re.compile(r"\(hw:(\d+),(\d+)\)\s*$")


def _design_decimation_filter(factor: int) -> np.ndarray:
    taps = 16 * factor + 1
    center = (taps - 1) / 2
    positions = np.arange(taps, dtype=np.float64) - center
    cutoff = 0.45 / factor
    coefficients = 2 * cutoff * np.sinc(2 * cutoff * positions)
    coefficients *= np.hamming(taps)
    coefficients /= coefficients.sum()
    return coefficients.astype(np.float32)


class AudioInputUnavailable(RuntimeError):
    """Raised when the requested physical input cannot be resolved or opened."""


@dataclass(frozen=True)
class _CapturedBlock:
    samples: np.ndarray
    sample_rate: int
    captured_monotonic_ns: int
    sequence: int


def read_alsa_card_numbers(proc_root: Path = Path("/proc/asound")) -> dict[str, int]:
    """Map stable ALSA card IDs to their current, hot-plug-sensitive numbers."""
    cards: dict[str, int] = {}
    for id_path in proc_root.glob("card[0-9]*/id"):
        match = re.fullmatch(r"card(\d+)", id_path.parent.name)
        if match is None:
            continue
        try:
            card_id = id_path.read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if card_id:
            cards[card_id] = int(match.group(1))
    return cards


def resolve_alsa_input_device(
    sounddevice_module: Any,
    *,
    card_id: str | None,
    pcm_device: int,
    proc_root: Path = Path("/proc/asound"),
) -> tuple[int, str]:
    """Resolve a stable ALSA card ID to an exact PortAudio hw device."""
    candidates: list[tuple[int, str, int, int]] = []
    for index, device in enumerate(sounddevice_module.query_devices()):
        if int(device.get("max_input_channels", 0)) < 1:
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
                "ALSA_INPUT_CARD is unset and hardware input auto-detection is "
                f"ambiguous (candidates: {names})"
            )
        index, name, _card_number, _device_number = matching[0]
        return index, name

    cards = read_alsa_card_numbers(proc_root)
    if card_id not in cards:
        available = ", ".join(sorted(cards)) or "none"
        raise AudioInputUnavailable(
            f"ALSA card ID {card_id!r} is unavailable (available IDs: {available})"
        )
    card_number = cards[card_id]
    for index, name, candidate_card, candidate_device in candidates:
        if candidate_card == card_number and candidate_device == pcm_device:
            return index, name
    raise AudioInputUnavailable(
        f"ALSA {card_id!r} currently maps to hw:{card_number},{pcm_device}, but "
        "PortAudio cannot see that hardware input; PulseAudio may still own it"
    )


class SoundDeviceSource:
    """Capture 16 kHz mono PCM without doing work in PortAudio's callback."""

    def __init__(
        self,
        *,
        settings: AudioConfig,
        sounddevice_module: Any | None = None,
        proc_asound_root: Path = Path("/proc/asound"),
    ) -> None:
        self.sample_rate = settings.sample_rate
        self.block_samples = round(settings.sample_rate * settings.block_ms / 1000)
        self.input_backend = settings.input_backend
        self.fallback_backend = settings.fallback_backend
        self.alsa_card = settings.alsa_card
        self.alsa_device = settings.alsa_device
        self.alsa_sample_rate = settings.alsa_sample_rate
        self.alsa_channels = settings.alsa_channels
        self.alsa_dtype = settings.alsa_dtype
        self.pulse_device = settings.pulse_device
        self.latency = settings.latency
        self._proc_asound_root = proc_asound_root
        self._heartbeat_timeout = settings.heartbeat_timeout_seconds
        self._reconnect_initial = settings.reconnect_initial_seconds
        self._reconnect_max = settings.reconnect_max_seconds
        capacity = max(2, round(settings.queue_seconds * 1000 / settings.block_ms))
        self._queue: queue.Queue[_CapturedBlock] = queue.Queue(maxsize=capacity)
        self._sd = sounddevice_module
        self._stream = None
        self._lock = threading.Lock()
        self._closed = True
        self._last_callback_monotonic = 0.0
        self._next_reconnect_monotonic = 0.0
        self._reconnect_delay = self._reconnect_initial
        self._outage_active = False
        self._capture_sequence = 0
        self._last_converted_sequence: int | None = None
        self.dropped_chunks = 0
        self.callback_errors = 0
        self.reconnections = 0
        self.reconnect_failures = 0
        self.discontinuity_count = 0
        self.active_backend: str | None = None
        self.active_device: str | None = None
        self.actual_latency_seconds: float | None = None
        self.active_sample_rate: int | None = None
        self.active_channels: int | None = None
        self.active_dtype: str | None = None
        factor = self.alsa_sample_rate // self.sample_rate
        self._decimation_filter = _design_decimation_filter(factor)
        self._filter_state = np.zeros(
            self._decimation_filter.size - 1, dtype=np.float32
        )

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
                "Microphone started: backend=%s device=%r latency_ms=%s "
                "capture=%sHz/%sch/%s pipeline=%dHz/mono block_ms=%.1f capacity=%d",
                self.active_backend,
                self.active_device,
                self._latency_ms_text(),
                self.active_sample_rate,
                self.active_channels,
                self.active_dtype,
                self.sample_rate,
                self.block_samples * 1000 / self.sample_rate,
                self._queue.maxsize,
            )

    def read(self, timeout: float = 0.2) -> AudioChunk | None:
        self._maintain_stream()
        try:
            captured = self._queue.get(timeout=timeout)
        except queue.Empty:
            self._maintain_stream()
            return None
        return AudioChunk(
            samples=self._convert_to_pipeline_format(captured),
            sample_rate=self.sample_rate,
            captured_monotonic_ns=captured.captured_monotonic_ns,
        )

    def close(self) -> None:
        with self._lock:
            self._closed = True
            stream, self._stream = self._stream, None
            self._outage_active = False
        self._dispose_stream(stream)
        self._drain_queue()

    def _callback(
        self,
        indata,
        _frames,
        _time_info,
        status,
        *,
        input_sample_rate: int,
    ) -> None:
        if status:
            logger.warning("PortAudio status: %s", status)
        try:
            samples = np.asarray(indata).copy()
            if samples.size == 0:
                logger.warning("PortAudio callback returned an empty input block")
                return
            chunk = _CapturedBlock(
                samples=samples,
                sample_rate=input_sample_rate,
                captured_monotonic_ns=time.monotonic_ns(),
                sequence=self._capture_sequence,
            )
        except Exception:  # noqa: BLE001
            self.callback_errors += 1
            logger.exception("Audio callback failed")
            return

        self._capture_sequence += 1
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
        backends = [self.input_backend]
        if (
            self.fallback_backend is not None
            and self.fallback_backend != self.input_backend
        ):
            backends.append(self.fallback_backend)
        failures: list[str] = []
        for position, backend in enumerate(backends):
            try:
                stream, device_name, stream_format = self._open_backend(backend)
            except Exception as exc:  # noqa: BLE001
                failures.append(f"{backend}: {type(exc).__name__}: {exc}")
                if position + 1 < len(backends):
                    logger.warning(
                        "Preferred audio input %s failed; falling back to %s: %s",
                        backend,
                        backends[position + 1],
                        exc,
                    )
                continue
            self.active_backend = backend
            self.active_device = device_name
            (
                self.active_sample_rate,
                self.active_channels,
                self.active_dtype,
            ) = stream_format
            stream_latency = getattr(stream, "latency", None)
            self.actual_latency_seconds = (
                float(stream_latency) if stream_latency is not None else None
            )
            return stream
        raise AudioInputUnavailable("; ".join(failures))

    def _open_backend(self, backend: str):
        if backend == "alsa":
            device, device_name = resolve_alsa_input_device(
                self._sd,
                card_id=self.alsa_card,
                pcm_device=self.alsa_device,
                proc_root=self._proc_asound_root,
            )
            input_sample_rate = self.alsa_sample_rate
            channels = self.alsa_channels
            dtype = self.alsa_dtype
        elif backend == "pulse":
            device = self.pulse_device
            device_name = self._device_name(device)
            input_sample_rate = self.sample_rate
            channels = 1
            dtype = "float32"
        else:
            raise AudioInputUnavailable(f"unsupported input backend: {backend}")

        input_block_samples = round(
            input_sample_rate * self.block_samples / self.sample_rate
        )

        def callback(indata, frames, time_info, status):
            self._callback(
                indata,
                frames,
                time_info,
                status,
                input_sample_rate=input_sample_rate,
            )
        stream = self._sd.InputStream(
            samplerate=input_sample_rate,
            blocksize=input_block_samples,
            channels=channels,
            dtype=dtype,
            device=device,
            latency=self.latency,
            callback=callback,
        )
        try:
            stream.start()
        except Exception:
            self._dispose_stream(stream)
            raise
        return stream, device_name, (input_sample_rate, channels, dtype)

    def _convert_to_pipeline_format(self, captured: _CapturedBlock) -> np.ndarray:
        if (
            self._last_converted_sequence is not None
            and captured.sequence != self._last_converted_sequence + 1
        ):
            self._filter_state.fill(0)
        self._last_converted_sequence = captured.sequence
        samples = captured.samples
        if samples.dtype == np.int16:
            normalized = samples.astype(np.float32) / 32768.0
        else:
            normalized = np.asarray(samples, dtype=np.float32)
        if normalized.ndim == 2:
            if normalized.shape[1] == 1:
                mono = normalized[:, 0]
            else:
                mono = normalized.mean(axis=1, dtype=np.float32)
        else:
            mono = normalized.reshape(-1)

        if captured.sample_rate == self.sample_rate:
            return np.ascontiguousarray(mono, dtype=np.float32)
        if captured.sample_rate % self.sample_rate:
            raise AudioInputUnavailable(
                f"cannot resample {captured.sample_rate} Hz to {self.sample_rate} Hz"
            )

        factor = captured.sample_rate // self.sample_rate
        combined = np.concatenate((self._filter_state, mono))
        filtered = np.convolve(
            combined, self._decimation_filter, mode="valid"
        ).astype(np.float32, copy=False)
        self._filter_state = combined[-(self._decimation_filter.size - 1) :].copy()
        # The fixed callback block is an exact multiple of the integer rate
        # ratio, so this preserves phase across blocks. FIR work happens in the
        # consumer thread, not PortAudio's real-time callback.
        return np.ascontiguousarray(filtered[::factor], dtype=np.float32)

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
                "Microphone reconnected: backend=%s device=%r latency_ms=%s",
                self.active_backend,
                self.active_device,
                self._latency_ms_text(),
            )

    def _device_name(self, device: int | str | None) -> str:
        query_devices = getattr(self._sd, "query_devices", None)
        if query_devices is None:
            return str(device)
        try:
            details = query_devices(device, "input")
        except Exception:  # noqa: BLE001
            logger.debug("Unable to resolve microphone device name", exc_info=True)
            return str(device)
        if isinstance(details, dict):
            return str(details.get("name", device))
        try:
            return str(details["name"])
        except (KeyError, TypeError):
            return str(device)

    def _latency_ms_text(self) -> str:
        if self.actual_latency_seconds is None:
            return "unknown"
        return f"{self.actual_latency_seconds * 1000:.1f}"

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
        self._filter_state.fill(0)
        self._last_converted_sequence = None
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                return
