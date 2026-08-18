"""Pluggable microphone processing and WebRTC APM render-reference bridge."""

from __future__ import annotations

import audioop
import importlib
import logging
import threading
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from ..config import AudioProcessingConfig

logger = logging.getLogger(__name__)


class AudioProcessor(Protocol):
    mode: str

    def process_capture(self, samples: np.ndarray, sample_rate: int) -> np.ndarray: ...

    def push_render(self, samples: np.ndarray, sample_rate: int) -> None: ...

    def metrics(self) -> dict[str, int | str | bool]: ...

    def close(self) -> None: ...


@dataclass
class _ProcessingCounters:
    capture_frames: int = 0
    render_frames: int = 0
    capture_errors: int = 0
    render_errors: int = 0


class Pcm16Resampler:
    """Stateful PCM16 rate conversion for playback and AEC reference streams."""

    def __init__(self, input_rate: int, output_rate: int, channels: int = 1) -> None:
        self.input_rate = input_rate
        self.output_rate = output_rate
        self.channels = channels
        self._state: Any = None

    def process(self, pcm: bytes) -> bytes:
        if not pcm:
            return b""
        if self.input_rate == self.output_rate:
            return pcm
        converted, self._state = audioop.ratecv(
            pcm,
            2,
            self.channels,
            self.input_rate,
            self.output_rate,
            self._state,
        )
        return converted

    def reset(self) -> None:
        self._state = None


class PassthroughAudioProcessor:
    """Used for both unprocessed and hardware-DSP microphone paths."""

    def __init__(self, mode: str) -> None:
        self.mode = mode

    def process_capture(self, samples: np.ndarray, sample_rate: int) -> np.ndarray:
        del sample_rate
        return np.asarray(samples, dtype=np.float32)

    def push_render(self, samples: np.ndarray, sample_rate: int) -> None:
        del samples, sample_rate

    def metrics(self) -> dict[str, int | str | bool]:
        return {
            "audio_processing_mode": self.mode,
            "aec_active": self.mode == "hardware",
        }

    def close(self) -> None:
        return None


class WebRtcApmProcessor:
    """10 ms WebRTC APM adapter with synchronized capture and render streams.

    The native binding is deliberately optional. It is imported only when the
    host selects ``mode=webrtc``; stable ``off`` and hardware-DSP deployments do
    not inherit a C++ runtime dependency.
    """

    mode = "webrtc"

    def __init__(
        self,
        settings: AudioProcessingConfig,
        *,
        apm: Any | None = None,
    ) -> None:
        self._settings = settings
        self._sample_rate = settings.sample_rate
        self._frame_samples = settings.sample_rate * settings.frame_ms // 1000
        self._frame_bytes = self._frame_samples * 2
        self._render_buffer = bytearray()
        self._render_resampler: Pcm16Resampler | None = None
        self._render_input_rate: int | None = None
        self._lock = threading.RLock()
        self._counters = _ProcessingCounters()
        self._closed = False
        self._apm = apm or self._create_native_apm(settings)
        self._configure_native_apm()

    @staticmethod
    def _create_native_apm(settings: AudioProcessingConfig) -> Any:
        errors: list[str] = []
        candidates = (
            ("aec_audio_processing", "AudioProcessor"),
            ("webrtc_audio_processing", "AudioProcessingModule"),
        )
        for module_name, class_name in candidates:
            try:
                module = importlib.import_module(module_name)
                cls = getattr(module, class_name)
                options = {
                    "enable_aec": settings.echo_cancellation,
                    "enable_ns": settings.noise_suppression,
                    "enable_agc": settings.automatic_gain_control,
                    "enable_vad": False,
                }
                if module_name == "aec_audio_processing":
                    options["ns_level"] = settings.noise_suppression_level
                return cls(**options)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{module_name}: {type(exc).__name__}: {exc}")
        detail = "; ".join(errors)
        raise RuntimeError(
            "AUDIO_PROCESSING_MODE=webrtc requires a native WebRTC APM binding "
            f"with reverse-stream support ({detail})"
        )

    def _configure_native_apm(self) -> None:
        try:
            self._apm.set_stream_format(
                self._sample_rate,
                1,
                self._sample_rate,
                1,
            )
        except TypeError:
            self._apm.set_stream_format(self._sample_rate, 1)
        reverse = getattr(self._apm, "set_reverse_stream_format", None)
        process_reverse = getattr(self._apm, "process_reverse_stream", None)
        if reverse is None or process_reverse is None:
            raise RuntimeError(
                "the installed WebRTC APM binding lacks reverse-stream AEC support"
            )
        reverse(self._sample_rate, 1)
        set_delay = getattr(self._apm, "set_stream_delay", None)
        if set_delay is not None:
            set_delay(self._settings.stream_delay_ms)
        set_ns_level = getattr(self._apm, "set_ns_level", None)
        if set_ns_level is not None and self._settings.noise_suppression:
            set_ns_level(self._settings.noise_suppression_level)
        logger.info(
            "WebRTC APM ready: rate=%dHz frame=%dms aec=%s ns=%s agc=%s delay=%dms",
            self._sample_rate,
            self._settings.frame_ms,
            self._settings.echo_cancellation,
            self._settings.noise_suppression,
            self._settings.automatic_gain_control,
            self._settings.stream_delay_ms,
        )

    def process_capture(self, samples: np.ndarray, sample_rate: int) -> np.ndarray:
        if sample_rate != self._sample_rate:
            raise ValueError(
                f"WebRTC APM expected {self._sample_rate}Hz capture, got {sample_rate}Hz"
            )
        values = np.asarray(samples, dtype=np.float32).reshape(-1)
        if values.size % self._frame_samples:
            raise ValueError("capture block does not contain complete 10ms APM frames")
        pcm = self._float_to_pcm16(values)
        output = bytearray()
        with self._lock:
            self._ensure_open()
            try:
                for offset in range(0, len(pcm), self._frame_bytes):
                    frame = pcm[offset : offset + self._frame_bytes]
                    processed = self._apm.process_stream(frame)
                    output.extend(self._as_bytes(processed))
                    self._counters.capture_frames += 1
            except Exception:
                self._counters.capture_errors += 1
                raise
        return self._pcm16_to_float(bytes(output))

    def push_render(self, samples: np.ndarray, sample_rate: int) -> None:
        values = np.asarray(samples, dtype=np.float32).reshape(-1)
        if not values.size:
            return
        pcm = self._float_to_pcm16(values)
        with self._lock:
            self._ensure_open()
            if self._render_input_rate != sample_rate:
                self._render_input_rate = sample_rate
                self._render_resampler = Pcm16Resampler(sample_rate, self._sample_rate)
                self._render_buffer.clear()
            assert self._render_resampler is not None
            self._render_buffer.extend(self._render_resampler.process(pcm))
            try:
                while len(self._render_buffer) >= self._frame_bytes:
                    frame = bytes(self._render_buffer[: self._frame_bytes])
                    del self._render_buffer[: self._frame_bytes]
                    self._apm.process_reverse_stream(frame)
                    self._counters.render_frames += 1
            except Exception:
                self._counters.render_errors += 1
                raise

    def metrics(self) -> dict[str, int | str | bool]:
        with self._lock:
            counters = _ProcessingCounters(**vars(self._counters))
        return {
            "audio_processing_mode": self.mode,
            "aec_active": self._settings.echo_cancellation,
            "webrtc_capture_frames": counters.capture_frames,
            "webrtc_render_frames": counters.render_frames,
            "webrtc_capture_errors": counters.capture_errors,
            "webrtc_render_errors": counters.render_errors,
        }

    def close(self) -> None:
        with self._lock:
            self._closed = True
            close = getattr(self._apm, "close", None)
            if close is not None:
                close()

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("WebRTC APM processor is closed")

    @staticmethod
    def _float_to_pcm16(samples: np.ndarray) -> bytes:
        scaled = np.rint(np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2")
        return scaled.tobytes()

    @staticmethod
    def _pcm16_to_float(pcm: bytes) -> np.ndarray:
        return np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0

    @staticmethod
    def _as_bytes(value: Any) -> bytes:
        if isinstance(value, bytes):
            return value
        if isinstance(value, bytearray):
            return bytes(value)
        if isinstance(value, str):
            return value.encode("latin1")
        return bytes(value)


def create_audio_processor(settings: AudioProcessingConfig) -> AudioProcessor:
    if settings.mode in {"off", "hardware"}:
        return PassthroughAudioProcessor(settings.mode)
    if settings.mode == "webrtc":
        return WebRtcApmProcessor(settings)
    raise ValueError(f"unsupported audio processing mode: {settings.mode!r}")
