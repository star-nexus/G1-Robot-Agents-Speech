"""Silero VAD segmentation through the already-used sherpa-onnx runtime."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from .config import VadConfig
from .contracts import AudioChunk, Utterance


class _SampleRingBuffer:
    """Fixed-size float32 history addressed by absolute sample positions."""

    def __init__(self, capacity: int) -> None:
        if capacity < 1:
            raise ValueError("ring buffer capacity must be greater than zero")
        self._data = np.empty(capacity, dtype=np.float32)
        self._write_index = 0
        self._size = 0
        self._total_samples = 0

    @property
    def start_sample(self) -> int:
        return self._total_samples - self._size

    @property
    def end_sample(self) -> int:
        return self._total_samples

    def append(self, samples: np.ndarray) -> None:
        values = np.asarray(samples, dtype=np.float32).reshape(-1)
        count = values.size
        if count == 0:
            return
        self._total_samples += count
        capacity = self._data.size
        if count >= capacity:
            self._data[:] = values[-capacity:]
            self._write_index = 0
            self._size = capacity
            return

        first = min(count, capacity - self._write_index)
        self._data[self._write_index : self._write_index + first] = values[:first]
        remaining = count - first
        if remaining:
            self._data[:remaining] = values[first:]
        self._write_index = (self._write_index + count) % capacity
        self._size = min(capacity, self._size + count)

    def read(self, start_sample: int, end_sample: int) -> np.ndarray | None:
        if (
            start_sample < self.start_sample
            or end_sample > self.end_sample
            or end_sample <= start_sample
        ):
            return None
        count = end_sample - start_sample
        oldest = (self._write_index - self._size) % self._data.size
        offset = start_sample - self.start_sample
        first_index = (oldest + offset) % self._data.size
        first = min(count, self._data.size - first_index)
        result = np.empty(count, dtype=np.float32)
        result[:first] = self._data[first_index : first_index + first]
        if first < count:
            result[first:] = self._data[: count - first]
        return result

    def clear(self) -> None:
        self._write_index = 0
        self._size = 0
        self._total_samples = 0


class SileroVadSegmenter:
    def __init__(
        self,
        *,
        settings: VadConfig,
        sample_rate: int = 16000,
        sherpa_module: Any | None = None,
    ) -> None:
        model_path = Path(settings.model).expanduser().resolve()
        if not model_path.is_file():
            raise FileNotFoundError(f"Silero VAD model not found: {model_path}")
        if sample_rate != 16000:
            raise ValueError("the current Silero VAD model requires 16000 Hz audio")
        if sherpa_module is None:
            import sherpa_onnx

            sherpa_module = sherpa_onnx
        config = sherpa_module.VadModelConfig()
        config.silero_vad.model = str(model_path)
        config.silero_vad.threshold = settings.threshold
        config.silero_vad.min_silence_duration = settings.min_silence_seconds
        config.silero_vad.min_speech_duration = settings.min_speech_seconds
        config.silero_vad.max_speech_duration = settings.max_speech_seconds
        config.sample_rate = sample_rate
        self._vad = sherpa_module.VoiceActivityDetector(
            config, buffer_size_in_seconds=settings.buffer_seconds
        )
        self._sample_rate = sample_rate
        self._window_size = config.silero_vad.window_size
        self._pre_roll_samples = round(settings.speech_pre_roll_seconds * sample_rate)
        self._history_limit_samples = round(settings.buffer_seconds * sample_rate)
        self._pending = np.empty(self._window_size, dtype=np.float32)
        self._pending_size = 0
        self._history = _SampleRingBuffer(self._history_limit_samples)

    def accept(self, chunk: AudioChunk) -> list[Utterance]:
        if chunk.sample_rate != self._sample_rate:
            raise ValueError("VAD received an unexpected sample rate")
        samples = np.asarray(chunk.samples, dtype=np.float32).reshape(-1)
        offset = 0
        if self._pending_size:
            count = min(self._window_size - self._pending_size, samples.size)
            end = self._pending_size + count
            self._pending[self._pending_size : end] = samples[:count]
            self._pending_size = end
            offset = count
            if self._pending_size == self._window_size:
                self._accept_window(self._pending)
                self._pending_size = 0

        while offset + self._window_size <= samples.size:
            self._accept_window(samples[offset : offset + self._window_size])
            offset += self._window_size

        remaining = samples.size - offset
        if remaining:
            self._pending[:remaining] = samples[offset:]
            self._pending_size = remaining

        utterances: list[Utterance] = []
        while not self._vad.empty():
            segment = self._vad.front
            vad_samples = np.asarray(segment.samples, dtype=np.float32)
            segment_start = int(segment.start)
            segment_end = segment_start + vad_samples.size
            wanted_start = max(
                self._history.start_sample,
                segment_start - self._pre_roll_samples,
            )
            samples = self._history.read(wanted_start, segment_end)
            if samples is None:
                samples = vad_samples.copy()
            self._vad.pop()
            ready_ns = time.monotonic_ns()
            # PortAudio timestamps each callback near the end of its block.  The
            # ring buffer contains only complete VAD windows, so subtract the
            # still-pending tail and then map the VAD's absolute sample endpoint
            # onto the monotonic clock.  Unlike the previous `now` timestamp,
            # this excludes the silence which made VAD decide the segment ended.
            history_end_ns = chunk.captured_monotonic_ns - self._samples_to_ns(
                self._pending_size
            )
            samples_after_segment = max(0, self._history.end_sample - segment_end)
            ended_ns = history_end_ns - self._samples_to_ns(samples_after_segment)
            duration_ns = round(samples.size * 1_000_000_000 / self._sample_rate)
            utterances.append(
                Utterance(
                    samples=samples,
                    sample_rate=self._sample_rate,
                    started_monotonic_ns=ended_ns - duration_ns,
                    ended_monotonic_ns=ended_ns,
                    ready_monotonic_ns=ready_ns,
                )
            )
        return utterances

    def reset(self) -> None:
        self._pending_size = 0
        self._history.clear()
        reset = getattr(self._vad, "reset", None)
        if reset is not None:
            reset()
            return
        clear = getattr(self._vad, "clear", None)
        if clear is not None:
            clear()
            return
        while not self._vad.empty():
            self._vad.pop()

    @property
    def speech_active(self) -> bool:
        """Expose the detector's early speech edge for full-duplex barge-in."""
        detected = getattr(self._vad, "is_detected", False)
        return bool(detected() if callable(detected) else detected)

    def _accept_window(self, window: np.ndarray) -> None:
        self._history.append(window)
        self._vad.accept_waveform(window)

    def _samples_to_ns(self, sample_count: int) -> int:
        return round(sample_count * 1_000_000_000 / self._sample_rate)
