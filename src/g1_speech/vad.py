"""Silero VAD segmentation through the already-used sherpa-onnx runtime."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from .contracts import AudioChunk, Utterance


class SileroVadSegmenter:
    def __init__(
        self,
        *,
        model: str,
        sample_rate: int = 16000,
        threshold: float = 0.5,
        speech_pre_roll_seconds: float = 0.3,
        min_silence_seconds: float = 0.35,
        min_speech_seconds: float = 0.25,
        max_speech_seconds: float = 15.0,
        buffer_seconds: float = 30.0,
        sherpa_module: Any | None = None,
    ) -> None:
        model_path = Path(model).expanduser().resolve()
        if not model_path.is_file():
            raise FileNotFoundError(f"Silero VAD model not found: {model_path}")
        if sample_rate != 16000:
            raise ValueError("the current Silero VAD model requires 16000 Hz audio")
        if sherpa_module is None:
            import sherpa_onnx

            sherpa_module = sherpa_onnx
        config = sherpa_module.VadModelConfig()
        config.silero_vad.model = str(model_path)
        config.silero_vad.threshold = threshold
        config.silero_vad.min_silence_duration = min_silence_seconds
        config.silero_vad.min_speech_duration = min_speech_seconds
        config.silero_vad.max_speech_duration = max_speech_seconds
        config.sample_rate = sample_rate
        self._vad = sherpa_module.VoiceActivityDetector(
            config, buffer_size_in_seconds=buffer_seconds
        )
        self._sample_rate = sample_rate
        self._window_size = config.silero_vad.window_size
        self._pre_roll_samples = round(speech_pre_roll_seconds * sample_rate)
        self._history_limit_samples = round(buffer_seconds * sample_rate)
        self._pending = np.empty(0, dtype=np.float32)
        self._history = np.empty(0, dtype=np.float32)
        self._history_start_sample = 0
        self._accepted_samples = 0

    def accept(self, chunk: AudioChunk) -> list[Utterance]:
        if chunk.sample_rate != self._sample_rate:
            raise ValueError("VAD received an unexpected sample rate")
        self._pending = np.concatenate(
            (self._pending, np.asarray(chunk.samples, dtype=np.float32).reshape(-1))
        )
        while self._pending.size >= self._window_size:
            window = self._pending[: self._window_size]
            self._pending = self._pending[self._window_size :]
            self._append_history(window)
            self._vad.accept_waveform(window)
            self._accepted_samples += window.size

        utterances: list[Utterance] = []
        now = time.monotonic_ns()
        while not self._vad.empty():
            segment = self._vad.front
            vad_samples = np.asarray(segment.samples, dtype=np.float32)
            segment_start = int(segment.start)
            segment_end = segment_start + vad_samples.size
            wanted_start = max(
                self._history_start_sample,
                segment_start - self._pre_roll_samples,
            )
            local_start = wanted_start - self._history_start_sample
            local_end = segment_end - self._history_start_sample
            if 0 <= local_start < local_end <= self._history.size:
                samples = self._history[local_start:local_end].copy()
            else:
                samples = vad_samples.copy()
            self._vad.pop()
            duration_ns = round(samples.size * 1_000_000_000 / self._sample_rate)
            utterances.append(
                Utterance(
                    samples=samples,
                    sample_rate=self._sample_rate,
                    started_monotonic_ns=now - duration_ns,
                    ended_monotonic_ns=now,
                )
            )
        return utterances

    def reset(self) -> None:
        self._pending = np.empty(0, dtype=np.float32)
        self._history = np.empty(0, dtype=np.float32)
        self._history_start_sample = 0
        self._accepted_samples = 0
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

    def _append_history(self, window: np.ndarray) -> None:
        self._history = np.concatenate((self._history, window))
        overflow = self._history.size - self._history_limit_samples
        if overflow > 0:
            self._history = self._history[overflow:]
            self._history_start_sample += overflow
