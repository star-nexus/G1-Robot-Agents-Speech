"""Stable contracts and events for the STAR speech pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np

from ..core.events import SpeechEvent
from ..transports.contracts import EventSink, SpeechTransport


@dataclass(frozen=True)
class AudioChunk:
    samples: np.ndarray
    sample_rate: int
    captured_monotonic_ns: int


@dataclass(frozen=True)
class Utterance:
    samples: np.ndarray
    sample_rate: int
    started_monotonic_ns: int
    ended_monotonic_ns: int
    # Time at which VAD emitted the completed utterance.  This is deliberately
    # separate from ended_monotonic_ns, which tracks the estimated acoustic end
    # of speech and therefore excludes the configured trailing-silence wait.
    ready_monotonic_ns: int | None = None

    @property
    def duration_ms(self) -> int:
        return round(self.samples.size * 1000 / self.sample_rate)

    @property
    def vad_ready_monotonic_ns(self) -> int:
        """Return VAD-ready time while remaining compatible with older adapters."""
        return self.ready_monotonic_ns or self.ended_monotonic_ns


@dataclass(frozen=True)
class RecognitionResult:
    text: str
    language: str
    inference_ms: float
    engine: str = "sensevoice"


@dataclass(frozen=True)
class PipelineMetricsSnapshot:
    audio_chunks_received: int
    audio_chunks_dropped: int
    audio_reconnections: int
    audio_reconnect_failures: int
    utterances_detected: int
    utterances_dropped: int
    recognitions_succeeded: int
    recognitions_empty: int
    recognition_errors: int
    playback_chunks_suppressed: int
    publish_enqueued: int
    audio_queue_size: int
    utterance_queue_size: int


class AudioSource(Protocol):
    dropped_chunks: int

    def start(self) -> None: ...

    def read(self, timeout: float = 0.2) -> AudioChunk | None: ...

    def close(self) -> None: ...


class Segmenter(Protocol):
    def accept(self, chunk: AudioChunk) -> list[Utterance]: ...

    def reset(self) -> None: ...


class AsrEngine(Protocol):
    def load(self) -> None: ...

    def transcribe(self, utterance: Utterance) -> RecognitionResult: ...

    def close(self) -> None: ...


SpeechCallback = Callable[[SpeechEvent], None]
