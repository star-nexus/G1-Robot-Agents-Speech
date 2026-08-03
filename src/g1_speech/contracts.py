"""Stable, application-independent contracts for the speech pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol

import numpy as np


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

    @property
    def duration_ms(self) -> int:
        return round(self.samples.size * 1000 / self.sample_rate)


@dataclass(frozen=True)
class RecognitionResult:
    text: str
    language: str
    inference_ms: float
    engine: str = "sensevoice"


@dataclass(frozen=True)
class SpeechEvent:
    event_id: str
    session_id: str
    sequence: int
    created_unix_ns: int
    source: str
    text: str
    language: str
    audio_duration_ms: int
    inference_ms: float
    engine: str = "sensevoice"
    is_final: bool = True


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


class EventSink(Protocol):
    def start(self) -> None: ...

    def publish(self, event: SpeechEvent) -> bool: ...

    def close(self) -> None: ...


class SpeechTransport(Protocol):
    """Bidirectional speech transport used by the service composition root."""

    sink: EventSink

    def start(self) -> None: ...

    def stop(self) -> None: ...

    def close(self) -> None: ...

    def metrics(self) -> dict[str, Any]: ...


SpeechCallback = Callable[[SpeechEvent], None]
