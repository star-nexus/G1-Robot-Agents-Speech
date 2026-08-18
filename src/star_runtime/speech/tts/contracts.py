"""Streaming TTS domain requests, audio chunks, and engine port."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Iterator, Protocol

@dataclass(frozen=True)
class TtsSynthesisRequest:
    request_id: str
    text: str
    language: str
    voice: str
    instructions: str
    final_sentence: bool
    finalize_only: bool = False


@dataclass(frozen=True)
class TtsAudioChunk:
    pcm16_mono: bytes
    sample_rate: int


class StreamingTtsEngine(Protocol):
    def stream(
        self,
        request: TtsSynthesisRequest,
        cancel: threading.Event,
    ) -> Iterator[TtsAudioChunk]: ...

    def close(self) -> None: ...

