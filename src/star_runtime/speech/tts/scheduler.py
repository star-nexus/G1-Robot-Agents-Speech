"""Incremental LLM-text to bounded speech-request scheduling."""

from __future__ import annotations

import re

from ...core.events import TtsTextChunk
from ..config import TtsConfig
from .contracts import TtsSynthesisRequest

_STYLE_TOKEN = re.compile(r"\[([A-Za-z0-9_-]+)\]")
_STRONG_BOUNDARY = re.compile(r"[。！？!?；;\n]+|\.+(?!\d)")
_SOFT_BOUNDARY = re.compile(r"[,，:：、—–]+")
_WRAP_BOUNDARY = re.compile(r"\s+")
_WORD = re.compile(r"[^\W_]+(?:['’\-][^\W_]+)*", re.UNICODE)
_SYLLABIC_CHAR = re.compile(
    r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff"
    r"\u3040-\u30ff\uac00-\ud7af]"
)
_TRAILING_CLOSERS = frozenset("\"'”’）)]】》」』")


class SentenceAssembler:
    """Turn LLM token fragments into bounded, style-aware TTS sentences."""

    def __init__(self, settings: TtsConfig) -> None:
        self._settings = settings
        self._request_id: str | None = None
        self._buffer = ""
        self._instructions = settings.instructions
        self._language = settings.language
        self._voice = settings.voice
        self._last_sequence = -1
        self._emitted_for_request = False
        self._finalized = False
        self._session_id = ""
        self._turn_id = ""
        self._epoch = 0

    def feed(self, chunk: TtsTextChunk) -> list[TtsSynthesisRequest]:
        if chunk.interrupt:
            self.reset()
            return []
        if self._request_id != chunk.request_id:
            self.reset()
            self._request_id = chunk.request_id
            self._session_id = chunk.session_id
            self._turn_id = chunk.turn_id
            self._epoch = chunk.epoch
        if self._finalized:
            return []
        if chunk.sequence <= self._last_sequence:
            return []
        self._last_sequence = chunk.sequence
        if chunk.language:
            self._language = chunk.language
        if chunk.voice:
            self._voice = chunk.voice
        if chunk.instructions:
            self._instructions = chunk.instructions

        text = self._consume_style_tokens(chunk.text)
        self._buffer += text
        # A complete response already paid the Agent generation latency. Preserve
        # its clauses as one prosodic unit unless it contains a true sentence end.
        # Adaptive comma/wrap flushing is reserved for still-streaming responses.
        sentences = self._drain_boundaries(allow_adaptive=not chunk.is_final)
        if chunk.is_final and self._buffer.strip():
            sentences.append(self._make_request(self._buffer.strip(), True))
            self._buffer = ""
        elif chunk.is_final and sentences:
            last = sentences[-1]
            sentences[-1] = TtsSynthesisRequest(
                **{**vars(last), "final_sentence": True}
            )
        elif chunk.is_final and self._emitted_for_request:
            # Streaming Agents commonly emit terminal punctuation as an ordinary
            # delta and then close the request with an empty final marker. The
            # punctuation may already have produced a synthesis request, but the
            # final marker must still drain the shared PCM timeline and publish
            # playback inactive. It is a control event, not an empty TTS call.
            sentences.append(self._make_request("", True, finalize_only=True))
        if sentences:
            self._emitted_for_request = True
        if chunk.is_final:
            self._finalized = True
        return sentences

    def reset(self) -> None:
        self._request_id = None
        self._buffer = ""
        self._instructions = self._settings.instructions
        self._language = self._settings.language
        self._voice = self._settings.voice
        self._last_sequence = -1
        self._emitted_for_request = False
        self._finalized = False
        self._session_id = ""
        self._turn_id = ""
        self._epoch = 0

    def _consume_style_tokens(self, text: str) -> str:
        def replace(match: re.Match[str]) -> str:
            instruction = self._settings.style_tokens.get(match.group(1).lower())
            if instruction is None:
                return match.group(0)
            self._instructions = instruction
            return ""

        return _STYLE_TOKEN.sub(replace, text)

    def _drain_boundaries(
        self,
        *,
        allow_adaptive: bool,
    ) -> list[TtsSynthesisRequest]:
        result: list[TtsSynthesisRequest] = []
        while self._buffer:
            split = self._strong_boundary()
            if split is None and allow_adaptive:
                split = self._soft_boundary()
            if split is None and (
                len(self._buffer) >= self._settings.max_buffer_characters
                or (
                    allow_adaptive
                    and self._speech_units(self._buffer)
                    >= self._settings.max_chunk_speech_units
                )
            ):
                split = self._wrap_boundary()
            if split is None:
                break
            text = self._buffer[:split].strip()
            self._buffer = self._buffer[split:]
            if text:
                result.append(self._make_request(text, False))
        return result

    def _strong_boundary(self) -> int | None:
        match = _STRONG_BOUNDARY.search(self._buffer)
        if match is None:
            return None
        end = match.end()
        while end < len(self._buffer) and self._buffer[end] in _TRAILING_CLOSERS:
            end += 1
        return end

    def _soft_boundary(self) -> int | None:
        total_units = self._speech_units(self._buffer)
        if total_units < self._settings.preferred_chunk_speech_units:
            return None
        candidates: list[tuple[float, int]] = []
        for match in _SOFT_BOUNDARY.finditer(self._buffer):
            units = self._speech_units(self._buffer[: match.end()])
            if not (
                self._settings.min_chunk_speech_units
                <= units
                <= self._settings.max_chunk_speech_units
            ):
                continue
            distance = abs(units - self._settings.preferred_chunk_speech_units)
            candidates.append((distance, match.end()))
        if not candidates:
            return None
        # On an equal score, keep more context in the emitted clause.
        return min(candidates, key=lambda item: (item[0], -item[1]))[1]

    def _wrap_boundary(self) -> int:
        limit = min(len(self._buffer), self._settings.max_buffer_characters)
        candidates: list[tuple[float, int]] = []
        for pattern in (_SOFT_BOUNDARY, _WRAP_BOUNDARY):
            for match in pattern.finditer(self._buffer[:limit]):
                units = self._speech_units(self._buffer[: match.end()])
                if not (
                    self._settings.min_chunk_speech_units
                    <= units
                    <= self._settings.max_chunk_speech_units
                ):
                    continue
                distance = abs(units - self._settings.preferred_chunk_speech_units)
                candidates.append((distance, match.end()))
        if candidates:
            return min(candidates, key=lambda item: (item[0], -item[1]))[1]

        # CJK text may contain no whitespace at all. Split at the preferred
        # estimated spoken length only after the maximum threshold forced a flush.
        for index in range(1, limit + 1):
            if (
                self._speech_units(self._buffer[:index])
                >= self._settings.preferred_chunk_speech_units
            ):
                return index
        return limit

    @staticmethod
    def _speech_units(text: str) -> float:
        """Estimate spoken length across CJK characters and Unicode words."""
        syllabic = len(_SYLLABIC_CHAR.findall(text))
        without_syllabic = _SYLLABIC_CHAR.sub(" ", text)
        words = len(_WORD.findall(without_syllabic))
        return syllabic + words * 1.5

    def _make_request(
        self,
        text: str,
        final: bool,
        *,
        finalize_only: bool = False,
    ) -> TtsSynthesisRequest:
        assert self._request_id is not None
        return TtsSynthesisRequest(
            request_id=self._request_id,
            text=text,
            language=self._language,
            voice=self._voice,
            instructions=self._instructions,
            final_sentence=final,
            finalize_only=finalize_only,
            session_id=self._session_id,
            turn_id=self._turn_id,
            epoch=self._epoch,
        )
