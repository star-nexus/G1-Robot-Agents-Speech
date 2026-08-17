"""Streaming Qwen3-TTS client, sentence scheduler, and barge-in controller."""

from __future__ import annotations

import json
import logging
import queue
import re
import socket
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Protocol

from .audio_output import AlsaOutputPlayer
from .config import TtsConfig

logger = logging.getLogger(__name__)

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


@dataclass(frozen=True)
class TtsTextChunk:
    request_id: str
    sequence: int
    text: str
    is_final: bool = False
    interrupt: bool = False
    language: str = ""
    voice: str = ""
    instructions: str = ""
    created_unix_ns: int = 0
    source: str = "agent"


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

    def feed(self, chunk: TtsTextChunk) -> list[TtsSynthesisRequest]:
        if chunk.interrupt:
            self.reset()
            return []
        if self._request_id != chunk.request_id:
            self.reset()
            self._request_id = chunk.request_id
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
        )


class VllmOmniStreamingTts:
    """Client for vLLM-Omni's incremental WebSocket speech endpoint."""

    def __init__(self, settings: TtsConfig, *, websocket_module: Any | None = None) -> None:
        self._settings = settings
        self._websocket_module = websocket_module
        self._socket_lock = threading.Lock()
        self._active_socket: Any | None = None

    def stream(
        self,
        request: TtsSynthesisRequest,
        cancel: threading.Event,
    ) -> Iterator[TtsAudioChunk]:
        websocket = self._websocket_module
        if websocket is None:
            try:
                import websocket
            except ImportError as exc:
                raise RuntimeError(
                    "streaming TTS requires websocket-client; install g1-speech[tts]"
                ) from exc
        with self._socket_lock:
            ws = self._active_socket
            if ws is None:
                ws = websocket.create_connection(
                    self._settings.websocket_url,
                    timeout=self._settings.connect_timeout_seconds,
                    enable_multithread=True,
                )
                ws.settimeout(self._settings.receive_timeout_seconds)
                self._active_socket = ws
        sample_rate = 24000
        reusable = False
        try:
            ws.send(json.dumps(self._session_config(request)))
            ws.send(json.dumps({"type": "input.text", "text": request.text}))
            ws.send(json.dumps({"type": "input.done"}))
            while not cancel.is_set():
                try:
                    message = ws.recv()
                except (socket.timeout, TimeoutError):
                    continue
                except Exception as exc:  # websocket timeout has a package-local type
                    if exc.__class__.__name__ == "WebSocketTimeoutException":
                        continue
                    if cancel.is_set():
                        break
                    raise
                if isinstance(message, (bytes, bytearray)):
                    if message:
                        yield TtsAudioChunk(bytes(message), sample_rate)
                    continue
                event = json.loads(message)
                event_type = event.get("type")
                if event_type == "audio.start":
                    sample_rate = int(event.get("sample_rate", sample_rate))
                elif event_type == "error":
                    raise RuntimeError(f"vLLM-Omni TTS error: {event.get('message')}")
                elif event_type == "audio.done" and event.get("error"):
                    raise RuntimeError("vLLM-Omni reported an audio generation error")
                elif event_type == "session.done":
                    reusable = True
                    break
        finally:
            # input.done is a flush, not a close. Reusing the WebSocket removes
            # one TCP/WebSocket handshake from every sentence after the first.
            if cancel.is_set() or not reusable:
                with self._socket_lock:
                    if self._active_socket is ws:
                        self._active_socket = None
                try:
                    ws.close()
                except Exception:  # noqa: BLE001
                    pass

    def cancel(self) -> None:
        with self._socket_lock:
            ws, self._active_socket = self._active_socket, None
        if ws is not None:
            try:
                ws.close()
            except Exception:  # noqa: BLE001
                pass

    def close(self) -> None:
        self.cancel()

    def _session_config(self, request: TtsSynthesisRequest) -> dict[str, Any]:
        config: dict[str, Any] = {
            "type": "session.config",
            "model": self._settings.model,
            "task_type": self._settings.task_type,
            "voice": request.voice,
            "language": request.language,
            "instructions": request.instructions,
            "response_format": "pcm",
            "stream_audio": True,
            "max_new_tokens": self._settings.max_new_tokens,
        }
        if self._settings.initial_codec_chunk_frames is not None:
            config["initial_codec_chunk_frames"] = (
                self._settings.initial_codec_chunk_frames
            )
        if self._settings.task_type == "Base":
            assert self._settings.reference_audio is not None
            reference = self._settings.reference_audio
            if "://" not in reference:
                reference = Path(reference).resolve().as_uri()
            config["ref_audio"] = reference
            config["ref_text"] = self._settings.reference_text
        return config


class TtsController:
    """Bounded synthesis/playback worker with VAD-triggered hard interruption."""

    def __init__(
        self,
        settings: TtsConfig,
        *,
        engine: StreamingTtsEngine,
        player: AlsaOutputPlayer,
        playback_state: Callable[[bool, str], None],
    ) -> None:
        self._settings = settings
        self._engine = engine
        self._player = player
        self._playback_state = playback_state
        self._assembler = SentenceAssembler(settings)
        self._queue: queue.Queue[TtsSynthesisRequest] = queue.Queue(
            maxsize=settings.request_queue_capacity
        )
        self._stop = threading.Event()
        self._cancel = threading.Event()
        self._thread: threading.Thread | None = None
        self._state_lock = threading.Lock()
        self._active_request_id: str | None = None
        self.sentences_synthesized = 0
        self.synthesis_errors = 0
        self.interruptions = 0
        self.first_audio_latency_ms: float | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._player.start()
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="speech-tts",
            daemon=True,
        )
        self._thread.start()

    def accept(self, chunk: TtsTextChunk) -> None:
        if chunk.interrupt:
            self.interrupt(reason="agent")
            return
        for request in self._assembler.feed(chunk):
            try:
                self._queue.put_nowait(request)
            except queue.Full:
                logger.error("TTS request queue full; interrupting stale playback")
                self.interrupt(reason="queue-full")
                self._queue.put_nowait(request)

    def interrupt(self, *, reason: str = "vad") -> None:
        # An interrupt also invalidates text that has not yet formed a synthesis
        # request.  This matters when the agent streams a partial sentence and
        # the user barges in before punctuation arrives.
        self._assembler.reset()
        with self._state_lock:
            active = self._active_request_id
        if active is None and self._queue.empty():
            return
        self._cancel.set()
        cancel_engine = getattr(self._engine, "cancel", None)
        if cancel_engine is not None:
            cancel_engine()
        self._player.abort()
        self._drain_queue()
        with self._state_lock:
            request_id, self._active_request_id = self._active_request_id, None
        if request_id is not None:
            self._playback_state(False, request_id)
            self.interruptions += 1
            logger.info("TTS interrupted: request_id=%s reason=%s", request_id, reason)

    def stop(self) -> None:
        self._stop.set()
        self.interrupt(reason="shutdown")
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            if self._thread.is_alive():
                logger.warning("TTS worker did not stop within 3s")
            self._thread = None

    def close(self) -> None:
        self.stop()
        self._engine.close()
        self._player.close()

    def metrics(self) -> dict[str, Any]:
        with self._state_lock:
            active_request_id = self._active_request_id
        payload = {
            "tts_enabled": True,
            "tts_queue_size": self._queue.qsize(),
            "tts_sentences_synthesized": self.sentences_synthesized,
            "tts_synthesis_errors": self.synthesis_errors,
            "tts_interruptions": self.interruptions,
            "tts_first_audio_latency_ms": self.first_audio_latency_ms,
            "tts_playback_active": active_request_id is not None,
            "tts_active_request_id": active_request_id,
        }
        payload.update(self._player.metrics())
        return payload

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                request = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue
            self._cancel.clear()
            with self._state_lock:
                publish_active = self._active_request_id != request.request_id
                self._active_request_id = request.request_id
            if publish_active:
                self._playback_state(True, request.request_id)
            started_ns = time.monotonic_ns()
            first_audio = True
            output_started = False
            try:
                if request.finalize_only:
                    if not self._cancel.is_set():
                        self._player.wait_until_idle(cancel=self._cancel)
                else:
                    for audio in self._engine.stream(request, self._cancel):
                        if self._cancel.is_set():
                            break
                        if first_audio:
                            self.first_audio_latency_ms = (
                                time.monotonic_ns() - started_ns
                            ) / 1_000_000
                            first_audio = False
                        if not self._player.enqueue(
                            audio.pcm16_mono,
                            audio.sample_rate,
                            cancel=self._cancel,
                        ):
                            break
                        output_started = True
                    if not self._cancel.is_set():
                        self.sentences_synthesized += 1
                        # Only the final sentence closes the response timeline. All
                        # preceding sentences remain buffered while generation runs
                        # ahead independently of real-time ALSA playback.
                        if request.final_sentence and output_started:
                            self._player.wait_until_idle(cancel=self._cancel)
            except Exception:  # noqa: BLE001
                self.synthesis_errors += 1
                logger.exception(
                    "TTS synthesis failed: request_id=%s text=%r",
                    request.request_id,
                    request.text,
                )
            finally:
                if request.final_sentence or self._cancel.is_set():
                    publish_inactive = False
                    with self._state_lock:
                        if self._active_request_id == request.request_id:
                            self._active_request_id = None
                            publish_inactive = True
                    # interrupt() clears the active ID and publishes the edge
                    # synchronously, so the worker must not publish it twice.
                    if publish_inactive:
                        self._playback_state(False, request.request_id)
                        logger.info(
                            "TTS response playback completed: request_id=%s",
                            request.request_id,
                        )

    def _drain_queue(self) -> None:
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                return
