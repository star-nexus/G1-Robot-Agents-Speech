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
_STRONG_BOUNDARY = re.compile(r".*?[。！？!?；;\n]+", re.DOTALL)
_SOFT_BOUNDARY = re.compile(r".*?[,，:：]+", re.DOTALL)


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

    def feed(self, chunk: TtsTextChunk) -> list[TtsSynthesisRequest]:
        if chunk.interrupt:
            self.reset()
            return []
        if self._request_id != chunk.request_id:
            self.reset()
            self._request_id = chunk.request_id
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
        sentences = self._drain_boundaries()
        if chunk.is_final and self._buffer.strip():
            sentences.append(self._make_request(self._buffer.strip(), True))
            self._buffer = ""
        elif chunk.is_final and sentences:
            last = sentences[-1]
            sentences[-1] = TtsSynthesisRequest(
                **{**vars(last), "final_sentence": True}
            )
        return sentences

    def reset(self) -> None:
        self._request_id = None
        self._buffer = ""
        self._instructions = self._settings.instructions
        self._language = self._settings.language
        self._voice = self._settings.voice
        self._last_sequence = -1

    def _consume_style_tokens(self, text: str) -> str:
        def replace(match: re.Match[str]) -> str:
            instruction = self._settings.style_tokens.get(match.group(1).lower())
            if instruction is None:
                return match.group(0)
            self._instructions = instruction
            return ""

        return _STYLE_TOKEN.sub(replace, text)

    def _drain_boundaries(self) -> list[TtsSynthesisRequest]:
        result: list[TtsSynthesisRequest] = []
        while self._buffer:
            match = _STRONG_BOUNDARY.match(self._buffer)
            if match is None and len(self._buffer) >= 12:
                match = _SOFT_BOUNDARY.match(self._buffer)
            if match is None and len(self._buffer) >= self._settings.max_buffer_characters:
                split = self._settings.max_buffer_characters
                result.append(self._make_request(self._buffer[:split].strip(), False))
                self._buffer = self._buffer[split:]
                continue
            if match is None:
                break
            text = match.group(0).strip()
            self._buffer = self._buffer[match.end() :]
            if text:
                result.append(self._make_request(text, False))
        return result

    def _make_request(self, text: str, final: bool) -> TtsSynthesisRequest:
        assert self._request_id is not None
        return TtsSynthesisRequest(
            request_id=self._request_id,
            text=text,
            language=self._language,
            voice=self._voice,
            instructions=self._instructions,
            final_sentence=final,
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
        payload = {
            "tts_enabled": True,
            "tts_queue_size": self._queue.qsize(),
            "tts_sentences_synthesized": self.sentences_synthesized,
            "tts_synthesis_errors": self.synthesis_errors,
            "tts_interruptions": self.interruptions,
            "tts_first_audio_latency_ms": self.first_audio_latency_ms,
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
                self._active_request_id = request.request_id
            self._playback_state(True, request.request_id)
            started_ns = time.monotonic_ns()
            first_audio = True
            output_started = False
            try:
                for audio in self._engine.stream(request, self._cancel):
                    if self._cancel.is_set():
                        break
                    if not output_started:
                        self._player.begin(audio.sample_rate)
                        output_started = True
                    if first_audio:
                        self.first_audio_latency_ms = (
                            time.monotonic_ns() - started_ns
                        ) / 1_000_000
                        first_audio = False
                    if not self._player.write(audio.pcm16_mono):
                        break
                if output_started and not self._cancel.is_set():
                    self._player.finish()
                if not self._cancel.is_set():
                    self.sentences_synthesized += 1
            except Exception:  # noqa: BLE001
                self.synthesis_errors += 1
                logger.exception(
                    "TTS synthesis failed: request_id=%s text=%r",
                    request.request_id,
                    request.text,
                )
            finally:
                if request.final_sentence or self._cancel.is_set() or self._queue.empty():
                    publish_inactive = False
                    with self._state_lock:
                        if self._active_request_id == request.request_id:
                            self._active_request_id = None
                            publish_inactive = True
                    # interrupt() clears the active ID and publishes the edge
                    # synchronously, so the worker must not publish it twice.
                    if publish_inactive:
                        self._playback_state(False, request.request_id)

    def _drain_queue(self) -> None:
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                return
