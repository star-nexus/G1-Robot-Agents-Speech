"""vLLM-Omni streaming WebSocket provider adapter."""

from __future__ import annotations

import json
import socket
import threading
from pathlib import Path
from typing import Any, Iterator

from ..config import TtsConfig
from .contracts import TtsAudioChunk, TtsSynthesisRequest


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

