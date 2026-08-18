"""Bounded TTS synthesis/playback lifecycle and barge-in control."""

from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Any, Callable

from ...core.events import TtsTextChunk
from ..audio.output import AlsaOutputPlayer
from ..config import TtsConfig
from .contracts import StreamingTtsEngine, TtsSynthesisRequest
from .scheduler import SentenceAssembler

logger = logging.getLogger(__name__)


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

