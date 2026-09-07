"""Bounded TTS synthesis/playback lifecycle and barge-in control."""

from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Any, Callable

from ...core.control import ControlStamp, RuntimeControlPlane
from ...core.events import PlaybackState, TtsTextChunk
from ...core.timing import RuntimeTimingAudit
from ..audio.output import AlsaOutputPlayer
from ..config import TtsConfig
from .contracts import StreamingTtsEngine, TtsSynthesisRequest
from .scheduler import SentenceAssembler

logger = logging.getLogger(__name__)


class TtsController:
    """Bounded synthesis/playback worker with VAD-triggered interruption."""

    def __init__(
        self,
        settings: TtsConfig,
        *,
        engine: StreamingTtsEngine,
        player: AlsaOutputPlayer,
        playback_state: Callable[..., None],
        control_plane: RuntimeControlPlane | None = None,
        timing_audit: RuntimeTimingAudit | None = None,
    ) -> None:
        self._settings = settings
        self._engine = engine
        self._player = player
        self._playback_state = playback_state
        self._control = control_plane
        self._timing = timing_audit
        self._assembler = SentenceAssembler(settings)
        self._queue: queue.Queue[TtsSynthesisRequest] = queue.Queue(
            maxsize=settings.request_queue_capacity
        )
        self._stop = threading.Event()
        self._cancel = threading.Event()
        self._thread: threading.Thread | None = None
        self._state_lock = threading.Lock()
        self._active_request_id: str | None = None
        self._active_stamp: ControlStamp | None = None
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
        if not self._is_current(chunk.control_stamp, "tts-text"):
            return
        if chunk.interrupt:
            self.interrupt(reason="agent")
            return
        for request in self._assembler.feed(chunk):
            if self._timing is not None and not request.finalize_only:
                self._timing.mark(
                    request.control_stamp,
                    "tts_segment_ready",
                    at_ns=time.monotonic_ns(),
                    characters=len(request.text),
                    final_sentence=request.final_sentence,
                )
            enqueued_ns = time.monotonic_ns()
            try:
                self._queue.put_nowait(request)
                if self._timing is not None and not request.finalize_only:
                    self._timing.mark(
                        request.control_stamp,
                        "tts_segment_enqueued",
                        at_ns=enqueued_ns,
                    )
            except queue.Full:
                logger.error("TTS request queue full; interrupting stale playback")
                self.interrupt(reason="queue-full")
                enqueued_ns = time.monotonic_ns()
                self._queue.put_nowait(request)
                if self._timing is not None and not request.finalize_only:
                    self._timing.mark(
                        request.control_stamp,
                        "tts_segment_enqueued",
                        at_ns=enqueued_ns,
                    )

    def interrupt(self, *, reason: str = "vad") -> bool:
        """Preempt visible output, then actively cancel provider generation."""

        interrupted = self.preempt_output(reason=reason)
        if interrupted:
            self.cancel_provider()
        return interrupted

    def preempt_output(self, *, reason: str = "vad") -> bool:
        """Synchronously invalidate local TTS/PCM work without provider I/O."""

        # An interrupt also invalidates text that has not yet formed a synthesis
        # request.  This matters when the agent streams a partial sentence and
        # the user barges in before punctuation arrives.
        self._assembler.reset()
        with self._state_lock:
            active = self._active_request_id
            active_stamp = self._active_stamp
        if active is None and self._queue.empty():
            return False
        self._cancel.set()
        if self._timing is not None and active_stamp is not None:
            self._timing.mark(
                active_stamp,
                "tts_interrupt",
                at_ns=time.monotonic_ns(),
                reason=reason,
            )
        # Flush and advance the software PCM timeline before provider teardown.
        # A WebSocket/provider cancel can block, while playback invalidation must
        # happen immediately after the authoritative epoch changes.
        self._player.abort()
        if self._timing is not None and active_stamp is not None:
            self._timing.mark(
                active_stamp,
                "playback_abort",
                at_ns=time.monotonic_ns(),
            )
        self._drain_queue()
        with self._state_lock:
            request_id, self._active_request_id = self._active_request_id, None
            stamp, self._active_stamp = self._active_stamp, None
        if request_id is not None:
            self._emit_playback(False, request_id, stamp)
            self.interruptions += 1
            logger.info("TTS interrupted: request_id=%s reason=%s", request_id, reason)
        return True

    def cancel_provider(self) -> None:
        """Best-effort provider cancellation after local output is already safe."""

        cancel_engine = getattr(self._engine, "cancel", None)
        if cancel_engine is not None:
            cancel_engine()

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
            if not self._is_current(request.control_stamp, "tts-queue"):
                continue
            with self._state_lock:
                publish_active = self._active_request_id != request.request_id
                self._active_request_id = request.request_id
                self._active_stamp = request.control_stamp
            if publish_active:
                self._emit_playback(True, request.request_id, request.control_stamp)
            started_ns = time.monotonic_ns()
            if self._timing is not None and not request.finalize_only:
                self._timing.mark(
                    request.control_stamp,
                    "tts_synthesis_start",
                    at_ns=started_ns,
                )
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
                        if not self._is_current(request.control_stamp, "tts-audio"):
                            self._cancel.set()
                            break
                        if first_audio:
                            first_pcm_ns = time.monotonic_ns()
                            self.first_audio_latency_ms = (
                                first_pcm_ns - started_ns
                            ) / 1_000_000
                            if self._timing is not None:
                                self._timing.mark(
                                    request.control_stamp,
                                    "tts_first_pcm",
                                    at_ns=first_pcm_ns,
                                    bytes=len(audio.pcm16_mono),
                                    sample_rate=audio.sample_rate,
                                )
                            first_audio = False
                            note_pcm_ready = getattr(
                                self._player,
                                "note_pcm_ready",
                                None,
                            )
                            if note_pcm_ready is not None:
                                note_pcm_ready()
                        set_control_stamp = getattr(
                            self._player,
                            "set_control_stamp",
                            None,
                        )
                        if set_control_stamp is not None:
                            set_control_stamp(request.control_stamp)
                        enqueue_started_ns = time.monotonic_ns()
                        if self._timing is not None:
                            self._timing.mark(
                                request.control_stamp,
                                "player_enqueue_begin",
                                at_ns=enqueue_started_ns,
                                bytes=len(audio.pcm16_mono),
                            )
                        if not self._player.enqueue(
                            audio.pcm16_mono,
                            audio.sample_rate,
                            cancel=self._cancel,
                        ):
                            break
                        if self._timing is not None:
                            self._timing.mark(
                                request.control_stamp,
                                "player_enqueue_complete",
                                at_ns=time.monotonic_ns(),
                            )
                        output_started = True
                    if not self._cancel.is_set():
                        self.sentences_synthesized += 1
                        # Only the final sentence closes the response timeline. All
                        # preceding sentences remain buffered while generation runs
                        # ahead independently of real-time ALSA playback.
                        if request.final_sentence and output_started:
                            self._player.wait_until_idle(cancel=self._cancel)
            except Exception:  # noqa: BLE001
                stale = self._control is not None and not self._control.is_current(
                    request.control_stamp
                )
                if self._cancel.is_set() or stale:
                    logger.info(
                        "TTS provider stopped after cancellation: request_id=%s",
                        request.request_id,
                    )
                else:
                    self.synthesis_errors += 1
                    if self._timing is not None:
                        self._timing.mark(
                            request.control_stamp,
                            "tts_error",
                            at_ns=time.monotonic_ns(),
                        )
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
                            self._active_stamp = None
                            publish_inactive = True
                    # interrupt() clears the active ID and publishes the edge
                    # synchronously, so the worker must not publish it twice.
                    if publish_inactive:
                        if self._is_current(
                            request.control_stamp,
                            "playback-complete",
                        ):
                            self._emit_playback(
                                False,
                                request.request_id,
                                request.control_stamp,
                            )
                            if self._control is not None:
                                self._control.complete(request.control_stamp)
                            if self._timing is not None:
                                completed_ns = time.monotonic_ns()
                                self._timing.mark(
                                    request.control_stamp,
                                    "playback_complete",
                                    at_ns=completed_ns,
                                )
                                self._timing.finish_if_complete(
                                    request.control_stamp
                                )
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

    def _is_current(self, stamp: ControlStamp, stage: str) -> bool:
        if self._control is None:
            return True
        if self._control.is_current(stamp):
            return True
        self._control.note_stale(stage, stamp)
        return False

    def _emit_playback(
        self,
        active: bool,
        request_id: str,
        stamp: ControlStamp | None,
    ) -> None:
        stamp = stamp or ControlStamp("", "", 0)
        if active and self._timing is not None:
            self._timing.mark(
                stamp,
                "playback_active",
                at_ns=time.monotonic_ns(),
                request_id=request_id,
            )
        state = PlaybackState(
            request_id=request_id,
            active=active,
            created_unix_ns=time.time_ns(),
            source="g1_speech_tts",
            session_id=stamp.session_id,
            turn_id=stamp.turn_id,
            epoch=stamp.epoch,
        )
        try:
            self._playback_state(state)
        except TypeError:
            # Compatibility with pre-v1 third-party callbacks.
            self._playback_state(active, request_id)
