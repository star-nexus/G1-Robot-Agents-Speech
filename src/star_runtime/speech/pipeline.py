"""Bounded three-stage speech pipeline: capture -> VAD -> ASR -> sink."""

from __future__ import annotations

import logging
import queue
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Callable

from ..core.control import ControlStamp, RuntimeControlPlane
from ..core.timing import RuntimeTimingAudit
from .contracts import (
    AsrEngine,
    AudioSource,
    EventSink,
    PipelineMetricsSnapshot,
    Segmenter,
    SpeechEvent,
    Utterance,
)
from .playback import PlaybackGate

logger = logging.getLogger(__name__)


@dataclass
class _MutableMetrics:
    audio_chunks_received: int = 0
    utterances_detected: int = 0
    utterances_dropped: int = 0
    recognitions_succeeded: int = 0
    recognitions_empty: int = 0
    recognition_errors: int = 0
    playback_chunks_suppressed: int = 0
    publish_enqueued: int = 0


class SpeechPipeline:
    def __init__(
        self,
        *,
        source: AudioSource,
        segmenter: Segmenter,
        engine: AsrEngine,
        sink: EventSink,
        playback_gate: PlaybackGate,
        source_name: str = "g1_speech_mic",
        utterance_queue_capacity: int = 4,
        session_id: str | None = None,
        control_plane: RuntimeControlPlane | None = None,
        timing_audit: RuntimeTimingAudit | None = None,
        on_speech_start: Callable[[int], None] | None = None,
        on_turn_superseded: Callable[[ControlStamp, ControlStamp], None]
        | None = None,
        suppress_during_playback: bool = True,
    ) -> None:
        self._source = source
        self._segmenter = segmenter
        self._engine = engine
        self._sink = sink
        self._gate = playback_gate
        self._source_name = source_name
        self._utterances: queue.Queue[Utterance] = queue.Queue(
            maxsize=utterance_queue_capacity
        )
        self._control = control_plane or RuntimeControlPlane(session_id)
        self._timing = timing_audit
        self._session_id = self._control.session_id
        self._on_speech_start = on_speech_start
        self._on_turn_superseded = on_turn_superseded
        self._suppress_during_playback = suppress_during_playback
        self._sequence = 0
        self._metrics = _MutableMetrics()
        self._metrics_lock = threading.Lock()
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._recognition_ready = threading.Event()
        self._recognition_start_error: Exception | None = None
        self._prepared = False
        self._started = False

    def prepare(self) -> None:
        """Load heavyweight inference resources without starting capture."""
        if self._prepared:
            return
        self._engine.load()
        self._prepared = True

    def start(self) -> None:
        if self._started:
            return
        self._sink.start()
        try:
            self.prepare()
            self._stop.clear()
            self._recognition_ready.clear()
            self._recognition_start_error = None
            recognition_thread = threading.Thread(
                target=self._recognition_loop,
                name="speech-asr",
                daemon=True,
            )
            self._threads = [recognition_thread]
            recognition_thread.start()
            self._recognition_ready.wait()
            if self._recognition_start_error is not None:
                raise RuntimeError("ASR recognition thread failed to start") from (
                    self._recognition_start_error
                )
            self._source.start()
        except Exception:
            self._stop.set()
            self._source.close()
            for thread in self._threads:
                thread.join(timeout=3.0)
            self._threads.clear()
            self._sink.close()
            raise
        segment_thread = threading.Thread(
            target=self._segment_loop,
            name="speech-vad",
            daemon=True,
        )
        self._threads.insert(0, segment_thread)
        segment_thread.start()
        self._started = True
        logger.info("Speech pipeline started: session_id=%s", self._session_id)

    def stop(self, *, join_timeout: float = 3.0) -> None:
        """Stop live I/O while keeping heavyweight models resident."""

        if not self._started:
            return
        self._stop.set()
        self._source.close()
        for thread in self._threads:
            thread.join(timeout=join_timeout)
            if thread.is_alive():
                logger.warning("Thread %s did not stop within %.1fs", thread.name, join_timeout)
        self._threads.clear()
        self._segmenter.reset()
        self._sink.close()
        self._started = False
        logger.info("Speech pipeline stopped")

    def close(self, *, join_timeout: float = 3.0) -> None:
        """Release live I/O and all prepared inference resources."""

        self.stop(join_timeout=join_timeout)
        if self._prepared:
            close_engine = getattr(self._engine, "close", None)
            if close_engine is not None:
                close_engine()
            self._prepared = False

    def wait(self, timeout: float | None = None) -> bool:
        return self._stop.wait(timeout)

    def metrics(self) -> PipelineMetricsSnapshot:
        with self._metrics_lock:
            metrics = _MutableMetrics(**vars(self._metrics))
        return PipelineMetricsSnapshot(
            audio_chunks_received=metrics.audio_chunks_received,
            audio_chunks_dropped=self._source.dropped_chunks,
            audio_reconnections=getattr(self._source, "reconnections", 0),
            audio_reconnect_failures=getattr(self._source, "reconnect_failures", 0),
            utterances_detected=metrics.utterances_detected,
            utterances_dropped=metrics.utterances_dropped,
            recognitions_succeeded=metrics.recognitions_succeeded,
            recognitions_empty=metrics.recognitions_empty,
            recognition_errors=metrics.recognition_errors,
            playback_chunks_suppressed=metrics.playback_chunks_suppressed,
            publish_enqueued=metrics.publish_enqueued,
            audio_queue_size=getattr(self._source, "queue_size", 0),
            utterance_queue_size=self._utterances.qsize(),
        )

    def _increment(self, field: str, amount: int = 1) -> None:
        with self._metrics_lock:
            setattr(self._metrics, field, getattr(self._metrics, field) + amount)

    def _segment_loop(self) -> None:
        was_muted = False
        speech_was_active = False
        discontinuity_count = getattr(self._source, "discontinuity_count", 0)
        while not self._stop.is_set():
            chunk = self._source.read(timeout=0.2)
            current_discontinuities = getattr(
                self._source, "discontinuity_count", discontinuity_count
            )
            if current_discontinuities != discontinuity_count:
                self._segmenter.reset()
                was_muted = False
                speech_was_active = False
                discontinuity_count = current_discontinuities
                logger.warning("Audio discontinuity detected; VAD state reset")
            if chunk is None:
                continue
            self._increment("audio_chunks_received")
            if self._suppress_during_playback and self._gate.is_muted(
                now_ns=chunk.captured_monotonic_ns
            ):
                self._increment("playback_chunks_suppressed")
                if not was_muted:
                    self._segmenter.reset()
                was_muted = True
                continue
            if was_muted:
                self._segmenter.reset()
                was_muted = False
            try:
                utterances = self._segmenter.accept(chunk)
            except Exception:  # noqa: BLE001
                logger.exception("VAD processing failed; resetting segmenter")
                self._segmenter.reset()
                continue
            speech_active = bool(getattr(self._segmenter, "speech_active", False))
            if speech_active and not speech_was_active and self._on_speech_start is not None:
                detected_ns = time.monotonic_ns()
                try:
                    self._on_speech_start(detected_ns)
                except Exception:  # noqa: BLE001
                    logger.exception("Speech-start callback failed")
            speech_was_active = speech_active
            for utterance in utterances:
                self._increment("utterances_detected")
                self._enqueue_utterance(utterance)

    def _enqueue_utterance(self, utterance: Utterance) -> None:
        try:
            self._utterances.put_nowait(utterance)
            return
        except queue.Full:
            self._increment("utterances_dropped")
        try:
            self._utterances.get_nowait()
        except queue.Empty:
            pass
        try:
            self._utterances.put_nowait(utterance)
        except queue.Full:
            self._increment("utterances_dropped")

    def _recognition_loop(self) -> None:
        try:
            warmup = getattr(self._engine, "warmup", None)
            if warmup is not None:
                warmup()
        except Exception as error:  # noqa: BLE001
            self._recognition_start_error = error
            logger.exception("ASR startup warm-up failed")
            self._recognition_ready.set()
            return
        self._recognition_ready.set()
        while not self._stop.is_set() or not self._utterances.empty():
            try:
                utterance = self._utterances.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                recognition_started_ns = time.monotonic_ns()
                result = self._engine.transcribe(utterance)
            except Exception:  # noqa: BLE001
                self._increment("recognition_errors")
                logger.exception("ASR recognition failed")
                continue
            if not result.text:
                self._increment("recognitions_empty")
                continue
            recognition_finished_ns = time.monotonic_ns()
            self._sequence += 1
            turn_id = str(uuid.uuid4())
            previous_stamp = self._control.current
            stamp = self._control.begin_turn(turn_id)
            turn_started_ns = time.monotonic_ns()
            # The new stamp is authoritative before stale-output cleanup.  The
            # callback is deliberately output-only: it must never invalidate
            # or advance the newly created turn.
            if (
                previous_stamp is not None
                and previous_stamp != stamp
                and self._on_turn_superseded is not None
            ):
                try:
                    self._on_turn_superseded(previous_stamp, stamp)
                except Exception:  # noqa: BLE001
                    logger.exception(
                        "Superseded-turn output cleanup failed: previous=%s new=%s",
                        previous_stamp,
                        stamp,
                    )
            if self._timing is not None:
                if previous_stamp is not None and previous_stamp != stamp:
                    self._timing.mark(
                        previous_stamp,
                        "superseded",
                        at_ns=turn_started_ns,
                        by_turn_id=stamp.turn_id,
                    )
                    self._timing.finish(
                        previous_stamp,
                        "superseded",
                        by_turn_id=stamp.turn_id,
                    )
                self._timing.mark(
                    stamp,
                    "speech_start",
                    at_ns=utterance.started_monotonic_ns,
                )
                self._timing.mark(
                    stamp,
                    "speech_end",
                    at_ns=utterance.ended_monotonic_ns,
                )
                self._timing.mark(
                    stamp,
                    "vad_ready",
                    at_ns=utterance.vad_ready_monotonic_ns,
                )
                self._timing.mark(
                    stamp,
                    "asr_start",
                    at_ns=recognition_started_ns,
                )
                self._timing.mark(
                    stamp,
                    "asr_final",
                    at_ns=recognition_finished_ns,
                    inference_ms=result.inference_ms,
                    text_characters=len(result.text),
                )
                self._timing.mark(stamp, "turn_started", at_ns=turn_started_ns)
            event = SpeechEvent(
                event_id=turn_id,
                session_id=self._session_id,
                sequence=self._sequence,
                created_unix_ns=time.time_ns(),
                source=self._source_name,
                text=result.text,
                language=result.language,
                audio_duration_ms=utterance.duration_ms,
                inference_ms=result.inference_ms,
                engine=result.engine,
                is_final=True,
                turn_id=stamp.turn_id,
                epoch=stamp.epoch,
            )
            vad_ready_ns = utterance.vad_ready_monotonic_ns
            logger.info(
                "Speech latency event_id=%s vad_tail=%.1fms asr_queue=%.1fms "
                "inference=%.1fms speech_end_to_final=%.1fms",
                event.event_id,
                max(0, vad_ready_ns - utterance.ended_monotonic_ns) / 1_000_000,
                max(0, recognition_started_ns - vad_ready_ns) / 1_000_000,
                result.inference_ms,
                max(0, recognition_finished_ns - utterance.ended_monotonic_ns)
                / 1_000_000,
            )
            self._increment("recognitions_succeeded")
            if self._sink.publish(event):
                self._increment("publish_enqueued")
                if self._timing is not None:
                    self._timing.mark(
                        stamp,
                        "speech_event_published",
                        at_ns=time.monotonic_ns(),
                    )
            else:
                logger.error("Transport rejected SpeechEvent: event_id=%s", event.event_id)
                if self._timing is not None:
                    failed_ns = time.monotonic_ns()
                    self._timing.mark(
                        stamp,
                        "speech_publish_error",
                        at_ns=failed_ns,
                    )
                    self._timing.finish(
                        stamp,
                        "speech_publish_error",
                        at_ns=failed_ns,
                    )
