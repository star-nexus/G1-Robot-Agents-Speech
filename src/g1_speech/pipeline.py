"""Bounded three-stage speech pipeline: capture -> VAD -> ASR -> sink."""

from __future__ import annotations

import logging
import queue
import threading
import time
import uuid
from dataclasses import dataclass

from .contracts import (
    AsrEngine,
    AudioSource,
    EventSink,
    PipelineMetricsSnapshot,
    Segmenter,
    SpeechEvent,
    Utterance,
)
from .gate import PlaybackGate

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
        self._session_id = session_id or uuid.uuid4().hex
        self._sequence = 0
        self._metrics = _MutableMetrics()
        self._metrics_lock = threading.Lock()
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._started = False

    def prepare(self) -> None:
        """Load heavyweight inference resources without starting capture."""
        self._engine.load()

    def start(self) -> None:
        if self._started:
            return
        self._sink.start()
        try:
            self.prepare()
            self._source.start()
        except Exception:
            self._source.close()
            self._sink.close()
            raise
        self._stop.clear()
        self._threads = [
            threading.Thread(target=self._segment_loop, name="speech-vad", daemon=True),
            threading.Thread(target=self._recognition_loop, name="speech-asr", daemon=True),
        ]
        for thread in self._threads:
            thread.start()
        self._started = True
        logger.info("Speech pipeline started: session_id=%s", self._session_id)

    def close(self, *, join_timeout: float = 3.0) -> None:
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
        close_engine = getattr(self._engine, "close", None)
        if close_engine is not None:
            close_engine()
        self._sink.close()
        self._started = False
        logger.info("Speech pipeline stopped")

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
        discontinuity_count = getattr(self._source, "discontinuity_count", 0)
        while not self._stop.is_set():
            chunk = self._source.read(timeout=0.2)
            current_discontinuities = getattr(
                self._source, "discontinuity_count", discontinuity_count
            )
            if current_discontinuities != discontinuity_count:
                self._segmenter.reset()
                was_muted = False
                discontinuity_count = current_discontinuities
                logger.warning("Audio discontinuity detected; VAD state reset")
            if chunk is None:
                continue
            self._increment("audio_chunks_received")
            if self._gate.is_muted(now_ns=chunk.captured_monotonic_ns):
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
        while not self._stop.is_set() or not self._utterances.empty():
            try:
                utterance = self._utterances.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                result = self._engine.transcribe(utterance)
            except Exception:  # noqa: BLE001
                self._increment("recognition_errors")
                logger.exception("ASR recognition failed")
                continue
            if not result.text:
                self._increment("recognitions_empty")
                continue
            self._sequence += 1
            event = SpeechEvent(
                event_id=str(uuid.uuid4()),
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
            )
            self._increment("recognitions_succeeded")
            if self._sink.publish(event):
                self._increment("publish_enqueued")
            else:
                logger.error("Transport rejected SpeechEvent: event_id=%s", event.event_id)
