"""Composition root for the speech recognition service."""

from __future__ import annotations

import logging
import time
from dataclasses import asdict
from typing import Any

from ..core.control import ControlStamp, RuntimeControlPlane
from ..core.timing import RuntimeTimingAudit
from .audio.input import SoundDeviceSource
from .audio.output import AlsaOutputPlayer
from .audio.processing import create_audio_processor
from .asr.factory import create_asr_engine
from .config import ServiceConfig
from ..transports.contracts import SpeechTransport
from .playback import PlaybackGate
from .pipeline import SpeechPipeline
from .vad import SileroVadSegmenter
from .tts.streaming import TtsController, VllmOmniStreamingTts

logger = logging.getLogger(__name__)


class SpeechService:
    def __init__(
        self,
        config: ServiceConfig,
        *,
        transport: SpeechTransport,
        playback_gate: PlaybackGate | None = None,
        timing_audit: RuntimeTimingAudit | None = None,
    ) -> None:
        self.config = config
        self.timing_audit = timing_audit
        self.control = RuntimeControlPlane()
        self.gate = playback_gate or PlaybackGate(
            resume_delay_ms=config.playback.resume_delay_ms,
            max_active_seconds=config.playback.max_active_seconds,
        )
        self.gate.set_validator(self.control.is_active)
        self.audio_processor = create_audio_processor(config.audio_processing)
        source = SoundDeviceSource(
            settings=config.audio,
            processor=self.audio_processor,
        )
        self.audio_source = source
        segmenter = SileroVadSegmenter(
            settings=config.vad,
            sample_rate=config.audio.sample_rate,
        )
        engine = create_asr_engine(config)
        self.transport = transport
        self.sink = self.transport.sink
        self.tts: TtsController | None = None
        if config.tts.enabled:
            register_tts = getattr(self.transport, "register_tts_handler", None)
            publish_playback = getattr(self.transport, "publish_playback_state", None)
            if register_tts is None or publish_playback is None:
                raise RuntimeError("selected transport does not support streaming TTS")
            player = AlsaOutputPlayer(
                config.audio_output,
                render_sink=self.audio_processor,
                timing_audit=timing_audit,
            )
            self.tts = TtsController(
                config.tts,
                engine=VllmOmniStreamingTts(config.tts),
                player=player,
                playback_state=publish_playback,
                control_plane=self.control,
                timing_audit=timing_audit,
            )
            register_tts(self.tts.accept)
        self.pipeline = SpeechPipeline(
            source=source,
            segmenter=segmenter,
            engine=engine,
            sink=self.sink,
            playback_gate=self.gate,
            source_name=config.source_name,
            utterance_queue_capacity=config.utterance_queue_capacity,
            session_id=self.control.session_id,
            control_plane=self.control,
            timing_audit=timing_audit,
            on_speech_start=(
                self._on_speech_start
                if self.tts is not None and config.audio_processing.mode != "off"
                else None
            ),
            on_turn_superseded=(
                self._flush_stale_output if self.tts is not None else None
            ),
            suppress_during_playback=config.audio_processing.mode == "off",
        )
        self._started = False
        self._prepared = False
        self._closed = False

    def _on_speech_start(self, detected_ns: int) -> None:
        note_vad_edge = getattr(
            getattr(self, "audio_processor", None),
            "note_vad_edge",
            None,
        )
        if note_vad_edge is not None:
            note_vad_edge(detected_ns)
        current = self.control.current
        if current is not None and self.timing_audit is not None:
            self.timing_audit.mark(
                current,
                "vad_speech_edge",
                at_ns=detected_ns,
            )
        if current is not None:
            logger.info(
                "VAD_SPEECH_EDGE session_id=%s turn_id=%s epoch=%d",
                current.session_id,
                current.turn_id,
                current.epoch,
            )
        self._invalidate_active_response("barge-in", tts_reason="vad")

    def _invalidate_active_response(
        self,
        reason: str,
        *,
        tts_reason: str | None = None,
    ) -> bool:
        """Advance the active epoch, then preempt its externally visible work."""

        event = self.control.invalidate(reason)
        if event is not None:
            invalidated_ns = time.monotonic_ns()
            if self.timing_audit is not None:
                self.timing_audit.mark(
                    event.stamp,
                    "invalidated",
                    at_ns=invalidated_ns,
                    reason=event.reason,
                )
            output_preempted = False
            if self.tts is not None:
                output_preempted = self.tts.preempt_output(
                    reason=tts_reason or reason
                )
            if (
                output_preempted
                and self.timing_audit is not None
                and tts_reason == "vad"
            ):
                self.timing_audit.mark(
                    event.stamp,
                    "vad_output_preempted",
                    at_ns=time.monotonic_ns(),
                )
            publish_control = getattr(self.transport, "publish_control", None)
            if publish_control is not None:
                publish_control(event)
            if self.tts is not None and output_preempted:
                self.tts.cancel_provider()
            if self.timing_audit is not None:
                self.timing_audit.finish(
                    event.stamp,
                    "invalidated",
                    reason=event.reason,
                )
            return True
        return False

    def _flush_stale_output(
        self,
        previous_stamp: ControlStamp,
        new_stamp: ControlStamp,
    ) -> None:
        """Clear old TTS/PCM after begin_turn without mutating Control Plane."""

        if self.tts is None:
            return
        interrupted = self.tts.interrupt(reason="asr-final-superseded")
        cleanup_ns = time.monotonic_ns()
        if self.timing_audit is not None:
            self.timing_audit.mark(
                previous_stamp,
                "asr_final_fallback_cleanup",
                at_ns=cleanup_ns,
                by_turn_id=new_stamp.turn_id,
                output_interrupted=bool(interrupted),
            )
        logger.info(
            "ASR_FINAL_FALLBACK_CLEANUP previous_turn_id=%s previous_epoch=%d "
            "new_turn_id=%s new_epoch=%d output_interrupted=%s",
            previous_stamp.turn_id,
            previous_stamp.epoch,
            new_stamp.turn_id,
            new_stamp.epoch,
            bool(interrupted),
        )

    def cancel_active_turn(self, reason: str = "explicit") -> bool:
        """Invalidate and stop the active epoch; safe from callback threads."""

        return self._invalidate_active_response(reason)

    def prepare(self) -> None:
        if self._closed:
            raise RuntimeError("speech service is closed")
        self.pipeline.prepare()
        self._prepared = True

    def start(self) -> None:
        if self._closed:
            raise RuntimeError("speech service is closed")
        if self._started:
            return
        try:
            if self.tts is not None:
                self.tts.start()
            self.transport.start()
            self.pipeline.start()
        except Exception:
            if self.tts is not None:
                self.tts.stop()
            self.transport.stop()
            raise
        self._started = True
        self._prepared = True

    def stop(self) -> None:
        if not self._started:
            return
        if self.tts is not None:
            self.tts.stop()
        self.pipeline.stop()
        self.transport.stop()
        self._started = False
        self._prepared = True

    def close(self) -> None:
        if self._closed:
            return
        try:
            if self._started:
                self.stop()
        finally:
            try:
                self.pipeline.close()
                self._prepared = False
            finally:
                try:
                    if self.tts is not None:
                        self.tts.close()
                finally:
                    try:
                        self.transport.close()
                    finally:
                        self.audio_processor.close()
                        self._closed = True

    def metrics(self) -> dict[str, Any]:
        payload = asdict(self.pipeline.metrics())
        payload.update(self.transport.metrics())
        payload.update(self.audio_processor.metrics())
        if self.tts is not None:
            payload.update(self.tts.metrics())
        else:
            payload["tts_enabled"] = False
        payload.update(
            {
                "control_session_id": self.control.session_id,
                "control_invalidations": self.control.invalidations,
                "control_stale_drops": self.control.stale_drops,
                "runtime_timing_timelines_emitted": (
                    0
                    if self.timing_audit is None
                    else self.timing_audit.timelines_emitted
                ),
            }
        )
        payload.update(
            {
                "audio_input_backend": getattr(
                    self.audio_source, "active_backend", None
                ),
                "audio_input_device": getattr(
                    self.audio_source, "active_device", None
                ),
                "audio_input_latency_ms": (
                    None
                    if getattr(self.audio_source, "actual_latency_seconds", None)
                    is None
                    else round(self.audio_source.actual_latency_seconds * 1000, 3)
                ),
            }
        )
        return payload
