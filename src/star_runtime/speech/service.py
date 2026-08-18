"""Composition root for the speech recognition service."""

from __future__ import annotations

import logging
from dataclasses import asdict
from typing import Any

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
    ) -> None:
        self.config = config
        self.gate = playback_gate or PlaybackGate(
            resume_delay_ms=config.playback.resume_delay_ms,
            max_active_seconds=config.playback.max_active_seconds,
        )
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
            )
            self.tts = TtsController(
                config.tts,
                engine=VllmOmniStreamingTts(config.tts),
                player=player,
                playback_state=publish_playback,
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
            on_speech_start=(
                (lambda: self.tts.interrupt(reason="vad"))
                if self.tts is not None and config.audio_processing.mode != "off"
                else None
            ),
            suppress_during_playback=config.audio_processing.mode == "off",
        )
        self._started = False
        self._prepared = False
        self._closed = False

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
