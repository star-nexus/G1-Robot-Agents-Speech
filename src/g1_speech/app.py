"""Composition root for the speech recognition service."""

from __future__ import annotations

import logging
from dataclasses import asdict
from typing import Any

from .audio import SoundDeviceSource
from .audio_output import AlsaOutputPlayer
from .audio_processing import create_audio_processor
from .asr import create_asr_engine
from .config import ServiceConfig
from .contracts import SpeechTransport
from .gate import PlaybackGate
from .pipeline import SpeechPipeline
from .vad import SileroVadSegmenter
from .transports import create_transport
from .tts import TtsController, VllmOmniStreamingTts

logger = logging.getLogger(__name__)


class SpeechService:
    def __init__(
        self,
        config: ServiceConfig,
        *,
        transport: SpeechTransport | None = None,
        transport_backend: str | None = None,
        ros_node: Any | None = None,
        ros_lifecycle: bool = False,
    ) -> None:
        self.config = config
        self.gate = PlaybackGate(
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
        self.transport = transport or create_transport(
            config,
            self.gate,
            backend=transport_backend,
            ros_node=ros_node,
            ros_lifecycle=ros_lifecycle,
        )
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

    def prepare(self) -> None:
        self.pipeline.prepare()
        self._prepared = True

    def start(self) -> None:
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
        self.pipeline.close()
        self.transport.stop()
        self._started = False
        self._prepared = False

    def close(self) -> None:
        if self._started:
            self.stop()
        elif self._prepared:
            self.pipeline.close()
            self._prepared = False
        if self.tts is not None:
            self.tts.close()
        self.transport.close()
        self.audio_processor.close()

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
