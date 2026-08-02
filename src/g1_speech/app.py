"""Composition root for the speech recognition service."""

from __future__ import annotations

import logging
from dataclasses import asdict
from typing import Any

from .audio import SoundDeviceSource
from .config import ServiceConfig
from .contracts import SpeechTransport
from .engine import SenseVoiceEngine
from .gate import PlaybackGate
from .pipeline import SpeechPipeline
from .vad import SileroVadSegmenter
from .transports import create_transport

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
        source = SoundDeviceSource(settings=config.audio)
        segmenter = SileroVadSegmenter(
            settings=config.vad,
            sample_rate=config.audio.sample_rate,
        )
        engine = SenseVoiceEngine(
            model_dir=config.sensevoice.model_dir,
            model_file=config.sensevoice.model_file,
            device=config.sensevoice.device,
            sample_rate=config.audio.sample_rate,
            language=config.sensevoice.language,
            use_itn=config.sensevoice.use_itn,
            num_threads=config.sensevoice.num_threads,
        )
        self.transport = transport or create_transport(
            config,
            self.gate,
            backend=transport_backend,
            ros_node=ros_node,
            ros_lifecycle=ros_lifecycle,
        )
        self.sink = self.transport.sink
        self.pipeline = SpeechPipeline(
            source=source,
            segmenter=segmenter,
            engine=engine,
            sink=self.sink,
            playback_gate=self.gate,
            source_name=config.source_name,
            utterance_queue_capacity=config.utterance_queue_capacity,
        )
        self._started = False

    def prepare(self) -> None:
        self.pipeline.prepare()

    def start(self) -> None:
        if self._started:
            return
        try:
            self.transport.start()
            self.pipeline.start()
        except Exception:
            self.transport.stop()
            raise
        self._started = True

    def stop(self) -> None:
        if not self._started:
            return
        self.pipeline.close()
        self.transport.stop()
        self._started = False

    def close(self) -> None:
        self.stop()
        self.transport.close()

    def metrics(self) -> dict[str, Any]:
        payload = asdict(self.pipeline.metrics())
        payload.update(self.transport.metrics())
        return payload
