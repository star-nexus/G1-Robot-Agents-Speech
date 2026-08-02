"""Composition root for the Orin speech service."""

from __future__ import annotations

import logging

from .audio import SoundDeviceSource
from .config import ServiceConfig
from .dds import (
    DdsPlaybackSubscriber,
    RetryingEventSink,
    UnitreeDdsEventWriter,
    initialize_unitree_dds,
)
from .engine import SenseVoiceEngine
from .gate import PlaybackGate
from .pipeline import SpeechPipeline
from .vad import SileroVadSegmenter

logger = logging.getLogger(__name__)


class SpeechService:
    def __init__(self, config: ServiceConfig) -> None:
        self.config = config
        initialize_unitree_dds(config.dds.domain_id, config.dds.network_interface)

        self.gate = PlaybackGate(
            resume_delay_ms=config.playback.resume_delay_ms,
            max_active_seconds=config.playback.max_active_seconds,
        )
        self.playback_subscriber = DdsPlaybackSubscriber(
            self.gate, topic=config.dds.playback_topic
        )
        writer = UnitreeDdsEventWriter(config.dds.speech_topic)
        self.sink = RetryingEventSink(
            writer,
            capacity=config.dds.outbox_capacity,
            write_timeout_seconds=config.dds.write_timeout_seconds,
            retry_interval_seconds=config.dds.retry_interval_seconds,
            delivery_ttl_seconds=config.dds.delivery_ttl_seconds,
        )
        source = SoundDeviceSource(
            sample_rate=config.audio.sample_rate,
            block_ms=config.audio.block_ms,
            device=config.audio.device,
            queue_seconds=config.audio.queue_seconds,
        )
        segmenter = SileroVadSegmenter(
            model=config.vad.model,
            sample_rate=config.audio.sample_rate,
            threshold=config.vad.threshold,
            speech_pre_roll_seconds=config.vad.speech_pre_roll_seconds,
            min_silence_seconds=config.vad.min_silence_seconds,
            min_speech_seconds=config.vad.min_speech_seconds,
            max_speech_seconds=config.vad.max_speech_seconds,
            buffer_seconds=config.vad.buffer_seconds,
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

    def start(self) -> None:
        if self._started:
            return
        self.playback_subscriber.start()
        try:
            self.pipeline.start()
        except Exception:
            self.playback_subscriber.close()
            raise
        self._started = True

    def close(self) -> None:
        if not self._started:
            return
        self.pipeline.close()
        self.playback_subscriber.close()
        self._started = False
