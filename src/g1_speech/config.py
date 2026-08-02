"""JSON configuration with path resolution relative to the config file."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class AudioConfig:
    sample_rate: int = 16000
    block_ms: int = 100
    device: int | str | None = None
    queue_seconds: float = 5.0


@dataclass(frozen=True)
class VadConfig:
    model: str = "models/silero_vad.onnx"
    threshold: float = 0.5
    speech_pre_roll_seconds: float = 0.3
    min_silence_seconds: float = 0.35
    min_speech_seconds: float = 0.25
    max_speech_seconds: float = 15.0
    buffer_seconds: float = 30.0


@dataclass(frozen=True)
class SenseVoiceConfig:
    model_dir: str = "models"
    model_file: str | None = None
    device: str = "cpu"
    language: str = "zh"
    use_itn: bool = True
    num_threads: int = 4


@dataclass(frozen=True)
class DdsConfig:
    domain_id: int = 0
    network_interface: str | None = None
    speech_topic: str = "rt/g1/hri/speech/final"
    playback_topic: str = "rt/g1/hri/playback/state"
    write_timeout_seconds: float = 0.5
    retry_interval_seconds: float = 0.2
    delivery_ttl_seconds: float = 30.0
    outbox_capacity: int = 128


@dataclass(frozen=True)
class PlaybackConfig:
    resume_delay_ms: int = 250
    max_active_seconds: float = 30.0


@dataclass(frozen=True)
class ServiceConfig:
    source_name: str = "g1_speech_mic"
    utterance_queue_capacity: int = 4
    metrics_interval_seconds: float = 60.0
    audio: AudioConfig = field(default_factory=AudioConfig)
    vad: VadConfig = field(default_factory=VadConfig)
    sensevoice: SenseVoiceConfig = field(default_factory=SenseVoiceConfig)
    dds: DdsConfig = field(default_factory=DdsConfig)
    playback: PlaybackConfig = field(default_factory=PlaybackConfig)

    def validate(self) -> None:
        if self.audio.sample_rate != 16000:
            raise ValueError("SenseVoice/Silero VAD 服务固定使用 16000 Hz")
        if self.audio.block_ms <= 0 or self.audio.block_ms > 500:
            raise ValueError("audio.block_ms 必须在 1..500 之间")
        if self.audio.queue_seconds <= 0:
            raise ValueError("audio.queue_seconds 必须大于 0")
        if self.utterance_queue_capacity < 1:
            raise ValueError("utterance_queue_capacity 必须大于 0")
        if not 0 < self.vad.threshold < 1:
            raise ValueError("vad.threshold 必须在 0..1 之间")
        if not 0 <= self.vad.speech_pre_roll_seconds <= 1:
            raise ValueError("vad.speech_pre_roll_seconds 必须在 0..1 之间")
        if self.sensevoice.device not in {"cpu", "cuda", "auto"}:
            raise ValueError("sensevoice.device 必须是 cpu、cuda 或 auto")
        if self.vad.max_speech_seconds <= self.vad.min_speech_seconds:
            raise ValueError("vad.max_speech_seconds 必须大于 min_speech_seconds")
        if self.dds.outbox_capacity < 1:
            raise ValueError("dds.outbox_capacity 必须大于 0")


def _merge_dataclass(cls, raw: dict[str, Any]):
    allowed = cls.__dataclass_fields__.keys()
    unknown = set(raw) - set(allowed)
    if unknown:
        raise ValueError(f"{cls.__name__} 含未知配置项: {sorted(unknown)}")
    return cls(**raw)


def load_config(path: str | Path) -> ServiceConfig:
    config_path = Path(path).expanduser().resolve()
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    allowed_top = ServiceConfig.__dataclass_fields__.keys()
    unknown_top = set(raw) - set(allowed_top)
    if unknown_top:
        raise ValueError(f"ServiceConfig 含未知配置项: {sorted(unknown_top)}")

    base = config_path.parent
    audio = _merge_dataclass(AudioConfig, raw.pop("audio", {}))
    vad_raw = raw.pop("vad", {})
    vad_raw["model"] = str(_resolve_path(base, vad_raw.get("model", VadConfig.model)))
    vad = _merge_dataclass(VadConfig, vad_raw)
    sense_raw = raw.pop("sensevoice", {})
    sense_raw["model_dir"] = str(
        _resolve_path(base, sense_raw.get("model_dir", SenseVoiceConfig.model_dir))
    )
    model_file = sense_raw.get("model_file")
    sense_raw["model_file"] = (
        str(_resolve_path(base, model_file)) if model_file else None
    )
    sensevoice = _merge_dataclass(SenseVoiceConfig, sense_raw)
    dds = _merge_dataclass(DdsConfig, raw.pop("dds", {}))
    playback = _merge_dataclass(PlaybackConfig, raw.pop("playback", {}))
    config = ServiceConfig(
        audio=audio,
        vad=vad,
        sensevoice=sensevoice,
        dds=dds,
        playback=playback,
        **raw,
    )
    config.validate()
    return config


def _resolve_path(base: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()
