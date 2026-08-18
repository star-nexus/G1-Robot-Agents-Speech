"""Compatibility facade for the structured speech settings package."""

from .settings import (
    AsrConfig,
    AudioConfig,
    AudioOutputConfig,
    AudioProcessingConfig,
    DdsConfig,
    PlaybackConfig,
    Qwen3AsrConfig,
    Ros2Config,
    SenseVoiceConfig,
    ServiceConfig,
    TransportConfig,
    TtsConfig,
    VadConfig,
    default_config_dict,
    load_config,
    write_config,
)

__all__ = [
    "AsrConfig",
    "AudioConfig",
    "AudioOutputConfig",
    "AudioProcessingConfig",
    "DdsConfig",
    "PlaybackConfig",
    "Qwen3AsrConfig",
    "Ros2Config",
    "SenseVoiceConfig",
    "ServiceConfig",
    "TransportConfig",
    "TtsConfig",
    "VadConfig",
    "default_config_dict",
    "load_config",
    "write_config",
]

