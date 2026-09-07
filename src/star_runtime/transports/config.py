"""Transport-only settings shared by speech and Agent compositions."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DdsConfig:
    domain_id: int = 0
    network_interface: str | None = None
    speech_topic: str = "rt/g1/hri/speech/final"
    playback_topic: str = "rt/g1/hri/playback/state"
    tts_topic: str = "rt/g1/hri/tts/request"
    control_topic: str = "rt/g1/hri/control/epoch"
    write_timeout_seconds: float = 0.5
    retry_interval_seconds: float = 0.2
    delivery_ttl_seconds: float = 120.0
    outbox_capacity: int = 128


@dataclass(frozen=True)
class Ros2Config:
    node_name: str = "g1_speech"
    speech_topic: str = "hri/speech/final"
    playback_topic: str = "hri/playback/state"
    tts_topic: str = "hri/tts/request"
    control_topic: str = "hri/control/epoch"
    qos_depth: int = 10


@dataclass(frozen=True)
class TransportConfig:
    backend: str = "dds"
