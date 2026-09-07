"""Canonical HRI messages independent of ASR, TTS, DDS, and ROS 2."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SpeechEvent:
    event_id: str
    session_id: str
    sequence: int
    created_unix_ns: int
    source: str
    text: str
    language: str
    audio_duration_ms: int
    inference_ms: float
    engine: str = "sensevoice"
    is_final: bool = True
    turn_id: str = ""
    epoch: int = 0

    @property
    def control_stamp(self):
        from .control import ControlStamp
        return ControlStamp(self.session_id, self.turn_id, self.epoch)


@dataclass(frozen=True)
class TtsTextChunk:
    request_id: str
    sequence: int
    text: str
    is_final: bool = False
    interrupt: bool = False
    language: str = ""
    voice: str = ""
    instructions: str = ""
    created_unix_ns: int = 0
    source: str = "agent"
    session_id: str = ""
    turn_id: str = ""
    epoch: int = 0

    @property
    def control_stamp(self):
        from .control import ControlStamp
        return ControlStamp(self.session_id, self.turn_id, self.epoch)


@dataclass(frozen=True)
class PlaybackState:
    request_id: str
    active: bool
    created_unix_ns: int
    source: str
    session_id: str = ""
    turn_id: str = ""
    epoch: int = 0

    @property
    def control_stamp(self):
        from .control import ControlStamp
        return ControlStamp(self.session_id, self.turn_id, self.epoch)
