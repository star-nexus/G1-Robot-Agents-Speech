"""Cyclone DDS IDL messages shared by speech and Agent processes.

Do not enable postponed annotations in this module: CycloneDDS 0.10.2 resolves
IDL field types at runtime and cannot resolve stringified built-in annotations.
"""

from dataclasses import dataclass

try:
    from cyclonedds.idl import IdlStruct
    from cyclonedds.idl import types as idl_types

    DDS_IDL_AVAILABLE = True

    @dataclass
    class SpeechEventMessage(IdlStruct, typename="g1_hri.msg.SpeechEvent"):
        event_id: str
        session_id: str
        sequence: idl_types.uint64
        created_unix_ns: idl_types.uint64
        source: str
        text: str
        language: str
        audio_duration_ms: idl_types.uint32
        inference_ms: idl_types.float32
        engine: str
        is_final: bool
        turn_id: str = ""
        epoch: idl_types.uint64 = 0

    @dataclass
    class PlaybackStateMessage(IdlStruct, typename="g1_hri.msg.PlaybackState"):
        request_id: str
        active: bool
        created_unix_ns: idl_types.uint64
        source: str
        session_id: str = ""
        turn_id: str = ""
        epoch: idl_types.uint64 = 0

    @dataclass
    class TtsTextChunkMessage(IdlStruct, typename="g1_hri.msg.TtsTextChunk"):
        request_id: str
        sequence: idl_types.uint64
        text: str
        is_final: bool
        interrupt: bool
        language: str
        voice: str
        instructions: str
        created_unix_ns: idl_types.uint64
        source: str
        session_id: str = ""
        turn_id: str = ""
        epoch: idl_types.uint64 = 0

    @dataclass
    class EpochInvalidatedMessage(IdlStruct, typename="g1_hri.msg.EpochInvalidated"):
        event_id: str
        session_id: str
        turn_id: str
        epoch: idl_types.uint64
        next_epoch: idl_types.uint64
        reason: str
        created_unix_ns: idl_types.uint64
        source: str

except ImportError:
    DDS_IDL_AVAILABLE = False

    @dataclass
    class SpeechEventMessage:
        event_id: str
        session_id: str
        sequence: int
        created_unix_ns: int
        source: str
        text: str
        language: str
        audio_duration_ms: int
        inference_ms: float
        engine: str
        is_final: bool
        turn_id: str = ""
        epoch: int = 0

    @dataclass
    class PlaybackStateMessage:
        request_id: str
        active: bool
        created_unix_ns: int
        source: str
        session_id: str = ""
        turn_id: str = ""
        epoch: int = 0

    @dataclass
    class TtsTextChunkMessage:
        request_id: str
        sequence: int
        text: str
        is_final: bool
        interrupt: bool
        language: str
        voice: str
        instructions: str
        created_unix_ns: int
        source: str
        session_id: str = ""
        turn_id: str = ""
        epoch: int = 0

    @dataclass
    class EpochInvalidatedMessage:
        event_id: str
        session_id: str
        turn_id: str
        epoch: int
        next_epoch: int
        reason: str
        created_unix_ns: int
        source: str
