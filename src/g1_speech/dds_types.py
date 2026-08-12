"""CycloneDDS IDL messages shared by the speech service and Agent process.

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

    @dataclass
    class PlaybackStateMessage(IdlStruct, typename="g1_hri.msg.PlaybackState"):
        request_id: str
        active: bool
        created_unix_ns: idl_types.uint64
        source: str

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

    @dataclass
    class PlaybackStateMessage:
        request_id: str
        active: bool
        created_unix_ns: int
        source: str

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
