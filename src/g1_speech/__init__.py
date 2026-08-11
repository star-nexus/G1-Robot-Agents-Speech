"""Pluggable offline speech recognition for robot Agents."""

from .asr import create_asr_engine
from .contracts import AudioChunk, RecognitionResult, SpeechEvent, Utterance
from .engine import SenseVoiceEngine
from .qwen3_asr import Qwen3AsrEngine, Qwen3AsrProfile
from .gate import PlaybackGate
from .pipeline import SpeechPipeline

__all__ = [
    "AudioChunk",
    "create_asr_engine",
    "PlaybackGate",
    "RecognitionResult",
    "Qwen3AsrEngine",
    "Qwen3AsrProfile",
    "SenseVoiceEngine",
    "SpeechEvent",
    "SpeechPipeline",
    "Utterance",
]
