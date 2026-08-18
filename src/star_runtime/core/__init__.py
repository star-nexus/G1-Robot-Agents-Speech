"""Dependency-light domain messages shared by in-process and remote adapters."""

from .events import PlaybackState, SpeechEvent, TtsTextChunk

__all__ = ["PlaybackState", "SpeechEvent", "TtsTextChunk"]
