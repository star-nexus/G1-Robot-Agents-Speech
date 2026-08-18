"""Compatibility facade for streaming text-to-speech components."""

from ...core.events import TtsTextChunk
from .contracts import StreamingTtsEngine, TtsAudioChunk, TtsSynthesisRequest
from .controller import TtsController
from .scheduler import SentenceAssembler
from .vllm_omni import VllmOmniStreamingTts

__all__ = [
    "SentenceAssembler",
    "StreamingTtsEngine",
    "TtsAudioChunk",
    "TtsController",
    "TtsSynthesisRequest",
    "TtsTextChunk",
    "VllmOmniStreamingTts",
]

