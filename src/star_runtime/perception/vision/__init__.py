"""Resource-bounded camera capture and visual-turn routing."""

from .camera import LatestFrameCamera, MjpegStreamParser
from .llama_cpp import LlamaVisionClassifier, LlamaVisionClassifierSettings
from .routing import (
    TEXT,
    UNCERTAIN,
    VISION,
    RoutedVisionInput,
    VisionRouteDecision,
    VisionRouteRecorder,
    VisionRouter,
)

__all__ = [
    "LatestFrameCamera",
    "LlamaVisionClassifier",
    "LlamaVisionClassifierSettings",
    "MjpegStreamParser",
    "RoutedVisionInput",
    "TEXT",
    "UNCERTAIN",
    "VISION",
    "VisionRouteDecision",
    "VisionRouteRecorder",
    "VisionRouter",
]
