"""Subset imported by vLLM's model-independent warm-up registry."""

from enum import Enum


class InterpolationMode(Enum):
    NEAREST = "nearest"
    NEAREST_EXACT = "nearest-exact"
    BILINEAR = "bilinear"
    BICUBIC = "bicubic"
    BOX = "box"
    HAMMING = "hamming"
    LANCZOS = "lanczos"
