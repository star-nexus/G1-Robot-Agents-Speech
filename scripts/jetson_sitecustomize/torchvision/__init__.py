"""Minimal text/audio-only torchvision compatibility module.

The isolated Jetson PyTorch runtime can see Ubuntu's incompatible system
torchvision. vLLM's generic kernel warm-up imports one vision processor even
while serving Qwen3-TTS; that processor only needs ``InterpolationMode``.
Keeping this tiny shim ahead of system packages avoids loading binary vision
operators and does not modify the operating-system installation.
"""

from .transforms import InterpolationMode

__all__ = ["InterpolationMode"]
__version__ = "0.0.text-audio-stub"
