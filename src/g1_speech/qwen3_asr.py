"""Compatibility alias for :mod:`star_runtime.speech.asr.qwen3`."""
import sys
from star_runtime.speech.asr import qwen3 as _impl
sys.modules[__name__] = _impl
