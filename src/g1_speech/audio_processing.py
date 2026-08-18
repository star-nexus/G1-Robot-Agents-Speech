"""Compatibility alias for :mod:`star_runtime.speech.audio.processing`."""
import sys
from star_runtime.speech.audio import processing as _impl
sys.modules[__name__] = _impl
