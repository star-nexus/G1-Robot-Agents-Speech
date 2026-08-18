"""Compatibility alias for :mod:`star_runtime.speech.audio.input`."""
import sys
from star_runtime.speech.audio import input as _impl
sys.modules[__name__] = _impl
