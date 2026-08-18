"""Compatibility alias for :mod:`star_runtime.speech.audio.output`."""
import sys
from star_runtime.speech.audio import output as _impl
sys.modules[__name__] = _impl
