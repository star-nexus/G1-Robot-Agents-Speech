"""Compatibility alias for :mod:`star_runtime.speech.playback`."""
import sys
from star_runtime.speech import playback as _impl
sys.modules[__name__] = _impl
