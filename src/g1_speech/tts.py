"""Compatibility alias for :mod:`star_runtime.speech.tts.streaming`."""
import sys
from star_runtime.speech.tts import streaming as _impl
sys.modules[__name__] = _impl
