"""Compatibility alias for :mod:`star_runtime.speech.vad`."""
import sys
from star_runtime.speech import vad as _impl
sys.modules[__name__] = _impl
