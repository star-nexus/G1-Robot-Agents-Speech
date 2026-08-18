"""Compatibility alias for :mod:`star_runtime.speech.pipeline`."""
import sys
from star_runtime.speech import pipeline as _impl
sys.modules[__name__] = _impl
