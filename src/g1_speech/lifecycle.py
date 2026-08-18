"""Compatibility alias for :mod:`star_runtime.speech.lifecycle`."""
import sys
from star_runtime.speech import lifecycle as _impl
sys.modules[__name__] = _impl
