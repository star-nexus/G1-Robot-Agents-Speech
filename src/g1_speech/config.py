"""Compatibility alias for :mod:`star_runtime.speech.config`."""
import sys
from star_runtime.speech import config as _impl
sys.modules[__name__] = _impl
