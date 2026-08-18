"""Compatibility alias for :mod:`star_runtime.speech.service`."""
import sys
from star_runtime.speech import service as _impl
sys.modules[__name__] = _impl
