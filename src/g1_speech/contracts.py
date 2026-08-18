"""Compatibility alias for :mod:`star_runtime.speech.contracts`."""
import sys
from star_runtime.speech import contracts as _impl
sys.modules[__name__] = _impl
