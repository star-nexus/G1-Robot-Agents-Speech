"""Compatibility alias for :mod:`star_runtime.speech.evaluation`."""
import sys
from star_runtime.speech import evaluation as _impl
sys.modules[__name__] = _impl
