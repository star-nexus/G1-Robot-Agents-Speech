"""Compatibility alias for :mod:`star_runtime.apps.speech_runtime`."""
import sys
from star_runtime.apps import speech_runtime as _impl
sys.modules[__name__] = _impl
