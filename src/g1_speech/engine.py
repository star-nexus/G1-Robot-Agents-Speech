"""Compatibility alias for :mod:`star_runtime.speech.asr.sensevoice`."""
import sys
from star_runtime.speech.asr import sensevoice as _impl
sys.modules[__name__] = _impl
