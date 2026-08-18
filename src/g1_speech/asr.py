"""Compatibility alias for :mod:`star_runtime.speech.asr.factory`."""
import sys
from star_runtime.speech.asr import factory as _impl
sys.modules[__name__] = _impl
