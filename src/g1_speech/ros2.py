"""Compatibility alias for :mod:`star_runtime.transports.ros2.speech`."""
import sys
from star_runtime.transports.ros2 import speech as _impl
sys.modules[__name__] = _impl
