"""Compatibility alias for :mod:`star_runtime.apps.local_voice_agent`."""

import sys

from star_runtime.apps import local_voice_agent as _impl

sys.modules[__name__] = _impl
