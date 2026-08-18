"""Compatibility alias for :mod:`star_runtime.cli.main`."""

import sys

from star_runtime.cli import main as _impl

sys.modules[__name__] = _impl
