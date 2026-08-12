"""Process-wide compatibility hooks for the isolated Jetson TTS runtime.

This module is loaded automatically by Python when its directory is on
``PYTHONPATH``.  The vLLM model registry inspects architectures in child
processes, so applying the hook only in the parent CLI process is insufficient.
"""

from __future__ import annotations

import os


if os.environ.get("VLLM_OMNI_DISABLE_TORCHVISION", "1") == "1":
    import transformers.utils as transformers_utils
    import transformers.utils.import_utils as transformers_import_utils

    def _unavailable() -> bool:
        return False

    transformers_import_utils._torchvision_available = False
    transformers_import_utils.is_torchvision_available = _unavailable
    transformers_utils.is_torchvision_available = _unavailable

    # Diffusers maintains an independent optional-dependency cache. Its generic
    # pipeline annotations are imported by vLLM-Omni even for text/audio-only
    # Qwen3-TTS serving.
    import diffusers.utils as diffusers_utils
    import diffusers.utils.import_utils as diffusers_import_utils

    diffusers_import_utils._torchvision_available = False
    diffusers_utils.is_torchvision_available = _unavailable
