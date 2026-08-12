#!/usr/bin/env python3
"""Jetson text/audio-only launcher for an isolated vLLM-Omni runtime.

JetPack's system-site ``torchvision`` is tied to the system PyTorch build. An
isolated, newer Jetson PyTorch can therefore see the package metadata but fail
while registering its compiled operators. Qwen3-TTS does not use torchvision,
so hide that optional capability before Transformers/vLLM imports it.
"""

from __future__ import annotations

import os


def main() -> None:
    if os.environ.get("VLLM_OMNI_DISABLE_TORCHVISION", "1") == "1":
        import transformers.utils as transformers_utils
        import transformers.utils.import_utils as import_utils

        def unavailable() -> bool:
            return False

        transformers_utils.is_torchvision_available = unavailable
        import_utils.is_torchvision_available = unavailable

        # vLLM-Omni imports Diffusers' generic pipeline types even for an
        # audio-only model. Diffusers keeps an independent package-availability
        # cache, so hide the mismatched JetPack system torchvision there too.
        import diffusers.utils as diffusers_utils
        import diffusers.utils.import_utils as diffusers_import_utils

        diffusers_import_utils._torchvision_available = False
        diffusers_utils.is_torchvision_available = unavailable

    from vllm_omni.entrypoints.cli.main import main as omni_main

    omni_main()


if __name__ == "__main__":
    main()
