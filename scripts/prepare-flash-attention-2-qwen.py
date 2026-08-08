#!/usr/bin/env python3
"""Trim the pinned FlashAttention sdist to Qwen3-ASR's head dimensions."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    args = parser.parse_args()
    root = args.source.resolve()

    setup_path = root / "setup.py"
    setup_text = setup_path.read_text(encoding="utf-8")
    kernel_line = re.compile(
        r'^\s+"csrc/flash_attn/src/flash_(?:fwd|bwd)[^"]+\.cu",\n',
        re.MULTILINE,
    )
    kernels = kernel_line.findall(setup_text)
    retained = [
        line for line in kernels if "_hdim64_" in line or "_hdim128_" in line
    ]
    if len(kernels) != 72 or len(retained) != 24:
        raise RuntimeError(
            "unexpected FlashAttention source layout: "
            f"found {len(kernels)} kernels and {len(retained)} Qwen head-dim kernels"
        )
    setup_text = kernel_line.sub(
        lambda match: (
            match.group(0)
            if "_hdim64_" in match.group(0) or "_hdim128_" in match.group(0)
            else ""
        ),
        setup_text,
    )
    sm80_flag = 'cc_flag.append("arch=compute_80,code=sm_80")'
    if setup_text.count(sm80_flag) != 1:
        raise RuntimeError("unexpected FlashAttention SM 8.0 compiler flag layout")
    setup_text = setup_text.replace(
        sm80_flag, 'cc_flag.append("arch=compute_87,code=sm_87")'
    )
    setup_path.write_text(setup_text, encoding="utf-8")

    switch_path = root / "csrc/flash_attn/src/static_switch.h"
    switch_text = switch_path.read_text(encoding="utf-8")
    start = switch_text.index("#define HEADDIM_SWITCH(HEADDIM, ...)")
    end = switch_text.index("  }()", start) + len("  }()")
    replacement = r'''#define HEADDIM_SWITCH(HEADDIM, ...)       \
  [&] {                                         \
    if ((HEADDIM) == 64) {                      \
      constexpr static int kHeadDim = 64;       \
      return __VA_ARGS__();                     \
    } else if ((HEADDIM) == 128) {              \
      constexpr static int kHeadDim = 128;      \
      return __VA_ARGS__();                     \
    }                                           \
    TORCH_CHECK(false, "This Qwen3-ASR build requires head dimension 64 or 128"); \
  }()'''
    switch_path.write_text(
        switch_text[:start] + replacement + switch_text[end:], encoding="utf-8"
    )
    print("[PASS] Retained 24 Qwen head-dim CUDA sources and targeted Orin SM 8.7")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
