#!/usr/bin/env python3
"""Measure the official single-process qwen-tts memory/latency floor."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import torch
from qwen_tts import Qwen3TTSModel


def cuda_snapshot() -> dict[str, int]:
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    return {
        "allocated_bytes": torch.cuda.memory_allocated(),
        "reserved_bytes": torch.cuda.memory_reserved(),
        "max_allocated_bytes": torch.cuda.max_memory_allocated(),
        "max_reserved_bytes": torch.cuda.max_memory_reserved(),
        "device_free_bytes": free_bytes,
        "device_total_bytes": total_bytes,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--text", default="欢迎来到未来世界。")
    parser.add_argument("--speaker", default="Vivian")
    parser.add_argument("--language", default="Chinese")
    parser.add_argument("--instructions", default="用温暖、自然、连贯的普通话说话。")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--attention", choices=("sdpa", "eager"), default="sdpa")
    parser.add_argument("--barrier-after-load", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    dtype = getattr(torch, args.dtype)
    torch.cuda.reset_peak_memory_stats()
    load_started = time.perf_counter()
    model = Qwen3TTSModel.from_pretrained(
        str(args.model),
        device_map="cuda:0",
        dtype=dtype,
        attn_implementation=args.attention,
    )
    torch.cuda.synchronize()
    after_load = {
        "event": "native_model_loaded",
        "pid": __import__("os").getpid(),
        "load_ms": (time.perf_counter() - load_started) * 1000.0,
        "cuda": cuda_snapshot(),
    }
    print(json.dumps(after_load), flush=True)
    if args.barrier_after_load:
        input("Press Enter to generate... ")

    generation_started = time.perf_counter()
    wavs, sample_rate = model.generate_custom_voice(
        text=args.text,
        language=args.language,
        speaker=args.speaker,
        instruct=args.instructions,
        max_new_tokens=args.max_new_tokens,
    )
    torch.cuda.synchronize()
    generation_ms = (time.perf_counter() - generation_started) * 1000.0
    audio_seconds = float(len(wavs[0])) / float(sample_rate)
    result = {
        "schema_version": 1,
        "runtime": {
            "python_executable": sys.executable,
            "python_version": sys.version,
            "platform": platform.platform(),
            "torch_version": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "gpu_capability": list(torch.cuda.get_device_capability()),
            "model": str(args.model),
            "dtype": args.dtype,
            "attention": args.attention,
        },
        "request": {
            "text": args.text,
            "speaker": args.speaker,
            "language": args.language,
            "instructions": args.instructions,
            "max_new_tokens": args.max_new_tokens,
        },
        "load": after_load,
        "generation": {
            "generation_ms": generation_ms,
            "sample_rate": sample_rate,
            "audio_samples": len(wavs[0]),
            "audio_seconds": audio_seconds,
            "rtf": generation_ms / 1000.0 / audio_seconds,
            "cuda": cuda_snapshot(),
            "streaming": False,
        },
        "notes": [
            "The official qwen-tts wrapper returns a complete waveform; this is a memory floor, not a streaming production candidate.",
            "CUDA allocator figures do not include every driver/library allocation; pair with measure-jetson-memory.py.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(result, indent=2, ensure_ascii=False)
    args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered, flush=True)


if __name__ == "__main__":
    main()
