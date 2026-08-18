#!/usr/bin/env python3
"""Benchmark the official Qwen3-ASR vLLM backend with synchronized timing."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
import wave
from pathlib import Path
from typing import Any


def _register_qwen3_asr() -> None:
    """Register Qwen's custom classes in both parent and spawned workers."""
    from qwen_asr.core.transformers_backend import (
        Qwen3ASRConfig,
        Qwen3ASRForConditionalGeneration as TransformersQwen3ASR,
        Qwen3ASRProcessor,
    )
    from qwen_asr.core.vllm_backend import (
        Qwen3ASRForConditionalGeneration as VllmQwen3ASR,
    )
    from transformers import AutoConfig, AutoModel, AutoProcessor
    from vllm import ModelRegistry

    AutoConfig.register("qwen3_asr", Qwen3ASRConfig, exist_ok=True)
    AutoModel.register(Qwen3ASRConfig, TransformersQwen3ASR, exist_ok=True)
    AutoProcessor.register(Qwen3ASRConfig, Qwen3ASRProcessor, exist_ok=True)
    ModelRegistry.register_model("Qwen3ASRForConditionalGeneration", VllmQwen3ASR)


# vLLM uses multiprocessing spawn. Module-level registration mirrors Qwen's
# official qwen-asr-serve entry point and therefore also runs in engine workers.
_register_qwen3_asr()


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as audio:
        return audio.getnframes() / audio.getframerate()


def _write_outputs(prefix: Path, payload: dict[str, Any]) -> None:
    prefix.parent.mkdir(parents=True, exist_ok=True)
    prefix.with_suffix(".json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    with prefix.with_suffix(".runs.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "run",
                "latency_ms",
                "rtf",
                "generated_tokens",
                "tokens_per_second",
                "text",
            ],
        )
        writer.writeheader()
        writer.writerows(payload["runs"])

    summary = payload["summary"]
    prefix.with_suffix(".md").write_text(
        "\n".join(
            [
                f"# {payload['title']}",
                "",
                f"- Audio: `{payload['audio']}` ({payload['audio_seconds']:.3f} s)",
                f"- Model: `{payload['model']}`",
                f"- dtype: `{payload['dtype']}`",
                f"- enforce_eager: `{payload['enforce_eager']}`",
                f"- Engine initialization: {payload['load_ms']:.1f} ms",
                f"- Warm-up: {payload['warmup']}; measured runs: {payload['measured_runs']}",
                "",
                "| Mean ms | Median ms | P95 ms | RTF | Tokens/s | Tokens |",
                "|---:|---:|---:|---:|---:|---:|",
                (
                    f"| {summary['mean_ms']:.3f} | {summary['median_ms']:.3f} | "
                    f"{summary['p95_ms']:.3f} | {summary['rtf']:.6f} | "
                    f"{summary['tokens_per_second']:.3f} | {summary['generated_tokens']} |"
                ),
                "",
                f"Transcript: `{summary['text']}`",
                "",
            ]
        ),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--audio", required=True, type=Path)
    parser.add_argument("--language", default="Chinese")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.7)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--mm-encoder-attn-backend", default="TORCH_SDPA")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--output-prefix", required=True, type=Path)
    parser.add_argument("--title", default="Qwen3-ASR vLLM benchmark")
    args = parser.parse_args()

    import torch
    from qwen_asr import Qwen3ASRModel

    audio = args.audio.resolve()
    audio_seconds = _wav_duration(audio)
    load_started = time.perf_counter()
    asr = Qwen3ASRModel.LLM(
        model=str(Path(args.model).resolve()),
        dtype=args.dtype,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_inference_batch_size=1,
        max_new_tokens=args.max_new_tokens,
        enforce_eager=args.enforce_eager,
        mm_encoder_attn_backend=args.mm_encoder_attn_backend,
    )
    torch.cuda.synchronize()
    load_ms = (time.perf_counter() - load_started) * 1000.0

    def transcribe() -> tuple[str, int, float]:
        torch.cuda.synchronize()
        started = time.perf_counter()
        result = asr.transcribe(
            audio=[str(audio)], language=[args.language], return_time_stamps=False
        )[0]
        torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - started) * 1000.0
        text = result.text
        generated_tokens = len(
            asr.processor.tokenizer.encode(text, add_special_tokens=False)
        )
        return text, generated_tokens, latency_ms

    warmup_ms = []
    for _ in range(args.warmup):
        _, _, latency_ms = transcribe()
        warmup_ms.append(latency_ms)

    runs = []
    for index in range(1, args.runs + 1):
        text, generated_tokens, latency_ms = transcribe()
        runs.append(
            {
                "run": index,
                "latency_ms": latency_ms,
                "rtf": latency_ms / 1000.0 / audio_seconds,
                "generated_tokens": generated_tokens,
                "tokens_per_second": generated_tokens / (latency_ms / 1000.0),
                "text": text,
            }
        )

    texts = {run["text"] for run in runs}
    token_counts = {run["generated_tokens"] for run in runs}
    if len(texts) != 1 or len(token_counts) != 1:
        raise RuntimeError("vLLM output was not deterministic across measured runs")
    latencies = [run["latency_ms"] for run in runs]
    token_count = runs[0]["generated_tokens"]
    mean_ms = statistics.fmean(latencies)
    payload = {
        "title": args.title,
        "model": str(Path(args.model).resolve()),
        "audio": str(audio),
        "audio_seconds": audio_seconds,
        "dtype": args.dtype,
        "enforce_eager": args.enforce_eager,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_model_len": args.max_model_len,
        "max_new_tokens": args.max_new_tokens,
        "mm_encoder_attn_backend": args.mm_encoder_attn_backend,
        "load_ms": load_ms,
        "warmup": args.warmup,
        "warmup_ms": warmup_ms,
        "measured_runs": args.runs,
        "summary": {
            "mean_ms": mean_ms,
            "median_ms": statistics.median(latencies),
            "p95_ms": _percentile(latencies, 0.95),
            "rtf": mean_ms / 1000.0 / audio_seconds,
            "generated_tokens": token_count,
            "tokens_per_second": token_count / (mean_ms / 1000.0),
            "text": runs[0]["text"],
        },
        "runs": runs,
    }
    _write_outputs(args.output_prefix, payload)
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
