#!/usr/bin/env python3
"""Measure Qwen3-TTS WebSocket streaming latency without audio playback."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import websocket


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def run_request(ws: websocket.WebSocket, args: argparse.Namespace, text: str) -> dict[str, object]:
    session = {
        "type": "session.config",
        "model": args.model,
        "task_type": "CustomVoice",
        "voice": args.voice,
        "language": args.language,
        "instructions": args.instructions,
        "response_format": "pcm",
        "stream_audio": True,
        "max_new_tokens": args.max_new_tokens,
        "initial_codec_chunk_frames": args.initial_codec_chunk_frames,
    }
    started = time.perf_counter()
    ws.send(json.dumps(session))
    ws.send(json.dumps({"type": "input.text", "text": text}))
    ws.send(json.dumps({"type": "input.done"}))
    first_pcm_ms = None
    pcm_bytes = 0
    chunks = 0
    sample_rate = 24000
    while True:
        message = ws.recv()
        now = time.perf_counter()
        if isinstance(message, (bytes, bytearray)):
            if message:
                if first_pcm_ms is None:
                    first_pcm_ms = (now - started) * 1000.0
                pcm_bytes += len(message)
                chunks += 1
            continue
        event = json.loads(message)
        event_type = event.get("type")
        if event_type == "audio.start":
            sample_rate = int(event.get("sample_rate", sample_rate))
        elif event_type == "error":
            raise RuntimeError(event.get("message", event))
        elif event_type == "audio.done" and event.get("error"):
            raise RuntimeError(f"audio.done error: {event}")
        elif event_type == "session.done":
            break
    total_ms = (time.perf_counter() - started) * 1000.0
    audio_seconds = pcm_bytes / (sample_rate * 2.0)
    return {
        "text": text,
        "first_pcm_ms": first_pcm_ms,
        "total_ms": total_ms,
        "pcm_bytes": pcm_bytes,
        "chunks": chunks,
        "sample_rate": sample_rate,
        "audio_seconds": audio_seconds,
        "audio_rtf": total_ms / 1000.0 / audio_seconds if audio_seconds else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="ws://127.0.0.1:8091/v1/audio/speech/stream")
    parser.add_argument("--model", required=True)
    parser.add_argument("--voice", default="Ryan")
    parser.add_argument("--language", default="English")
    parser.add_argument("--instructions", default="Speak clearly and warmly.")
    parser.add_argument("--text", action="append", default=[])
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--initial-codec-chunk-frames", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    texts = args.text or ["Welcome to STAR.", "How may I help you?"]

    ws = websocket.create_connection(args.url, timeout=args.timeout, enable_multithread=True)
    ws.settimeout(args.timeout)
    try:
        warmups = [run_request(ws, args, texts[index % len(texts)]) for index in range(args.warmup)]
        runs = [run_request(ws, args, texts[index % len(texts)]) for index in range(args.runs)]
    finally:
        ws.close()

    first_pcm = [float(run["first_pcm_ms"]) for run in runs if run["first_pcm_ms"] is not None]
    totals = [float(run["total_ms"]) for run in runs]
    rtfs = [float(run["audio_rtf"]) for run in runs if run["audio_rtf"] is not None]
    result = {
        "url": args.url,
        "model": args.model,
        "warmup": warmups,
        "runs": runs,
        "summary": {
            "first_pcm_mean_ms": statistics.mean(first_pcm),
            "first_pcm_median_ms": statistics.median(first_pcm),
            "first_pcm_p95_ms": percentile(first_pcm, 0.95),
            "total_mean_ms": statistics.mean(totals),
            "audio_rtf_mean": statistics.mean(rtfs),
            "audio_rtf_median": statistics.median(rtfs),
        },
    }
    rendered = json.dumps(result, indent=2, ensure_ascii=False)
    print(rendered)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
