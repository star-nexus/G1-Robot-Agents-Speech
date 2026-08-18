"""Acceptance capture of one post-warmup Qwen3-ASR profiler trace."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from g1_speech.asr import create_asr_engine
from g1_speech.config import load_config
from g1_speech.contracts import Utterance
from g1_speech.qwen3_asr import Qwen3AsrEngine

from qwen3_benchmarking import load_audio


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--audio", required=True)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--export-trace", action="store_true")
    args = parser.parse_args()

    import torch

    config = load_config(args.config)
    samples, sample_rate = load_audio(args.audio, config.audio.sample_rate)
    utterance = Utterance(samples, sample_rate, 0, 0)
    engine = create_asr_engine(config)
    if not isinstance(engine, Qwen3AsrEngine):
        raise TypeError("this profiler requires the qwen3_asr backend")

    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    engine.load()
    try:
        for _ in range(args.warmup):
            engine.transcribe(utterance)
        torch.cuda.reset_peak_memory_stats()
        activities = [torch.profiler.ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        with torch.profiler.profile(
            activities=activities,
            record_shapes=True,
            profile_memory=True,
            with_stack=False,
        ) as profiler:
            result, profile = engine.transcribe_profiled(utterance, record_ranges=True)

        cpu_table = profiler.key_averages().table(
            sort_by="self_cpu_time_total", row_limit=40
        )
        cuda_table = profiler.key_averages().table(
            sort_by="self_cuda_time_total", row_limit=40
        )
        events = list(profiler.events())
        launch_events = [
            event
            for event in events
            if "cudaLaunch" in event.name or "cuLaunchKernel" in event.name
        ]
        cuda_events = [
            event
            for event in events
            if str(getattr(event, "device_type", "")).lower().endswith("cuda")
        ]
        payload = {
            "generated_at": datetime.now().astimezone().isoformat(),
            "audio": Path(args.audio).name,
            "profile": profile.as_dict(),
            "text": result.text,
            "event_count": len(events),
            "cuda_event_count": len(cuda_events),
            "kernel_launch_event_count": len(launch_events),
            "peak_memory_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_memory_reserved_bytes": torch.cuda.max_memory_reserved(),
            "trace_exported": args.export_trace,
        }
        prefix.with_suffix(".json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        prefix.with_suffix(".txt").write_text(
            "CPU time top ops\n================\n"
            + cpu_table
            + "\n\nCUDA time top ops\n=================\n"
            + cuda_table
            + "\n",
            encoding="utf-8",
        )
        if args.export_trace:
            profiler.export_chrome_trace(str(prefix.with_suffix(".trace.json")))
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    finally:
        engine.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
