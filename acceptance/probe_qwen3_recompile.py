"""Probe dynamic-shape requests for compile/recompile/graph fallback evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from g1_speech.asr import create_asr_engine
from g1_speech.config import load_config
from g1_speech.contracts import Utterance
from g1_speech.qwen3_asr import Qwen3AsrEngine

from qwen3_benchmarking import load_audio


def _counters(values: defaultdict[str, Counter]) -> dict[str, dict[str, int]]:
    return {
        group: {str(name): int(count) for name, count in counters.items() if count}
        for group, counters in values.items()
        if any(counters.values())
    }


def _compiled_state(model: Any) -> dict[str, Any]:
    names = (
        "_compiled_call",
        "_compiled_call_impl",
        "_compiled_forward",
        "_generation_compile_config",
    )
    return {
        name: {
            "present": hasattr(model, name),
            "is_none": getattr(model, name, None) is None,
            "type": (
                f"{type(getattr(model, name)).__module__}."
                f"{type(getattr(model, name)).__qualname__}"
                if getattr(model, name, None) is not None
                else None
            ),
        }
        for name in names
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--audio", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    import torch

    config = load_config(args.config)
    inputs = []
    for path in args.audio:
        samples, sample_rate = load_audio(path, config.audio.sample_rate)
        inputs.append((Path(path).name, samples, sample_rate))
    engine = create_asr_engine(config)
    if not isinstance(engine, Qwen3AsrEngine):
        raise TypeError("this probe requires qwen3_asr")
    engine.load()
    engine.warmup()
    torch._dynamo.utils.counters.clear()
    rows = []
    sequence = inputs + list(reversed(inputs))
    try:
        for index, (name, samples, sample_rate) in enumerate(sequence, 1):
            before = _counters(torch._dynamo.utils.counters)
            started = time.perf_counter()
            result, profile = engine.transcribe_profiled(
                Utterance(samples, sample_rate, 0, 0)
            )
            wall_ms = (time.perf_counter() - started) * 1000
            after = _counters(torch._dynamo.utils.counters)
            rows.append(
                {
                    "sequence": index,
                    "audio": name,
                    "audio_sample_count": int(samples.size),
                    "audio_seconds": samples.size / sample_rate,
                    "text_sha256": hashlib.sha256(
                        result.text.encode("utf-8")
                    ).hexdigest(),
                    "text_characters": len(result.text),
                    "wall_ms": wall_ms,
                    "profile": profile.as_dict(),
                    "dynamo_counters_before": before,
                    "dynamo_counters_after": after,
                    "compiled_state": _compiled_state(engine._model),  # noqa: SLF001
                }
            )
    finally:
        engine.close()
    payload = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "torch_logs": __import__("os").environ.get("TORCH_LOGS"),
        "compile_dynamic": config.qwen3_asr.compile_dynamic,
        "cache_implementation": config.qwen3_asr.cache_implementation,
        "rows": rows,
        "final_dynamo_counters": _counters(torch._dynamo.utils.counters),
    }
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
