"""Run the real Orin service and fail on queue saturation or RSS growth."""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict

from g1_speech.app import SpeechService
from g1_speech.config import load_config


def rss_mb() -> float:
    statm = "/proc/self/statm"
    if os.path.exists(statm):
        pages = int(open(statm, encoding="utf-8").read().split()[1])
        return pages * os.sysconf("SC_PAGE_SIZE") / 1024 / 1024
    import resource

    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value / 1024 / 1024 if value > 10_000_000 else value / 1024


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seconds", type=float, default=3600)
    parser.add_argument("--sample-seconds", type=float, default=10)
    parser.add_argument("--max-rss-growth-mb", type=float, default=64)
    args = parser.parse_args()
    config = load_config(args.config)
    service = SpeechService(config)
    service.start()
    baseline_rss = rss_mb()
    maxima = {"audio": 0, "utterance": 0, "outbox": 0, "rss": baseline_rss}
    deadline = time.monotonic() + args.seconds
    try:
        while time.monotonic() < deadline:
            time.sleep(min(args.sample_seconds, max(0.1, deadline - time.monotonic())))
            metrics = service.pipeline.metrics()
            maxima["audio"] = max(maxima["audio"], metrics.audio_queue_size)
            maxima["utterance"] = max(maxima["utterance"], metrics.utterance_queue_size)
            maxima["outbox"] = max(maxima["outbox"], service.sink.queue_size)
            maxima["rss"] = max(maxima["rss"], rss_mb())
            print(json.dumps({**asdict(metrics), "outbox": service.sink.queue_size}))
    finally:
        final_metrics = service.pipeline.metrics()
        service.close()

    audio_capacity = max(
        2, round(config.audio.queue_seconds * 1000 / config.audio.block_ms)
    )
    rss_growth = maxima["rss"] - baseline_rss
    report = {
        "baseline_rss_mb": baseline_rss,
        "max_rss_mb": maxima["rss"],
        "rss_growth_mb": rss_growth,
        "max_audio_queue": maxima["audio"],
        "max_utterance_queue": maxima["utterance"],
        "max_dds_outbox": maxima["outbox"],
        "final": asdict(final_metrics),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    failed = (
        maxima["audio"] >= audio_capacity
        or maxima["utterance"] >= config.utterance_queue_capacity
        or maxima["outbox"] >= config.dds.outbox_capacity
        or rss_growth > args.max_rss_growth_mb
    )
    print("FAIL" if failed else "PASS")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
