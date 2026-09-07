#!/usr/bin/env python3
"""Repeated real-ALSA interruption/recovery check for JetPack Orin.

This isolates the PCM player from provider latency. It submits an old tone,
interrupts after hardware playback starts, then requires a replacement tone to
reach a successful PortAudio write within a bounded deadline. Run both
interrupt strategies to obtain directly comparable device evidence.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import platform
import statistics
import time
from pathlib import Path

import numpy as np

from star_runtime.speech.audio.output import AlsaOutputPlayer
from star_runtime.speech.audio.processing import PassthroughAudioProcessor
from star_runtime.speech.config import AudioOutputConfig


def tone(frequency: float, duration_ms: int, sample_rate: int) -> bytes:
    samples = round(sample_rate * duration_ms / 1000)
    phase = np.arange(samples, dtype=np.float64) * (2 * math.pi * frequency)
    pcm = np.rint(np.sin(phase / sample_rate) * 800).astype("<i2")
    return pcm.tobytes()


def wait_for(predicate, description: str, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.002)
    raise TimeoutError(f"timed out waiting for {description}")


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil(len(ordered) * fraction) - 1)
    return round(ordered[index], 3)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--card", default="Audio")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--strategy",
        choices=("persistent", "hard_abort"),
        required=True,
    )
    parser.add_argument("--cycles", type=int, default=50)
    parser.add_argument("--timeout", type=float, default=2.0)
    parser.add_argument("--old-ms", type=int, default=200)
    parser.add_argument("--replacement-ms", type=int, default=40)
    args = parser.parse_args()
    if args.cycles < 1:
        parser.error("--cycles must be positive")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s.%(msecs)03d %(levelname)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    sample_rate = 48000
    player = AlsaOutputPlayer(
        AudioOutputConfig(
            alsa_card=args.card,
            alsa_device=args.device,
            sample_rate=sample_rate,
            channels=2,
            block_ms=10,
            latency="low",
            volume=1.0,
            buffer_seconds=0.5,
            interrupt_strategy=args.strategy,
        ),
        render_sink=PassthroughAudioProcessor("hardware"),
    )
    old_pcm = tone(440.0, args.old_ms, sample_rate)
    replacement_pcm = tone(880.0, args.replacement_ms, sample_rate)
    recovery_latencies: list[float] = []
    failures: list[dict[str, object]] = []
    started_ns = time.monotonic_ns()

    try:
        player.start()
        for cycle in range(1, args.cycles + 1):
            baseline_blocks = int(player.metrics()["audio_output_blocks"])
            if not player.enqueue(old_pcm, sample_rate):
                raise RuntimeError(f"cycle {cycle}: old PCM enqueue rejected")
            wait_for(
                lambda: int(player.metrics()["audio_output_blocks"])
                > baseline_blocks,
                f"cycle {cycle} old PCM hardware write",
                args.timeout,
            )

            baseline_recoveries = int(player.metrics()["audio_output_recoveries"])
            player.abort()
            player.note_pcm_ready()
            if not player.enqueue(replacement_pcm, sample_rate):
                raise RuntimeError(f"cycle {cycle}: replacement enqueue rejected")
            try:
                wait_for(
                    lambda: int(player.metrics()["audio_output_recoveries"])
                    > baseline_recoveries,
                    f"cycle {cycle} first replacement hardware write",
                    args.timeout,
                )
                if not player.wait_until_idle(timeout=args.timeout):
                    raise TimeoutError(
                        f"timed out draining cycle {cycle} replacement PCM"
                    )
            except Exception as exc:  # noqa: BLE001
                failures.append(
                    {
                        "cycle": cycle,
                        "error": str(exc),
                        "metrics": player.metrics(),
                    }
                )
                break

            metrics = player.metrics()
            latency = metrics["audio_output_last_recovery_ms"]
            if isinstance(latency, (int, float)):
                recovery_latencies.append(float(latency))
            logging.info(
                "PLAYBACK_STRESS_CYCLE cycle=%d/%d recovery_ms=%s "
                "underflows=%s restarts=%s aborts=%s",
                cycle,
                args.cycles,
                latency,
                metrics["audio_output_write_underflows"],
                metrics["audio_output_stream_restart_attempts"],
                metrics["audio_output_stream_abort_calls"],
            )

        metrics = player.metrics()
        invariant_errors = []
        if metrics["audio_output_forbidden_stale_write_attempts"] != 0:
            invariant_errors.append("stale PCM reached the hardware submission boundary")
        if metrics["audio_output_recoveries"] != args.cycles:
            invariant_errors.append("not every interruption recovered")
        if not metrics["audio_output_playback_thread_alive"]:
            invariant_errors.append("playback thread exited")
        if not metrics["audio_output_stream_active"]:
            invariant_errors.append("PortAudio stream ended inactive")
        if metrics["audio_output_write_errors"]:
            invariant_errors.append("PortAudio writes failed")
        if metrics["audio_output_stream_restart_errors"]:
            invariant_errors.append("PortAudio restart failed")

        release = "unknown"
        release_path = Path("/etc/nv_tegra_release")
        if release_path.exists():
            release = release_path.read_text(encoding="utf-8").strip()
        result = {
            "status": "pass" if not failures and not invariant_errors else "fail",
            "strategy": args.strategy,
            "requested_cycles": args.cycles,
            "completed_cycles": len(recovery_latencies),
            "duration_ms": round((time.monotonic_ns() - started_ns) / 1_000_000, 3),
            "recovery_latency_ms": {
                "mean": (
                    round(statistics.fmean(recovery_latencies), 3)
                    if recovery_latencies
                    else None
                ),
                "p50": percentile(recovery_latencies, 0.50),
                "p95": percentile(recovery_latencies, 0.95),
                "max": max(recovery_latencies, default=None),
            },
            "failures": failures,
            "invariant_errors": invariant_errors,
            "platform": platform.platform(),
            "jetson_release": release,
            "metrics": metrics,
        }
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0 if result["status"] == "pass" else 1
    finally:
        player.close()


if __name__ == "__main__":
    raise SystemExit(main())
