"""Shared helpers for Qwen3-ASR acceptance experiments."""

from __future__ import annotations

import csv
import json
import statistics
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from g1_speech.cli import _read_wav


PROFILE_METRICS = (
    "processor_ms",
    "h2d_ms",
    "generate_ms",
    "decode_ms",
    "total_ms",
    "generated_tokens",
    "generated_tokens_per_second",
    "rtf",
)


def load_audio(path: str | Path, sample_rate: int = 16000) -> tuple[np.ndarray, int]:
    """Read WAV directly or decode another ffmpeg-supported format to mono float32."""
    source = Path(path)
    if source.suffix.lower() == ".wav":
        return _read_wav(str(source))
    # Jetson ffmpeg plugins can emit an EGL diagnostic on stdout and corrupt a
    # pipe:1 PCM stream. A temporary raw file keeps diagnostic text separate.
    with tempfile.TemporaryDirectory(prefix="g1-speech-audio-") as directory:
        decoded = Path(directory) / "decoded.f32"
        command = [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(source),
            "-f",
            "f32le",
            "-acodec",
            "pcm_f32le",
            "-ac",
            "1",
            "-ar",
            str(sample_rate),
            "-y",
            str(decoded),
        ]
        subprocess.run(
            command,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        samples = np.fromfile(decoded, dtype="<f4")
    if not samples.size:
        raise ValueError(f"decoded audio is empty: {source}")
    return samples, sample_rate


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = round((len(ordered) - 1) * fraction)
    return ordered[index]


def summarize_profiles(profiles: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for metric in PROFILE_METRICS:
        values = [float(profile[metric]) for profile in profiles]
        summary[metric] = {
            "mean": statistics.fmean(values),
            "median": statistics.median(values),
            "p95": percentile(values, 0.95),
            "min": min(values),
            "max": max(values),
        }
    return summary


def write_benchmark_outputs(payload: dict[str, Any], output_prefix: str | Path) -> None:
    prefix = Path(output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    prefix.with_suffix(".json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    summary_rows: list[dict[str, Any]] = []
    run_rows: list[dict[str, Any]] = []
    for case in payload["cases"]:
        row: dict[str, Any] = {
            "case": case["name"],
            "audio_seconds": case["audio_seconds"],
            "warmup": case["warmup"],
            "runs": case["runs"],
            "text": case["text"],
        }
        for metric, values in case["summary"].items():
            for statistic, value in values.items():
                row[f"{metric}_{statistic}"] = value
        summary_rows.append(row)
        for index, run in enumerate(case["measurements"], 1):
            run_rows.append({"case": case["name"], "run": index, **run})

    _write_csv(prefix.with_suffix(".summary.csv"), summary_rows)
    _write_csv(prefix.with_suffix(".runs.csv"), run_rows)
    prefix.with_suffix(".md").write_text(_markdown(payload), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        f"# {payload['title']}",
        "",
        f"Generated: {payload['generated_at']}",
        "",
        "| Case | Audio (s) | Mean (ms) | Median (ms) | P95 (ms) | RTF | Tokens | Tokens/s |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for case in payload["cases"]:
        summary = case["summary"]
        lines.append(
            "| {name} | {audio:.3f} | {mean:.1f} | {median:.1f} | {p95:.1f} | "
            "{rtf:.4f} | {tokens:.1f} | {tokens_s:.2f} |".format(
                name=case["name"],
                audio=case["audio_seconds"],
                mean=summary["total_ms"]["mean"],
                median=summary["total_ms"]["median"],
                p95=summary["total_ms"]["p95"],
                rtf=summary["rtf"]["mean"],
                tokens=summary["generated_tokens"]["mean"],
                tokens_s=summary["generated_tokens_per_second"]["mean"],
            )
        )
    lines.extend(
        [
            "",
            "Times are synchronized wall-clock measurements. Tokens/s is generated "
            "tokens divided by `generate_ms`.",
            "",
        ]
    )
    return "\n".join(lines)
