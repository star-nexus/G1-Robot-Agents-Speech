"""Compare latency, RTF, and corpus CER for multiple configured ASR adapters."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

from g1_speech.asr import create_asr_engine
from g1_speech.cli import _read_wav
from g1_speech.config import load_config
from g1_speech.contracts import Utterance
from g1_speech.evaluation import edit_distance, normalize_characters


@dataclass(frozen=True)
class Case:
    audio: Path
    text: str
    language: str | None


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def load_manifest(path: Path) -> list[Case]:
    cases: list[Case] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        try:
            raw = json.loads(line)
            audio = Path(raw["audio"]).expanduser()
            if not audio.is_absolute():
                audio = (path.parent / audio).resolve()
            text = str(raw["text"])
            if not text:
                raise ValueError("reference text must not be empty")
            cases.append(Case(audio=audio, text=text, language=raw.get("language")))
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid manifest line {line_number}: {line}") from exc
    if not cases:
        raise ValueError("manifest contains no cases")
    return cases


def parse_config(value: str) -> tuple[str, Path]:
    label, separator, encoded_path = value.partition("=")
    if not separator or not label or not encoded_path:
        raise argparse.ArgumentTypeError("use LABEL=CONFIG.json")
    return label, Path(encoded_path).expanduser().resolve()


def benchmark(label: str, config_path: Path, cases: list[Case], warmup: int, runs: int):
    config = load_config(config_path)
    engine = create_asr_engine(config)
    times: list[float] = []
    audio_ms_total = 0.0
    reference_characters: list[str] = []
    hypothesis_characters: list[str] = []
    transcripts = []
    try:
        started = time.perf_counter()
        engine.load()
        load_ms = (time.perf_counter() - started) * 1000
        for case_index, case in enumerate(cases):
            samples, sample_rate = _read_wav(str(case.audio))
            if sample_rate != config.audio.sample_rate:
                raise ValueError(
                    f"{case.audio} is {sample_rate} Hz; expected {config.audio.sample_rate} Hz"
                )
            utterance = Utterance(samples, sample_rate, 0, 0)
            if case_index == 0:
                for _ in range(warmup):
                    engine.transcribe(utterance)
            results = [engine.transcribe(utterance) for _ in range(runs)]
            result = results[0]
            times.extend(item.inference_ms for item in results)
            audio_ms_total += utterance.duration_ms * runs
            reference_characters.extend(normalize_characters(case.text))
            hypothesis_characters.extend(normalize_characters(result.text))
            transcripts.append(
                {
                    "audio": str(case.audio),
                    "reference": case.text,
                    "expected_language": case.language,
                    "text": result.text,
                    "language": result.language,
                }
            )
        return {
            "label": label,
            "backend": config.asr.backend,
            "engine": transcripts and result.engine,
            "cases": len(cases),
            "runs_per_case": runs,
            "load_ms": round(load_ms, 1),
            "mean_ms": round(statistics.fmean(times), 1),
            "median_ms": round(statistics.median(times), 1),
            "p95_ms": round(percentile(times, 0.95), 1),
            "rtf": round(sum(times) / audio_ms_total, 4),
            "cer": round(
                edit_distance(reference_characters, hypothesis_characters)
                / len(reference_characters),
                4,
            ),
            "transcripts": transcripts,
        }
    finally:
        engine.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", action="append", required=True, type=parse_config)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--runs", type=int, default=5)
    args = parser.parse_args()
    if args.warmup < 0 or args.runs < 1:
        raise ValueError("warmup must be >= 0 and runs must be >= 1")
    cases = load_manifest(args.manifest.expanduser().resolve())
    payload = [
        benchmark(label, path, cases, args.warmup, args.runs)
        for label, path in args.config
    ]
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
