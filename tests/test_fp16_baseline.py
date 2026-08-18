from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).parents[1]
BASELINE_DIR = ROOT / "benchmarks/orin_nx_2026-08-10_fp16_baseline"


def test_sealed_fp16_baseline_passes_declared_aggregate_gate():
    baseline = json.loads(
        (BASELINE_DIR / "stable_baseline.json").read_text(encoding="utf-8")
    )
    gate = baseline["cer_gate"]

    assert baseline["status"] == "stable"
    assert baseline["configuration"]["dtype"] == "float16"
    assert gate["passed"] is True
    assert gate["fp16"]["content_cer"] - gate["bf16"]["content_cer"] <= gate[
        "maximum_content_cer_regression_absolute"
    ]
    assert gate["bf16"]["deterministic"] is True
    assert gate["fp16"]["deterministic"] is True
    assert gate["bf16"]["language_accuracy"] == 1.0
    assert gate["fp16"]["language_accuracy"] == 1.0


def test_orin_qwen_example_selects_the_sealed_configuration():
    baseline = json.loads(
        (BASELINE_DIR / "stable_baseline.json").read_text(encoding="utf-8")
    )
    example = json.loads(
        (ROOT / "configs/examples/config.qwen3-asr.example.json").read_text(
            encoding="utf-8"
        )
    )

    assert example["qwen3_asr"]["dtype"] == baseline["configuration"]["dtype"]
    assert (
        example["qwen3_asr"]["attention_implementation"]
        == baseline["configuration"]["attention_implementation"]
    )
    assert example["qwen3_asr"]["language"] == baseline["configuration"]["language"]


def test_community_baseline_excludes_exploratory_corpus_and_transcripts():
    assert not (BASELINE_DIR / "corpus_manifest.jsonl").exists()
    assert not (BASELINE_DIR / "qwen3_asr_bf16_vs_fp16_cer.json").exists()
