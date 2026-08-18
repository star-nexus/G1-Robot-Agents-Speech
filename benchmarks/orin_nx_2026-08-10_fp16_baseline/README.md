# Qwen3-ASR Orin NX FP16 stable baseline

Status: **passed and sealed on 2026-08-10**.

The stable configuration for this Jetson Orin NX stack is Qwen3-ASR-0.6B, CUDA,
SDPA, FP16, greedy generation, no compile, and no static cache. FP16 passed the
accuracy-regression gate against the previous BF16 reference and retains the earlier
measured 4.9% fixed-sample latency improvement. The production example keeps the
Chinese language hint; this multilingual CER gate sets the hint to `null` only so
language identification is measured rather than supplied.

## CER gate

The local validation set contains 10 labeled utterances and 50.188 seconds across
three supported languages. Every configuration used three warm-up requests and three
formal runs per utterance. The exact audio, references, and hypotheses are
intentionally not distributed: the exploratory corpus is not suitable as a
community-facing benchmark. Only aggregate, non-content evidence is retained here.

`cer` is strict NFKC/casefold CER with whitespace removed. `content_cer` additionally
removes Unicode punctuation so optional commas and sentence punctuation do not look
like lexical errors. It does not convert traditional and simplified Chinese.

| Metric | BF16 | FP16 | Gate |
|---|---:|---:|---:|
| Content CER | 6.22% (13/209) | **6.22% (13/209)** | FP16 regression ≤ 0.5 pp |
| Strict CER | 14.22% (30/211) | **13.27% (28/211)** | Informational |
| Content transcript match | — | **10/10 vs BF16** | 10/10 |
| Language identification | 10/10 | **10/10** | 10/10 |
| Deterministic over 3 runs | 10/10 | **10/10** | 10/10 |

The strict-CER difference is two optional commas emitted only by BF16. After removing
punctuation, all FP16 and BF16 content transcripts are identical. The FP16 content-CER
regression is therefore exactly zero and the gate passes.

This is a precision-regression gate, not a claim that 10 utterances estimate general
production CER. Robot-domain vocabulary, far-field speech, noise, accents, and a
larger Mandarin set still require a separately curated human-labeled corpus.

## Frozen evidence

- `stable_baseline.json`: machine-readable decision, configuration, thresholds,
  runtime versions, model hashes, and aggregate metrics.
- Corpus manifests and transcript-level raw JSON are deliberately excluded. They must
  not be reconstructed from this aggregate result or treated as a public benchmark.

Model fingerprint:

| File | SHA-256 |
|---|---|
| `model.safetensors` | `d3f212dd20abecd315d830bc54ae3865e56ebfc3276484e57b771288ba27fd35` |
| `config.json` | `9eecf6f1b383e343889c2e6010e632590fa57d4bc678e151c7d6a160a0dfb04a` |
| `processor_config.json` | `bc0b230081b44e629dd5b9045b78495615c1831b4b9f4cffe97bd37e82a6156a` |

Runtime: JetPack 6.2, CUDA 12.6, SM 8.7, Jetson PyTorch
`2.5.0a0+872d972e41.nv24.08`, and Transformers `5.13.0`.

## Repeat the gate with a community-appropriate corpus

Prepare a reviewed, redistributable local JSONL manifest using
`acceptance/asr_manifest.example.jsonl` as the schema. Do not publish private,
offensive, copyrighted, or otherwise unsuitable speech content. Create two host-local
configs from the Qwen example, set the actual model directory, and disable the
language hint if language identification is part of the gate:

```bash
.venv/bin/g1-speech config init \
  --base config.qwen3-asr.local.json \
  --output /tmp/qwen3-asr-cer-fp16.json \
  --set qwen3_asr.dtype=float16 \
  --set qwen3_asr.language=null

.venv/bin/g1-speech config init \
  --base config.qwen3-asr.local.json \
  --output /tmp/qwen3-asr-cer-bf16.json \
  --set qwen3_asr.dtype=bfloat16 \
  --set qwen3_asr.language=null

PYTHONPATH=src .venv-gpu/bin/python acceptance/compare_asr.py \
  --config bf16=/tmp/qwen3-asr-cer-bf16.json \
  --config fp16=/tmp/qwen3-asr-cer-fp16.json \
  --manifest acceptance/asr_manifest.local.jsonl \
  --warmup 3 --runs 3
```
