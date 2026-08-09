# Orin NX Qwen3-ASR benchmark evidence

This directory is the saved evidence for `docs/qwen3_asr_orin_nx_latency.md`.
JSON files contain run-level samples and summary statistics; `*.runs.csv` contains
one row per measured request; `*.summary.csv` and `*.md` are exact lookup views.

## Protocol

- Device: Jetson Orin NX 16GB, Jetson Linux 36.4.3 / JetPack 6.2, CUDA 12.6,
  compute capability 8.7.
- Runtime: Jetson-native PyTorch `2.5.0a0+872d972e41.nv24.08` and Transformers 5.13.
- Model: Qwen3-ASR-0.6B, GPU SDPA, greedy decoding, `max_new_tokens=256`.
- Profiling precision: BF16; FP16 is tested as a separate variant. FP16 was later
  promoted to the stable Orin baseline after the labeled CER gate documented in
  [`../orin_nx_2026-08-10_fp16_baseline`](../orin_nx_2026-08-10_fp16_baseline/README.md).
- Warm-up: 3 requests. Formal measurements: 10 requests, except no reduced run
  count was needed for the controlled 8× scan. The provided 120-second M4A uses
  5 formal requests as allowed by the protocol.
- Timing: wall-clock `perf_counter`; every processor/H2D/generate/decode boundary
  synchronizes CUDA. Compile/cold requests are excluded from steady-state statistics.
- Power mode: `MAXN_SUPER`, dynamic clocks. Clock locking was not used because it
  was non-essential and required a privileged system change.
- Timezone: Asia/Shanghai. Data collected 2026-08-09.

The controlled scaling scan tiles the same 5.592-second PCM sample in memory. The
provided-audio scan decodes the four local M4A inputs to mono 16 kHz float32 using
ffmpeg. The M4A files remain host-local and are intentionally ignored by Git.

## Reproduction commands

```bash
export PYTHONPATH=src

.venv-gpu/bin/python acceptance/benchmark_qwen3_profile.py \
  --config config.qwen3-asr-0.6b.local.json \
  --audio models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17/test_wavs/zh.wav \
  --factors 1,2,4,8 --warmup 3 --runs 10 \
  --output-prefix benchmarks/orin_nx_2026-08-09/qwen3_asr_bf16_sdpa_controlled_scaling

.venv-gpu/bin/python acceptance/benchmark_qwen3_profile.py \
  --config config.qwen3-asr-0.6b.local.json \
  --audio tests/test_audio_6s.m4a \
  --audio tests/test_audio_30s.m4a \
  --audio tests/test_audio_60s.m4a \
  --audio tests/test_audio_120s.m4a \
  --warmup 3 --runs 10 --long-runs 5 \
  --output-prefix benchmarks/orin_nx_2026-08-09/qwen3_asr_bf16_sdpa_natural_length_scan
```

The failed optimization probes used local configs derived without modifying the
baseline. Reproduce either variant with `g1-speech config init`, then run the same
benchmark command shown above with the derived config:

```bash
.venv/bin/g1-speech config init \
  --output config.qwen3-asr-compile.local.json \
  --base config.qwen3-asr-0.6b.local.json \
  --set qwen3_asr.compile=true \
  --set qwen3_asr.compile_mode=reduce-overhead

.venv/bin/g1-speech config init \
  --output config.qwen3-asr-static-cache.local.json \
  --base config.qwen3-asr-0.6b.local.json \
  --set qwen3_asr.cache_implementation=static
```

`qwen3_asr_bf16_sdpa_torch_compile.failure.txt` and
`qwen3_asr_bf16_sdpa_static_cache.failure.txt` are complete, sanitized failure
traces. Nsight `.nsys-rep` and SQLite intermediates were kept in `/tmp`; the compact
CSV summaries needed to audit the conclusions are committed here.

No chart is included: the requested comparisons have four or fewer primary rows,
and exact tables are clearer than a plotted trend at this sample size.
