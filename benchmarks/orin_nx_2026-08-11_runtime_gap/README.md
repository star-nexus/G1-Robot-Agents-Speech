# Qwen3-ASR E2E runtime-gap attribution evidence

This directory contains the raw and summarized evidence for
[`docs/qwen3_asr_e2e_runtime_gap.md`](../../docs/qwen3_asr_e2e_runtime_gap.md).

## Inventory

| Artifact | Purpose |
|---|---|
| `benchmark.json` | Complete same-process A/B, profile-overhead A/B, token scaling, shapes, regression |
| `runtime_fingerprint.json` | Exact executable, packages, model/config, compile settings, environment |
| `runtime_comparison.json` | Historical, contemporary fixed-shape, and E2E runtime comparison |
| `same_input_ab.runs.csv` / `.summary.csv` | Per-run and summarized exact-PCM path comparison |
| `profiling_overhead_ab.runs.csv` | Interleaved synchronized profiling off/on measurements |
| `token_scaling.runs.csv` / `.summary.csv` | Same-audio 5/10/15/20/30/50 Token measurements |
| `shape_trace.json` | Diagnostic-only prefill/decode input shapes; excluded from timing claims |
| `recompile_probe.json` / `.log` | Dynamic shape counters and raw `TORCH_LOGS` output |
| `controlled_static_shape.*` | Contemporary fixed-shape 10-run control |
| `e2e_dynamic_nsys_*` | Warm E2E scoped Nsight request plus raw summary CSV |
| `nsight_summary.csv` | Controlled-vs-E2E launch structure comparison |
| `extracted_audio_manifest.json` | Local extracted WAV hashes/sample counts; audio intentionally not committed |

No benchmark failure file exists because all requested benchmark processes completed.
The recompile log is retained even though it contains no recompile/graph-break entry;
that negative result is evidence. Nsight `.nsys-rep`/SQLite and extracted WAV files
were removed after CSV/hash extraction because they are reproducible binary/local-audio
artifacts and are not required by the requested repository delivery.
Host-local transcripts in JSON/CSV are represented by SHA-256 and character count;
the benchmark checks hash determinism for every formal run group, while the committed
evidence does not publish the user's local speech content.

The legacy raw field `generated_tokens_per_second` means **request-level apparent
generation throughput** (`generated_tokens / entire generate_ms`), not decoder speed.
Decoder capability is reported separately as **incremental decode throughput**, estimated
from the slope of the controlled token-scaling regression.

## Main attribution command

The original raw JSON retains the historical config filename in `argv`. Root-level
host configs were later consolidated; the equivalent current command uses the single
selected Qwen profile and backend-neutral settings from `deploy.env`:

```bash
set -a
source deploy.env
set +a
export PYTHONPATH="$PWD/src:$PWD/acceptance"
export LD_LIBRARY_PATH="$SPEECH_GPU_LIBRARY_PATH${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

"$SPEECH_PYTHON_GPU" acceptance/benchmark_qwen3_runtime_gap.py \
  --config "$SPEECH_CONFIG_GPU" \
  --controlled-audio models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/test_wavs/zh.wav \
  --real-audio tests/test_audio_6s.m4a \
  --scaling-audio tests/test_audio_30s.m4a \
  --output-dir benchmarks/orin_nx_2026-08-11_runtime_gap \
  --warmup 3 --runs 10 --scaling-warmup 2 --scaling-runs 10 \
  --token-targets 5,10,15,20,30,50
```

The static-shape control used a generated copy of the same config with only
`compile_dynamic=false` and `startup_warmup_seconds=0`, plus a fresh temporary
Inductor cache. The recompile probe used
`TORCH_LOGS=recompiles,graph_breaks`. Nsight used CUDA profiler API capture after
three actual-shape warm-ups; exact options are preserved in
`e2e_dynamic_nsys_request.json` and the report methodology.

## Source and presentation notes

Quantitative claims come from files in this directory and the historical baseline
directory `benchmarks/orin_nx_2026-08-10_runtime_optimizations`. No external source
was used. No chart is included: six exact Token anchors plus the regression formula
and R² are clearer as an audit table, and the summary CSV is ready for plotting if a
later report requires a visual.
