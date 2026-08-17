# Qwen3-TTS single-user memory profile on Jetson Orin NX

## Executive conclusion

For a single robot generating one short reply at a time, the largest removable
allocation was not the 0.6B model itself. It was Stage 0's utilization-based KV
cache: the previous profile reserved **1.96 GiB / 18,352 tokens** even though the
largest measured request used only **181 Stage-0 tokens**.

The recommended profile keeps the validated
`TRITON_ATTN + FULL_AND_PIECEWISE + Stage-1 eager` runtime, but gives Stage 0 an
explicit **64 MiB KV cache (576 tokens)** and sizes the two stages for the
observed short-reply workload. On this Orin NX, it reduced incremental TTS
memory by about **1.98 GiB at idle** and **2.17 GiB at sampled workload peak**,
without degrading steady streaming latency:

| Result | Previous profile | Single-user profile | Change |
|---|---:|---:|---:|
| Incremental system memory, idle | 6.68 GiB | 4.71 GiB | **-1.98 GiB** |
| Incremental system memory, post-workload / sampled peak | 8.28 GiB | 6.11 GiB | **-2.17 GiB** |
| Stage-0 Jetson GPU mapping, post-workload | 4.16 GiB | 2.22 GiB | **-1.94 GiB** |
| First PCM, steady mean | 239.3 ms | 231.6 ms | -7.7 ms |
| First PCM, steady p95 | 256.0 ms | 276.4 ms | +20.4 ms |
| Streaming RTF, steady mean | 0.717 | 0.712 | -0.005 |

All five steady-state requests delivered first PCM in under 300 ms, used
multiple WebSocket PCM chunks, and completed below RTF 1. Cold compilation is
reported separately and is not hidden in the steady-state figures.

The production profile is
[`profiles/qwen3-tts-orin-nx.yaml`](../profiles/qwen3-tts-orin-nx.yaml). The exact
pre-change rollback profile is preserved at
[`benchmarks/orin_nx_2026-08-17_tts_memory/profiles/baseline_1024_65536.yaml`](../benchmarks/orin_nx_2026-08-17_tts_memory/profiles/baseline_1024_65536.yaml).

## Measurement method

The experiment ran on JetPack 6.2 / CUDA 12.6 / SM87 with Qwen3-TTS
12Hz-0.6B-CustomVoice and vLLM-Omni 0.26.0. TTS concurrency was one. The ASR
service remained running in both the TTS-off reference and TTS-on measurements.

Three views are retained because Jetson uses unified physical memory:

- `MemTotal - MemAvailable` measures whole-system pressure. Subtracting the
  TTS-off reference gives the most useful product-level incremental figure.
- Process PSS helps attribute host process overhead without double-counting
  shared pages.
- jtop's per-process GPU mapping identifies Stage allocations, but overlaps
  system memory and PSS. **These columns must not be added together.**

The repeatable capture utility is
[`scripts/measure-jetson-memory.py`](../scripts/measure-jetson-memory.py). Raw
snapshots, workload results, profiles, and the summary table are under
[`benchmarks/orin_nx_2026-08-17_tts_memory/`](../benchmarks/orin_nx_2026-08-17_tts_memory/).

The TTS-off reference was 5,839,288 KiB system memory used before the experiment
and 5,720,816 KiB afterward. The 116 MiB drift is small relative to the roughly
2 GiB measured saving, but it is still a limitation of whole-system sampling.

## Short-reply workload envelope

The corpus covers 2, 8, 11, 22, and 31 Chinese characters. Because sampling is
stochastic, bounds use the maximum observed value across both profiles instead
of choosing the more favorable run.

| Workload maximum | Observed | Production bound | Headroom |
|---|---:|---:|---:|
| Stage-0 input tokens | 47 | — | — |
| Stage-0 semantic output tokens | 134 | request `max_new_tokens=256` | 1.91x output |
| Stage-0 total sequence tokens | 181 | `max_model_len=384` | 2.12x sequence |
| Stage-0 KV capacity | 181 required | 576 tokens / 64 MiB | 3.18x observed |
| Stage-1 flattened codec tokens | about 2,144 | `max_model_len=4096` | 1.91x observed |
| Generated audio | 10.64 s | — | — |

The complete per-prompt values are in
[`workload_envelope.json`](../benchmarks/orin_nx_2026-08-17_tts_memory/workload_envelope.json).
The 31-character sample is deliberately just beyond the nominal 30-character
product limit.

## Where the memory went

### Model and runtime memory that remains

- Native `qwen-tts` loaded the full model at about **2.01 GiB CUDA allocated**
  and **2.13 GiB CUDA reserved**. This is a useful lower bound for weights and
  the native PyTorch runtime, not a production serving number.
- vLLM-Omni uses an API process, a multiprocessing resource tracker, a Stage-0
  engine, and a Stage-1 engine. In the optimized idle snapshot their PSS values
  were approximately 1.10 GiB, 0.38 GiB, 1.66 GiB, and 0.93 GiB respectively.
  This includes Python/framework code, host tensors, IPC state, model metadata,
  and process-private pages. It is real serving overhead, but PSS does not
  cleanly separate CUDA unified-memory mappings.
- Stage 0 reports about **1.91 GiB model weights** and Stage 1 about **0.43 GiB
  model weights**. Both stages are required by the current semantic-token to
  waveform streaming architecture.
- The continuous PCM ring buffer is only about 0.37 MiB for eight seconds of
  mono 24 kHz signed-16 PCM. It is not a meaningful contributor.

### Serving redundancy removed

The previous profile specified only `gpu_memory_utilization: 0.23`. vLLM filled
the remaining budget with a 1.96 GiB Stage-0 KV cache, enough for 18,352 tokens
or 17.92 concurrent 1,024-token requests. That is general serving capacity, not
useful capacity for a concurrency-one robot.

The new profile keeps the utilization safety guard but also specifies
`kv_cache_memory_bytes: 67108864`. Startup confirmed an actual capacity of 576
tokens. The roughly 1.94 GiB reduction in Stage-0 GPU mapping accounts for
almost all of the repeatable whole-system saving.

The CUDA Graph configuration remains enabled. Its reported graph memory fell
from about 0.05 GiB to about 0.01–0.02 GiB as the captured/request bounds became
smaller, but this is a minor secondary saving. Removing graphs would conflict
with the latency baseline and is not recommended.

### Changes that do not materially save memory

- Stage 1 reports that it has no KV cache. Repeating the optimized Stage-0 test
  with Stage-1 bounds at 65,536 instead of 4,096 changed idle system memory by
  only about 0.1 GiB, with PSS moving in the opposite direction. This is
  process/cache noise, not evidence of a large preallocation. The 4,096 bound is
  retained as a scheduler/correctness guard, not claimed as a headline saving.
- The process-wide speaker cache has a 512 MiB **limit**, not a 512 MiB eager
  allocation. Likewise, the Stage-1 reference-context cache has a 64 MiB limit
  and grows only when entries are inserted. Lowering these caps for the preset
  CustomVoice workload would not reduce the measured idle baseline.
- Lowering the PCM ring capacity or WebSocket chunk queues would save KiB to a
  few MiB and could increase underrun risk. It is not worthwhile.

## Native qwen-tts lower bound

The official single-process wrapper was measured in an isolated package path
without changing the stable robot environments. Its result was:

| Metric | Native qwen-tts |
|---|---:|
| CUDA reserved after load | 2.13 GiB |
| CUDA max allocated during request | 2.09 GiB |
| Native process PSS after load | 0.825 GiB |
| 2.40 s audio generation time | 13.96 s |
| RTF | **5.82** |
| Streaming | No; complete waveform return |

This path proves that several GiB of the vLLM-Omni footprint are serving/runtime
overhead rather than model weights. It does **not** provide an acceptable robot
runtime: it is non-streaming and approximately eight times slower in RTF than
the optimized vLLM-Omni measurements. Replacing the working server with it is
therefore rejected.

The reproduction utility and raw result are
[`scripts/benchmark-qwen3-tts-native.py`](../scripts/benchmark-qwen3-tts-native.py)
and
[`native_qwen_tts.json`](../benchmarks/orin_nx_2026-08-17_tts_memory/native_qwen_tts.json).

## Production profile and validation

The selected settings are:

```yaml
# Stage 0
max_num_seqs: 1
kv_cache_memory_bytes: 67108864
max_num_batched_tokens: 384
max_model_len: 384
default_sampling_params:
  max_tokens: 256
compilation_config:
  cudagraph_mode: FULL_AND_PIECEWISE
  cudagraph_capture_sizes: [1, 2]

# Stage 1
max_num_seqs: 1
max_num_batched_tokens: 4096
max_model_len: 4096
enforce_eager: true
```

`TtsConfig.max_new_tokens` and the example configuration now default to 256 so
the request cannot accidentally exceed the validated Stage-0 envelope.

Validation results:

- WebSocket streaming: passed; every steady request returned 2–5 PCM chunks.
- First PCM: mean 231.6 ms, p95 276.4 ms, maximum 284.3 ms.
- Steady RTF: mean 0.712; all requests below 1.
- Longest test: 31 Chinese characters, no truncation or server error.
- Generation/playback overlap and the continuous PCM player were not modified;
  the existing playback/queue regression suite passed.
- Full project test suite: **119 passed**.

The initial request after an empty Inductor cache took several seconds while
graphs/kernels compiled. A subsequent process start using the persistent cache
completed warm-up in about nine seconds. Those cold costs are preserved in the
raw workload files and excluded only from steady latency statistics.

No automated perceptual metric or MOS test was run. Audio duration, PCM chunk
continuity, absence of truncation/errors, and the existing audible continuous
playback baseline were used as quality guards. The sampling parameters and
model precision were unchanged.

## Recommendation and stopping decisions

Use the new profile for production single-user robot speech. It removes nearly
2 GiB of unneeded KV reservation with no measured latency or streaming penalty
and keeps a conservative capacity margin over the real short-reply envelope.

Do not continue shrinking Stage 1, speaker-cache limits, or PCM buffers for
memory alone; the evidence shows little potential return. Do not adopt the
native wrapper as the production runtime because it fails both streaming and
RTF requirements.

A custom single-process or shared-runtime TTS server could potentially recover
another 2–3 GiB of host/runtime overhead by avoiding vLLM-Omni's multi-process
architecture. That is a separate, higher-risk engineering project: the native
experiment shows that merely removing vLLM is not sufficient, because it also
removes the performance and streaming behavior the robot needs. It is worth
considering only if another multi-GiB reduction becomes a product requirement.
Quantization is also deferred because it changes the numerical/quality baseline
and the current low-risk change already captured the dominant removable memory.

The follow-up three-model capacity experiment is available as a self-contained
report at [ASR + Agent + TTS full-stack capacity](qwen3_full_stack_capacity_orin_nx.html).
It verifies that a Qwen3-4B Q5 llama.cpp Agent can coexist with SenseVoice and
Qwen3-TTS for sequential dialogue, while documenting the remaining zram and
vision-workload safety limits.

## Raw artifacts

- [`summary.csv`](../benchmarks/orin_nx_2026-08-17_tts_memory/summary.csv): compact before/after table.
- [`baseline_workload.json`](../benchmarks/orin_nx_2026-08-17_tts_memory/baseline_workload.json): baseline streaming timings.
- [`single_user_64m_workload.json`](../benchmarks/orin_nx_2026-08-17_tts_memory/single_user_64m_workload.json): optimized streaming timings.
- [`single_user_64m_workload_peak.json`](../benchmarks/orin_nx_2026-08-17_tts_memory/single_user_64m_workload_peak.json): sampled workload memory trace.
- [`vllm_omni_baseline_after_workload.json`](../benchmarks/orin_nx_2026-08-17_tts_memory/vllm_omni_baseline_after_workload.json): baseline process/GPU snapshot.
- [`single_user_64m_after_workload.json`](../benchmarks/orin_nx_2026-08-17_tts_memory/single_user_64m_after_workload.json): optimized process/GPU snapshot.
- [`single_user_64m_stage1_65536_idle.json`](../benchmarks/orin_nx_2026-08-17_tts_memory/single_user_64m_stage1_65536_idle.json): Stage-1 attribution control.
- [`native_qwen_tts.json`](../benchmarks/orin_nx_2026-08-17_tts_memory/native_qwen_tts.json): single-process native lower bound.
