# Qwen3-TTS CUDA Graph attribution on Jetson Orin NX

## Executive conclusion

The earlier statement “CUDA Graph serving does not work on Orin SM87” was too
broad. The failure is narrower:

> The locally built SM87 FlashAttention 2 backend stalls when vLLM captures the
> Stage-0 single-token FULL decode graph. The same model, vLLM-Omni orchestration,
> and FULL graph configuration completes normally with `TRITON_ATTN`.

This is an experimentally isolated compatibility interaction, not proof of the
lower-level defect inside FA2. Candidate mechanisms include lazy workspace or
split-KV state during capture, capture-unsafe allocation/synchronization, or a
defect in the target-specific minimal FA2 build. Identifying the exact CUDA call
still requires a native stack or compute-sanitizer capture.

The selected Orin baseline is:

```text
Stage 0 (Talker): TRITON_ATTN + FULL_AND_PIECEWISE CUDA Graph
Stage 1 (Code2Wav): eager
```

It produced 244.8 ms mean first PCM and 0.716 mean audio RTF over six warm
steady-state WebSocket requests. The output endpoint, incremental 24 kHz PCM, and
persistent connection all completed correctly.

## What the two external analyses got right

Both analyses correctly separated three questions that had previously been mixed:

1. Stage-0 vLLM graph capture and Stage-1 Code2Wav graph capture are independent.
2. `PIECEWISE` is a valid bounded test; it compiles graph partitions without
   requiring the problematic full single-token replay graph.
3. Replacing the attention backend is an orthogonal root-cause experiment.

The more detailed analysis also correctly identified the separate
`CUDAGraphDecoderWrapper` inside Qwen3-TTS Code2Wav and the relevant streaming
chunk sizes. The shorter analysis correctly proposed `TRITON_ATTN` as a control.

Several statements were hypotheses, not established facts: a fixed percentage
for PIECEWISE speedup, a guaranteed sub-real-time result, or an assertion that
split-KV is definitely the deadlocking kernel. The experiments below replace
those predictions with measurements.

## Why MiniMax appeared although this project does not use MiniMax

vLLM's startup warm-up and model registry are generic. They discover/import
processor and kernel classes for models beyond the one being served so that the
runtime can determine supported capabilities and compile paths. That import graph
reached a MiniMax vision processor and then the incompatible system `torchvision`.
It did not load MiniMax weights, route a request to MiniMax, or indicate that the
project configuration selected MiniMax. The Jetson launcher therefore supplies a
text/audio-only torchvision compatibility shim for those generic imports.

## Experimental design

Platform and runtime:

| Item | Value |
|---|---|
| Device | Jetson Orin NX 16 GB, SM 8.7 |
| OS | JetPack 6.2 / Ubuntu 22.04 / CUDA 12.6 |
| PyTorch / Triton | 2.11.0 / 3.5.1 |
| vLLM-Omni | official v0.26.0 release |
| vLLM | native SM87 source build, 0.26.1.dev0 (`568afb3a1`) |
| Model | Qwen3-TTS-12Hz-0.6B-CustomVoice, Ryan, BF16 |
| Client | persistent localhost WebSocket, incremental PCM, no playback |

The two short English prompts were alternated. Qwen3 CustomVoice samples
stochastically, so generated duration varies. Audio RTF (`generation wall time /
returned PCM duration`) is the more useful throughput comparison; first PCM is the
robot-interaction latency comparison. Cold JIT/compilation is reported separately
and excluded from steady-state means.

## Results

| Stage-0 attention / graph | Stage-1 graph | Runs | First PCM mean | Audio RTF mean | Outcome |
|---|---|---:|---:|---:|---|
| FA2 / eager | eager | 4 | 509.1 ms | 2.113 | functional baseline |
| FA2 / PIECEWISE | eager | 4 | 382.8 ms | 1.494 | passes |
| TRITON / PIECEWISE | eager | 6 | 420.8 ms | 1.637 | passes |
| **TRITON / FULL+PIECEWISE** | **eager** | **6** | **244.8 ms** | **0.716** | **selected** |
| FA2 / eager | inner graph 25,97 | 4 | 592.7 ms | 2.087 | passes, no useful latency gain |
| FA2 / PIECEWISE | inner graph 25,97 | 6 | 469.2 ms | 1.438 | small RTF gain, worse first PCM |

Key capture observations:

- Historical FA2 FULL capture: no progress for 201 seconds at Stage-0
  single-token decode capture; manually stopped.
- TRITON FULL capture: the same capture size completed in 2.08 seconds; all
  Stage-0 graph captures completed in about 5 seconds and used about 0.05 GiB.
- Stage-1 inner graph sizes 25 and 97: 12/12 observed graph hits and no fallback,
  but reserved memory increased from about 0.72 to 2.18 GiB. It slightly improved
  throughput in one combined test while worsening first PCM, so it is not selected
  for the 16 GB robot baseline.
- A first request can still spend seconds compiling uncaptured Triton/code-predictor
  shapes. Production readiness must be judged only after an explicit warm-up request.

The large speedup comes from enabling FULL single-token replay, not merely from
switching attention kernels: TRITON PIECEWISE was slower than FA2 PIECEWISE, while
TRITON FULL was substantially faster than both.

Raw per-run JSON and the machine-readable experiment summary are in
`benchmarks/orin_nx_2026-08-13_vllm_omni/experiments/`.

## Root-cause ranking

| Candidate | Finding |
|---|---|
| Stage-0 FA2 + FULL capture compatibility | **Confirmed boundary**: stalls with FA2, passes with TRITON |
| Generic vLLM-Omni CUDA Graph machinery | **Ruled out as sole cause** |
| Qwen3-TTS model graph inherently uncapturable | **Ruled out as sole cause** |
| Stage-1 Code2Wav graph | **Independent**; graph hits work, poor trade-off here |
| Lazy/JIT work before steady state | **Confirmed**; first request must be warmed |
| Exact FA2 kernel or API causing the stall | **Unresolved**; requires lower-level trace |

## Implemented solution

The production profile now enables Stage-0 `FULL_AND_PIECEWISE` for capture sizes
1 and 2. The launcher defaults to the graph-safe backend:

```bash
QWEN3_TTS_ATTENTION_BACKEND=TRITON_ATTN \
bash scripts/run-qwen3-tts-vllm-omni.sh
```

Extra CLI arguments are forwarded by the launcher. To diagnose the known FA2
path without changing files, set `QWEN3_TTS_ATTENTION_BACKEND=FLASH_ATTN_2`; do
not use that combination for unattended startup until its FULL capture defect is
fixed.

The quickest safe fallback is to copy the profile to a host-local path, set both
stages to `enforce_eager: true`, and point `QWEN3_TTS_DEPLOY_CONFIG` at it. This
keeps the checked-in low-latency profile and the local diagnostic override
separate.

## Remaining work before robot production acceptance

This benchmark proves model-server synthesis performance, not the full robot
experience. Before deployment, measure peak unified memory alongside ASR and the
interaction model, run long multi-request soak tests, verify audio quality in both
languages, pre-warm every production voice/style, and measure end-to-speaker PCM,
underruns, barge-in, AEC, and stop-to-silence latency. Those gates remain valid
even though the server itself is now faster than real time.
