# ASR + Agent + TTS full-stack capacity experiment

This directory contains the host-level jtop and `/proc` snapshots, llama.cpp
responses, TTS WebSocket results, peak trace, and the canonical report artifact
for the 2026-08-17 Orin NX experiment.

The active speech service used SenseVoice on GPU. Qwen3-TTS used the production
`profiles/qwen3-tts-orin-nx.yaml` profile. The temporary Agent used
`Qwen3-4B-Q5_K_M.gguf` with llama.cpp CUDA.

## Tested Agent profiles

The 4096-context control used:

```text
parallel=1
context=4096
batch=256
ubatch=64
GPU layers=all
flash attention=on
K/V cache=q8_0
```

The selected robot candidate used:

```text
parallel=1
context=2048
batch=128
ubatch=32
GPU layers=all
flash attention=on
K/V cache=q8_0
```

It can be started through the repository launcher without placing host-specific
paths in `deploy.env`:

```bash
QWEN3_AGENT_LLAMA_SERVER=/path/to/llama-server \
QWEN3_AGENT_MODEL=/path/to/Qwen3-4B-Q5_K_M.gguf \
bash scripts/run-qwen3-agent-llama.sh
```

The launcher binds to `127.0.0.1:8080` by default. It intentionally exposes one
slot only. Its endpoint is OpenAI-compatible at `/v1/chat/completions`.

For short robot replies, send `chat_template_kwargs.enable_thinking=false` and
bound `max_tokens`; otherwise Qwen3 reasoning can consume latency, context, and
output budget that the spoken response does not need.

## Experimental sequence

1. Boot without the desktop UI, start and warm SenseVoice ASR.
2. Start Qwen3-TTS, warm its graph/runtime, and generate several sentences.
3. Capture `asr_tts_before_agent.json`.
4. Start the 4096 Agent, capture idle, send one 52+13-token controlled request,
   then send its answer to the persistent TTS WebSocket.
5. Sample system memory at 10 Hz for 60 seconds across the request sequence.
6. Run three additional TTS requests to separate the first page-in from steady
   behavior.
7. Stop the Agent, then repeat with the 2048 candidate.

The second profile inherits zram/page-cache state from the first test. Therefore
its whole-system used/available memory must not be compared as if both profiles
started from a fresh boot. The contemporaneous jtop Agent GPU mapping and
same-input inference timing remain useful A/B evidence.

## Files

- `asr_tts_before_agent.json`: ASR + warmed TTS reference.
- `full_stack_idle_4096_q8kv.json`: three-model idle snapshot.
- `full_stack_request_peak_4096_q8kv.json`: 60-second request peak trace.
- `agent_response_*.json`: same-input Agent outputs and timing fields.
- `tts_*agent*.json`: persistent-WebSocket First PCM, chunks, duration, and RTF.
- `full_stack_idle_2048_q8kv.json`: smaller-context A/B snapshot.
- `artifact.json`: validated report input.

Jetson GPU mappings, process PSS, and whole-system memory overlap because CPU and
GPU share physical RAM. Do not add those columns together.
