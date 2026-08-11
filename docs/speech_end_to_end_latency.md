# Speech end-to-end latency and streaming plan

The live latency target must be measured from the last acoustic speech sample, not
from the time at which VAD finally emits an utterance. The service now records these
components with one `event_id`:

```text
last speech sample
  -> VAD ready                vad_tail
  -> ASR thread starts        asr_queue
  -> final transcript         inference / speech_end_to_final
  -> brain DDS callback       created_to_received
  -> first generated token    brain_first_token
  -> first TTS audio sample   tts_first_audio
```

The ASR log emits a joinable line such as:

```text
Speech latency event_id=... vad_tail=...ms asr_queue=...ms inference=...ms speech_end_to_final=...ms
```

`DdsSpeechSubscriber` and `examples/agent_subscriber.py` report
`created_to_received` for the same event. The interaction brain must retain the
incoming `event_id` in its first-token and TTS-start logs. The full user-perceived
latency is then:

```text
speech_end_to_first_token = speech_end_to_final + created_to_received + brain_first_token
speech_end_to_first_audio = speech_end_to_first_token + tts_first_audio
```

`created_unix_ns` is generated immediately after final ASR output. It is suitable
for cross-process DDS/brain measurements because it uses the wall clock. VAD and
ASR stage durations use the monotonic clock so system-clock corrections cannot
corrupt them. The current interaction-brain example produces text but has no TTS
playback implementation, so this repository can measure through first text token;
first audible response requires instrumentation at the actual audio output callback.

## Run the measurement

To observe the DDS handoff independently of the brain:

```bash
PYTHONPATH=src .venv/bin/python examples/agent_subscriber.py wlP1p1s0 --domain 0
```

For a controlled WAV run through the selected backend and real VAD:

```bash
PYTHONPATH=src .venv-gpu/bin/python acceptance/e2e_latency.py \
  --config config.local.json --wav tests/test_audio_6s.m4a --json \
  --target-ms 2000
```

The tool reports the configured silence threshold, observed VAD wait, ASR queue,
inference time, end-to-final latency, and RTF. WAV input is read directly; M4A and
other formats are decoded to mono 16 kHz through `ffmpeg`.

Qwen3-ASR live generation detail is opt-in because synchronized per-stage timing
adds small measurement overhead. Enable it only during diagnosis:

```json
{
  "qwen3_asr": {
    "log_profile": true
  }
}
```

It adds `generated_tokens`, `generate_ms`, request-level apparent generation
throughput, processor, H2D, and decode timing to each recognition. This is a
model-profile setting and intentionally has no `deploy.env` override.

## 200 ms VAD experiment

The Orin host-local `deploy.env` uses `VAD_MIN_SILENCE_SECONDS=0.20`. This shared
removes about 150 ms from the former 350 ms tail threshold when the VAD decision
tracks configuration exactly, regardless of the selected ASR model. It is an
experiment, not a universal default: noise,
hesitation, and natural pauses can split one sentence into multiple utterances.

Validate at least these cases before making 200 ms the deployment default:

- a normal sentence with a short internal pause;
- a hesitant command with a 200–400 ms pause;
- far-field and noisy speech;
- immediate robot playback after recognition.

Compare `utterances_detected`, empty/error counts, transcript accuracy, and
`vad_tail` against the 350 ms setting. Revert the local value to `0.35` if sentence
fragmentation becomes material.

One 2026-08-11 profiled smoke test on `tests/test_audio_6s.m4a` completed successfully
with the 200 ms local setting: observed VAD wait 226.4 ms, ASR queue 0.1 ms,
Qwen3-ASR inference 679.5 ms, and acoustic-end-to-final 906.4 ms. Generation produced
10 tokens in 654.3 ms (15.28 tokens/s), accounting for 96.3% of ASR inference time.
This is a single functional check, not a latency distribution or an accuracy
comparison; the 200 ms setting must still pass conversational pause testing before
it becomes a stable default. The raw result is stored in
[`benchmarks/orin_nx_2026-08-11_e2e_latency`](../benchmarks/orin_nx_2026-08-11_e2e_latency/README.md).

## True streaming Qwen3-ASR

The current Transformers adapter is offline utterance inference. Qwen3-ASR may
internally process audio in chunks, but `model.generate()` is still called only
after VAD completes the utterance; this is not application-level streaming.

The [official Qwen3-ASR repository](https://github.com/QwenLM/Qwen3-ASR) exposes a
stateful streaming API through its vLLM backend. Its
[official streaming example](https://github.com/QwenLM/Qwen3-ASR/blob/main/examples/example_qwen3_asr_vllm_streaming.py)
uses `init_streaming_state`, repeated `streaming_transcribe`, and
`finish_streaming_transcribe`. The example uses a 2-second model chunk and tests
500–4000 ms feed steps, so a final result within 100 ms must be demonstrated on the
target hardware rather than assumed.

On this Orin NX 16GB / JetPack 6 system, native SM87 vLLM 0.14.0 already builds and
imports, but Qwen3-ASR-0.6B loading has been killed by system OOM even with reduced
memory settings. See [the recorded vLLM experiment](vllm_orin_jp6.md). Therefore the
official true-streaming path remains **blocked by runtime memory**, not disproven.

The next implementation should use a distinct streaming capability with this
lifecycle:

```text
start_stream -> accept_audio_chunk -> partial hypothesis -> finish_stream
```

Do not simulate it by retranscribing the entire accumulated waveform for every
microphone chunk. That increases work approximately quadratically, produces unstable
partial text, and competes with the interaction brain for unified memory bandwidth.
Useful next experiments are reducing vLLM model-load peak memory in the isolated
environment, or adding another genuinely streaming ASR backend behind the same
pluggable service boundary while keeping Qwen3-ASR as the high-accuracy final pass.
