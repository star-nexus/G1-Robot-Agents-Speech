# Orin NX speech end-to-end smoke test — 2026-08-11

This is a single functional measurement of the new acoustic-end/VAD/ASR
instrumentation. It is not a latency distribution and must not be compared with the
10-run model-only benchmark as if both used the same boundary.

Configuration:

- Qwen3-ASR-0.6B, FP16, SDPA, Transformers static-cache dynamic compilation;
- Jetson Orin NX 16GB;
- `vad.min_silence_seconds = 0.20`;
- input `tests/test_audio_6s.m4a`, first VAD-completed utterance;
- synchronized Qwen stage profiling enabled.

Result:

| Metric | Value |
|---|---:|
| Observed VAD tail | 226.4 ms |
| ASR queue | 0.1 ms |
| Qwen inference | 679.5 ms |
| Speech end to final text | 906.4 ms |
| Generate stage | 654.3 ms |
| Generated tokens | 10 |
| Generated tokens/s | 15.28 |
| ASR RTF | 0.2067 |

The generate stage is 96.3% of ASR inference in this request. VAD is the second
visible contribution; the ASR queue is negligible. Raw output is in
[`result.json`](result.json).
