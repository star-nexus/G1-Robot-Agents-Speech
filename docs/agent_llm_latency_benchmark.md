# Agent LLM Responses latency benchmark

`scripts/benchmark-qwen3-agent-responses.py` measures the local llama.cpp Agent
without starting STAR Runtime. ASR, sentence segmentation, TTS, playback and AEC
are therefore outside the measurement boundary.

With the Agent server already running, benchmark the Tifa Role Package with:

```bash
cd /home/nvidia/Developer/G1-Robot-Agents-Speech

.venv-gpu/bin/python scripts/benchmark-qwen3-agent-responses.py \
  --role-package roles/tifa-lockhart \
  --prompt '你的伙伴都有谁？' \
  --max-output-tokens 64 \
  --warmup 1 \
  --runs 5 \
  --output artifacts/agent-llm-responses.json
```

The model is discovered from `/v1/models`. The script calls
`/v1/responses/input_tokens` before the run and streams every measured request
through `/v1/responses`. It uses standard `response.completed.usage` for total
input/output tokens and llama.cpp's `timings_per_token` extension for exact
provider prompt/decode timing and cumulative token progress.

The principal metrics are:

- `time_to_first_text_delta_ms`: client monotonic request start to the first
  non-empty `response.output_text.delta` (TTFT).
- `response_completed_ms`: client request start to `response.completed`.
- `server.prompt_eval_ms`: llama.cpp prompt evaluation time.
- `server.decode_tokens_per_second`: llama.cpp final predicted-token rate.
- `token_arrival_by_second`: token increments bucketed by client arrival time
  after the first text delta.

Do not interpret llama.cpp's console `prompt eval ... / N tokens` value as total
input size when prefix caching is active. It is the newly evaluated portion.
Use `usage.input_tokens` for total input and `cached_input_tokens`/`cache_n` for
the reused prefix.

To reproduce a later conversation turn, provide previous messages as a JSON
array and pass it with `--history-file`:

```json
[
  {"role": "user", "content": "你是谁？"},
  {"role": "assistant", "content": "私はティファ。よろしくね。"}
]
```

Keep the prompt, history, seed and maximum output tokens identical when comparing
idle and concurrent-load runs. Warmups are retained in the JSON report but are
excluded from the summary.

Repeating one identical `--prompt` measures the exact-prompt hot-cache path; it
can legitimately reach nearly 100% `cached_input_tokens`. To approximate normal
consecutive turns while retaining the same Role/system prefix, repeat the option
with different questions:

```bash
.venv-gpu/bin/python scripts/benchmark-qwen3-agent-responses.py \
  --role-package roles/tifa-lockhart \
  --prompt '你的伙伴都有谁？' \
  --prompt '介绍一下爱丽丝。' \
  --prompt '你和克劳德是什么关系？' \
  --warmup 0 --runs 6 \
  --max-output-tokens 64 \
  --output artifacts/agent-llm-consecutive-turns.json
```

Never compare TTFT without also comparing `input_tokens` and
`cached_input_tokens`. The final decode tok/s remains a separate metric.
