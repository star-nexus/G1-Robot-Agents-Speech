# Orin NX Qwen3-TTS single-user memory experiment

This directory contains the raw data used by
[`docs/qwen3_tts_memory_orin_nx.md`](../../docs/qwen3_tts_memory_orin_nx.md).
Measurements were made on the same Jetson Orin NX while the normal speech
service remained active. Jetson GPU mappings, process PSS, and whole-system used
memory overlap and must not be summed.

## Profiles

- `profiles/baseline_1024_65536.yaml`: frozen pre-change profile.
- `profiles/single_user_64m_384_4096.yaml`: selected production candidate.
- `profiles/single_user_64m_stage1_65536.yaml`: Stage-1 length control.

## Reproduction outline

Start either profile in the isolated vLLM-Omni runtime:

```bash
VLLM_OMNI_RUNTIME_ROOT=/home/nvidia/Developer/vllm-omni-orin-jp6 \
VLLM_OMNI_SOURCE_DIR=/home/nvidia/Developer/vllm-omni-orin-jp6/source-v0.26.0 \
QWEN3_TTS_MODEL=/home/nvidia/Developer/llama/Qwen3-TTS-12Hz-0.6B-CustomVoice \
QWEN3_TTS_CUDSS_DIR=/home/nvidia/Developer/vllm-orin-jp6/cudss-0.7.1/rootfs/usr/lib/aarch64-linux-gnu/libcudss/12 \
QWEN3_TTS_DEPLOY_CONFIG="$PWD/benchmarks/orin_nx_2026-08-17_tts_memory/profiles/single_user_64m_384_4096.yaml" \
bash scripts/run-qwen3-tts-vllm-omni.sh
```

Run the streaming corpus over one persistent WebSocket:

```bash
.venv-gpu/bin/python scripts/benchmark-qwen3-tts-websocket.py \
  --model /home/nvidia/Developer/llama/Qwen3-TTS-12Hz-0.6B-CustomVoice \
  --voice Vivian --language Chinese \
  --instructions '用温暖、自然、连贯的普通话说话。' \
  --max-new-tokens 256 --warmup 1 --runs 5 \
  --text '你好。' \
  --text '欢迎来到未来世界。' \
  --text '我是STAR机器人，很高兴见到你。' \
  --text '前方正在举行花车巡游，请跟我从右侧通道慢慢前进。' \
  --text '欢迎来到迪士尼公园，我是雪宝机器人，今天陪你探索冰雪世界和奇妙冒险。' \
  --output /tmp/qwen3_tts_workload.json
```

Capture a process and memory snapshot:

```bash
/usr/bin/python3 scripts/measure-jetson-memory.py \
  --label single_user_64m_after_workload \
  --match vllm --match g1-speech --match omni \
  --output /tmp/qwen3_tts_memory.json
```

For a workload peak trace, add `--samples 600 --interval 0.1 --no-jtop` and run
the WebSocket corpus during the sampling window. jtop is intentionally captured
only once per snapshot because polling it at 10 Hz perturbs the experiment.

The native lower-bound environment was target-installed under the ignored
`.runtime/` directory and reused the isolated Jetson-native PyTorch runtime. It
did not modify `.venv-gpu` or the system Python. See
`scripts/benchmark-qwen3-tts-native.py` for the exact measurement fields.

## Known limitations

- Qwen3-TTS sampling is stochastic; semantic-token and waveform lengths vary.
- Whole-system memory contains normal OS/cache drift. TTS-off snapshots before
  and after the experiment are retained for context.
- Some jtop process rows can remain stale immediately after worker turnover.
  Stage mapping values in `summary.csv` are included only where PIDs matched the
  contemporaneous process snapshot.
- The peak trace was produced by schema v1 of the capture script. Its high-rate
  `selected_pss_kib: 0` means “smaps not sampled”, not zero resident memory. Use
  the final `selected_process_totals.pss_kib`; schema v2 emits `null` instead.
- The cold compile request is retained in each workload JSON under `warmup` but
  is excluded from steady-state summary statistics.
