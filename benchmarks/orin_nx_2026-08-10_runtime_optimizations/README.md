# Orin NX Qwen3-ASR runtime optimization evidence

This directory contains the run-level evidence for
[`docs/qwen3_asr_orin_nx_runtime_optimizations.md`](../../docs/qwen3_asr_orin_nx_runtime_optimizations.md).

## Controlled benchmark

- Device: Jetson Orin NX 16GB, JetPack 6.2 / L4T 36.4.3, CUDA 12.6, SM87.
- Audio: the repository's 5.592-second Chinese `zh.wav` fixture.
- Model: local Qwen3-ASR-0.6B-HF, greedy decode, SDPA, FP16, batch 1.
- Protocol: three warm-ups followed by ten measured requests. CUDA is
  synchronized at every measured segment boundary.
- Stable service environment: untouched. These runs use an isolated
  Jetson-native PyTorch 2.9.1 / Triton 3.5.1 environment.
- Every successful variant produced the same transcript:
  `開放時間：早上九點至下午五點。`

Each successful prefix has raw JSON, per-run CSV, summary CSV, and a Markdown
lookup table. Sanitized `*.failure.txt` files are intentionally retained for
failed experiments.

The fresh-process repeat intentionally uses one warm-up and five measured runs.
The Nsight capture contains one post-warm-up request and is evidence about launch
structure, not an uninstrumented latency result: it recorded 2,832 ordinary
`cudaLaunchKernel` calls, 14 `cudaGraphLaunch` calls, and 65.9 ms of ordinary
launch API time. The prior eager baseline recorded 26,135 ordinary launches and
513.3 ms in the same `generate` range.

## Variants

| Prefix | Meaning |
|---|---|
| `torch291_fp16_sdpa_baseline` | PyTorch 2.9.1 eager baseline |
| `torch291_fp16_sdpa_compile_default` | decoder `torch.compile(mode="default")` |
| `torch291_fp16_sdpa_static_cache` | Transformers static KV cache and automatic compiled decode |
| `torch291_fp16_sdpa_static_cache_restart` | fresh-process repeat using the persisted Inductor cache (one warm-up, five runs) |
| `torch291_bnb_nf4_sdpa` | native-SM87 bitsandbytes, decoder-only NF4 W4A16 |
| `torch291_bnb_nf4_sdpa_static_cache` | NF4 plus static-cache request |
| `torch291_fp16_sdpa_compile*.failure.txt` | full-model and decoder-only `reduce-overhead` CUDA Graph failures |
| `tensorrt_llm_1_2_1_conversion.failure.txt` | uninstalled source-tree conversion probe |
| `vllm014_native_sm87_build.json` | isolated native wheel/build manifest |
| `vllm014_fp16_runtime.failure.txt` | exact end-to-end model-load/OOM boundary |
| `qwen3_fp16_static_cache_nsys_*.csv` | scoped CUDA API, kernel, and NVTX summaries for compiled static cache |

Local JSON configurations and compiler caches are ignored by Git. The native
PyTorch, Triton, cuDSS, vLLM, and bitsandbytes build trees live under
an external, isolated build directory and are not runtime dependencies of the
stable service.
