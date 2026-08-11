# Qwen3-ASR Runtime Optimization on Jetson Orin NX

## Overview

This report documents a runtime-optimization study of **Qwen3-ASR-0.6B** on a **Jetson Orin NX 16GB**. The goal was to determine whether the previously identified `model.generate()` bottleneck could be reduced in practice, and which optimization paths actually help on this device.

The key result is:

> **Transformers static KV cache + generation-aware compiled decoding reduced steady-state latency from the 1.2-second range to 451–494 ms, while increasing request-level apparent generation throughput from roughly 12 tokens/s to 32–34 tokens/s.**

Nsight Systems confirms that the improvement primarily comes from reducing repeated CUDA kernel-launch overhead during autoregressive decoding.

---

## Test Environment

- Device: **Jetson Orin NX 16GB**
- JetPack: **6.2**
- Jetson Linux / L4T: **36.4.3**
- CUDA: **12.6**
- GPU architecture: **SM87**
- Model: **Qwen3-ASR-0.6B**
- Precision: **FP16**
- Attention backend: **SDPA**
- Batch size: **1**
- Decoding: greedy
- Audio fixture: **5.592-second Chinese WAV**
- Warm-up: 3 requests
- Measured runs: 10 requests unless otherwise stated
- Timing: wall-clock timing with CUDA synchronization at measured segment boundaries

The stable robot runtime was left untouched. Runtime-optimization experiments used an isolated Jetson-native environment with:

- PyTorch **2.9.1**
- Triton **3.5.1**
- CUDA **12.6**
- native **SM87** support

All successful variants produced the same transcript hash on the fixed evaluation
sample. This is a **transcript-consistency check for the fixed fixture**, not a
corpus-level CER claim. The literal transcript is intentionally omitted from the
public report.

---

## Measured Runtime Results

| Variant | Mean latency | P95 | RTF | Request-level apparent generation throughput | Relative to PyTorch 2.9 eager |
|---|---:|---:|---:|---:|---:|
| PyTorch 2.9 eager, FP16 + SDPA | 1394.7 ms | 1408.7 ms | 0.2494 | 10.89 tok/s | baseline |
| Decoder `torch.compile(mode="default")` | 567.2 ms | 570.8 ms | 0.1014 | 26.90 tok/s | **-59.3% latency** |
| **Static KV cache + automatic compiled decode** | **451.6 ms** | **460.1 ms** | **0.0808** | **34.27 tok/s** | **-67.6% latency** |
| Static cache, fresh-process repeat | 494.3 ms | 572.4 ms | 0.0884 | 31.56 tok/s | **-64.6% latency** |
| bitsandbytes NF4 W4A16 | 1897.8 ms | 1970.8 ms | 0.3394 | 8.00 tok/s | +36.1% latency |
| NF4 + static-cache request | 2163.7 ms | 2219.7 ms | 0.3869 | 7.03 tok/s | +55.1% latency |

The previously preserved PyTorch 2.5 FP16 baseline was **1218.3 ms**. PyTorch 2.9 eager was therefore not faster by itself; in fact, it was slower.

The major gain came from changing the **generation execution path**, not merely upgrading PyTorch.

The best measured result, **451.6 ms**, is approximately **62.9% lower latency** than the preserved PyTorch 2.5 FP16 baseline.

---

## What Actually Worked

### Static KV Cache + Generation-Aware Compiled Decode

Directly compiling the full multimodal model was not reliable.

The experiments showed three distinct behaviors:

1. Compiling the full multimodal forward path caused graph breaks in the audio frontend and CUDA Graph output-aliasing failures.
2. Compiling only the repeated language decoder removed the audio-side graph breaks, but manual `reduce-overhead` execution still encountered KV-cache/output lifetime issues.
3. Using Transformers with:

```python
cache_implementation="static"
```

allowed the generation runtime to use its own compilable cache and compiled decode path.

This worked because the generation runtime manages the cache, decoding step boundaries, and tensor lifetimes itself.

The practical conclusion is:

> **CUDA Graph is effective, but the integration boundary matters. Static KV cache + generation-aware compiled decoding works; manually wrapping the full Qwen3-ASR model with `torch.compile(mode="reduce-overhead")` does not.**

---

## Nsight Evidence: CUDA Graph Removes Most Launch Overhead

Nsight Systems profiling of the optimized static-cache path provides direct evidence that the new runtime structure addresses the previously observed kernel-launch bottleneck:

| CUDA API metric | PyTorch 2.5 eager baseline | PyTorch 2.9 static cache | Change |
|---|---:|---:|---:|
| Ordinary `cudaLaunchKernel` calls | 26,135 | 2,832 | **-89.2%** |
| CPU time in ordinary kernel launch APIs | 513.3 ms | 65.9 ms | **-87.2%** |
| `cudaGraphLaunch` calls | 0 | 14 | CUDA Graph replay active |

The optimized path reduced ordinary kernel launches by nearly **90%**.

This directly supports the earlier bottleneck analysis:

> The large number of small, repeatedly launched kernels during batch-1 autoregressive decoding was not merely a profiling artifact; it was a real optimization target.

The static-cache/compiled-decode path converts much of that repeated work into CUDA Graph replay, substantially reducing CPU-side launch overhead.

---

## Cold-Start Cost

The steady-state improvement is large, but compilation introduces a significant cold-start cost.

Measured first-request behavior:

- Decoder compile with `mode="default"`: approximately **180.6 s**
- Static-cache compiled path: approximately **49.9 s**
- Fresh Python process with persisted Inductor cache: approximately **17.56 s** before steady-state execution

After warm-up, the static-cache fresh-process repeat reached:

- **494.3 ms mean latency**
- **31.56 tokens/s request-level apparent generation throughput**

This makes the optimization well suited to a **long-running ASR service**, where compilation and warm-up can happen during startup.

It is less suitable for short-lived command-line processes that start, transcribe once, and exit.

---

## Why NF4 Quantization Was Slower

A native SM87 build of bitsandbytes 0.49.2 was used to avoid drawing conclusions from an incompatible generic aarch64 wheel.

Only the language-model linear weights were quantized to NF4 W4A16; the audio tower, multimodal projector, and `lm_head` remained FP16.

The result was a clear regression:

- FP16 eager: **1394.7 ms**
- NF4 eager: **1897.8 ms**
- NF4 + static-cache request: **2163.7 ms**

Reducing weight traffic was not enough to offset the cost of per-layer dequantization and generic 4-bit GEMM kernels on this workload.

Therefore:

> **bitsandbytes NF4 is not a low-latency solution for batch-1 Qwen3-ASR on this Orin NX configuration.**

This result should **not** be generalized to AWQ, GPTQ, Marlin, TensorRT-LLM, or other fused low-bit runtimes with different kernel implementations.

---

## vLLM 0.14.0: Native Build Succeeded, Runtime Is Currently Memory-Limited

A native **aarch64 / CUDA 12.6 / SM87** build of vLLM 0.14.0 was completed successfully in an isolated environment.

The resulting build passed:

- `vllm._C` import
- native SM87 CUDA-op smoke tests
- FlashInfer import
- Qwen3-ASR backend import and model resolution

The runtime reached Qwen3-ASR model loading, but the process was terminated by the Linux OOM killer before inference completed.

The failure was reproduced with progressively smaller configurations, including:

- lower `gpu_memory_utilization`
- same-process execution
- eager mode
- `max_model_len=1024`
- `gpu_memory_utilization=0.1`

Because the failure still occurred during weight loading, the current blocker is **load-time unified-memory peak**, not KV-cache capacity.

Therefore the current vLLM conclusion is:

> **JetPack 6 / CUDA 12.6 / SM87 can build and import vLLM 0.14.0 natively, but this 16GB Orin NX configuration has not yet completed end-to-end Qwen3-ASR inference because of load-time memory pressure.**

No vLLM latency claim is made until a complete transcription succeeds.

---

## TensorRT-LLM: No End-to-End Qwen3-ASR Path Yet

TensorRT-LLM 1.2.1 supports standard Qwen3 causal-language-model architectures, but its model registry and converter do not provide an adapter for:

```text
Qwen3ASRForConditionalGeneration
```

End-to-end Qwen3-ASR support would require integration of:

- the audio tower
- multimodal projector
- audio position / embedding injection
- the language decoder

In addition, TensorRT-LLM 1.2.1 expects a newer TensorRT stack than the one provided by JetPack 6.2.

Therefore this experiment does **not** show that TensorRT-LLM is slow or ineffective. It shows that there is currently no ready-made Qwen3-ASR conversion/runtime path to benchmark on this platform.

---

## Interpretation

The results reinforce the earlier profiling conclusion.

The dominant latency problem on Jetson Orin NX is not audio preprocessing and not the attention kernel. It is the execution pattern of **batch-1 autoregressive decoding**:

- repeated GEMM execution
- repeated model-weight / memory traffic
- tens of thousands of short CUDA kernel launches
- significant CPU-side launch overhead

The most effective optimization tested so far is not a different attention implementation or generic weight quantization.

It is:

> **Static KV cache + generation-aware compiled decoding + CUDA Graph replay.**

This changes the execution structure of decoding itself.

On the tested Orin NX, that reduced steady-state latency from the 1.2-second range to approximately **0.45–0.49 seconds** and increased request-level apparent generation throughput from roughly **12 tokens/s to 32–34 tokens/s**.

### Metric clarification from the E2E attribution study

The request-level metric above is:

```text
generated_tokens / entire_generate_time
```

It includes the audio encoder, multimodal projector, LM prefill, first-token latency,
autoregressive decode, and generation bookkeeping. It must not be described as decoder
throughput.

A later same-runtime, same-PCM token-scaling experiment estimated:

```text
generate_ms = 157.216 ms + 28.147 ms * generated_tokens
R² = 0.999725
incremental decode throughput = 35.53 tokens/s
```

The same attribution study confirmed that the earlier 15.3 tokens/s E2E observation
was a request-level apparent value depressed by fixed encoder/prefill cost and the
first warm-up of a new audio-shape CUDA Graph. It was not a twofold decoder regression.

It also found that, for this workload and configuration, dynamic compilation was
slower than the fixed-shape control: total latency increased from 459.1 ms to 580.1 ms
(+26.3%), while the generate stage increased by approximately 123 ms (+27.8%). This is
a separate runtime-configuration finding, not evidence of a decoder throughput loss.
The complete controlled attribution and raw-data index are available in
[`qwen3_asr_e2e_runtime_gap.md`](qwen3_asr_e2e_runtime_gap.md).

---

## Bottom Line

For **Qwen3-ASR-0.6B, batch 1, Jetson Orin NX 16GB**, the strongest measured configuration in this study is:

```text
FP16
+ SDPA
+ Transformers static KV cache
+ generation-aware compiled decode
+ CUDA Graph replay
```

Measured steady-state performance:

- **451.6 ms mean latency**
- **460.1 ms P95**
- **RTF 0.0808**
- **34.27 tokens/s request-level apparent generation throughput**

A fresh-process repeat reproduced the benefit at:

- **494.3 ms mean latency**
- **RTF 0.0884**
- **31.56 tokens/s request-level apparent generation throughput**

The key optimization lesson is:

> **Optimize the generation runtime, not the attention kernel.**

The decoder-specific result should be stated separately:

> **Steady-state incremental decode throughput remains approximately 35.5 tokens/s on
> this Orin NX. The earlier 15.3 tokens/s observation was a request-level apparent
> generation throughput value, not a decoder performance regression.**

---

## Reproducibility Notes

The benchmark evidence includes:

- raw JSON results
- per-run CSV files
- summary CSV and Markdown views
- Nsight Systems summaries
- retained failure traces for unsuccessful optimization paths
- native vLLM build metadata

The stable robot service environment, system CUDA, JetPack installation, and system Python were not replaced during these experiments.
