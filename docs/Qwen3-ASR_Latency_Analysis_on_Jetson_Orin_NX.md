## Qwen3-ASR Latency Analysis on Jetson Orin NX

We profiled Qwen3-ASR-0.6B on a Jetson Orin NX 16GB running JetPack 6.2 / CUDA 12.6.

Unless otherwise stated, measurements use GPU inference with SDPA on a fixed 5.592-second Chinese WAV. CUDA synchronization is applied at timing boundaries to avoid under-reporting caused by asynchronous execution.

### 98.8% of End-to-End Latency Is Spent in `generate()`

The ASR pipeline was split into processor, host-to-device transfer, model generation, and output decoding:

| Stage | Mean latency | Share |
|---|---:|---:|
| Audio processor | 12.2 ms | 0.95% |
| Host-to-device transfer | 2.3 ms | 0.18% |
| **`model.generate()`** | **1266.5 ms** | **98.83%** |
| Output decoding | 0.3 ms | 0.03% |
| **Total** | **1281.5 ms** | 100% |

The result is unambiguous: almost all latency is inside `model.generate()`.

Audio preprocessing, CPU-to-GPU transfer, and final token decoding together account for only about **14.8 ms**, so they are not meaningful optimization targets for this workload. :contentReference[oaicite:0]{index=0}

### Nsight Profiling: GEMM and Small-Kernel Dispatch Dominate

Nsight Systems profiling of a single `generate()` call producing only 15 output tokens shows:

| Metric | Measured value |
|---|---:|
| GPU kernel instances | **26,137** |
| `cudaLaunchKernel` calls | **26,135** |
| GEMM share of GPU kernel time | **65.2%** |
| Attention kernel share | **2.35%** |
| Copy / concatenation kernel share | **14.1%** |
| Kernel instances averaging < 50 μs | **86.2%** |
| CPU-side `cudaLaunchKernel` API time | **513.3 ms** |

The dominant pattern is therefore:

> **large amounts of GEMM work, repeated model-weight/memory traffic, and a very large number of short CUDA kernel launches.**

Attention is only a small part of the total GPU execution time. This is consistent with the backend comparison: SDPA improves over eager attention only modestly, while FlashAttention 2 is slower than SDPA for this batch-1, short-utterance workload. :contentReference[oaicite:1]{index=1}

### Autoregressive Generation Is the Main Bottleneck

Controlled audio-length scaling further supports this conclusion:

| Audio duration | Mean latency | RTF | Generated tokens | Output throughput |
|---:|---:|---:|---:|---:|
| 5.592 s | 1257.9 ms | 0.2249 | 15 | 12.03 tok/s |
| 11.184 s | 2155.8 ms | 0.1928 | 26 | 12.13 tok/s |
| 22.368 s | 3955.8 ms | 0.1768 | 48 | 12.22 tok/s |
| 44.736 s | 6158.3 ms | 0.1377 | 74 | 12.10 tok/s |

Across all four lengths, output generation remains almost constant at approximately **12 tokens/s**.

A linear regression between mean latency and generated token count gives:

- slope: approximately **83 ms/token**
- \(R^2 = 0.99991\)

This strongly indicates that latency is dominated by autoregressive generation rather than audio preprocessing or attention alone. :contentReference[oaicite:2]{index=2}

### Memory Bandwidth Is Likely an Additional Constraint

The profiler data is also consistent with a memory-traffic bottleneck:

- GEMM accounts for roughly 65% of GPU kernel time.
- Copy / concatenation kernels account for about 14%.
- Batch size is one, so decoder GEMMs are relatively narrow.
- Model weights must be repeatedly accessed during token-by-token decoding.
- More than 86% of kernel instances are very short.

These observations strongly suggest that memory traffic and memory bandwidth contribute materially to the runtime behavior on Orin NX.

However, this has **not** yet been directly confirmed with hardware memory counters such as DRAM throughput, L2 hit rate, or memory-stall metrics. It should therefore be treated as a strong architectural inference rather than a directly measured result. :contentReference[oaicite:3]{index=3} :contentReference[oaicite:4]{index=4}

### Conclusion

For Qwen3-ASR-0.6B on Jetson Orin NX, the primary latency bottleneck is not audio preprocessing and not the attention kernel.

The evidence points to:

> **batch-1 autoregressive decoding dominated by GEMM, repeated model-weight/memory traffic, and tens of thousands of short CUDA kernel launches.**

In the measured 5.592-second workload:

- `model.generate()` accounts for **98.8%** of end-to-end ASR latency.
- GEMM accounts for about **65.2%** of GPU kernel time.
- Attention accounts for only about **2.35%**.
- A single 15-token generation triggers roughly **26K CUDA kernel launches**.
- Output generation throughput is approximately **12 tokens/s**.

These results indicate that further optimization effort should focus on the **generation runtime and execution scheduling**, rather than on further attention-kernel optimization.
