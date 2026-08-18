# JetPack 6 上 Qwen3-ASR native vLLM 可行性

## 结论

当前 `.venv-gpu` **不能直接构建或运行 Qwen3-ASR 官方 vLLM 路线**。Qwen3-ASR
官方包固定 `vllm==0.14.0`；该版本固定 `torch==2.9.1`，而项目稳定环境是 Jetson-native
PyTorch `2.5.0a0+872d972e41.nv24.08`。稳定环境因此保持不变。

本轮已经在仓库外的独立 `$VLLM_WORKDIR` 中完成 native aarch64 /
CUDA 12.6 / SM87 wheel。主 CUDA 扩展导入和 SM87 op 均通过，Qwen3-ASR 引擎也能进入
权重加载；但四种低内存配置都在加载阶段被 Linux OOM killer 终止。因此结论已从
“能否在 Jetson 编译”推进为“**能编译，Orin NX 16GB 的运行时加载峰值仍待解决**”。
尚未获得 vLLM eager 或 CUDA Graph latency，不能把失败解释为这些方案无效。

## 已核验的版本事实

| 组件 | 官方/当前要求 | 本机稳定环境 | 判断 |
|---|---|---|---|
| Qwen3-ASR Python 包 | `vllm==0.14.0` | 未安装 vLLM | 必须使用 0.14.0 API 路线 |
| vLLM 0.14.0 | `torch==2.9.1` | PyTorch 2.5 Jetson build | 不兼容，必须新建 PyTorch 栈 |
| vLLM CUDA 架构 | CMake 明确包含 8.7 | Orin SM 8.7 | 主 vLLM CUDA 扩展架构可行 |
| CUDA | 源码构建可指定现有 toolkit | CUDA 12.6 | 可行，必须 `CUDA_HOME=/usr/local/cuda` |
| 构建工具链 | CMake ≥3.26.1；完整构建依赖 Triton 等 | 隔离环境安装 CMake / Triton | 可行，未替换系统包 |
| Jetson Containers | recipe 支持动态 SM87 patch | BuildKit 0.18.2 无 device entitlement | 容器路线阻塞，改走隔离 native build |
| native vLLM wheel | 0.14.0 + cu126 + aarch64 | SM87 minimal dense build | 构建/导入成功 |
| Qwen3-ASR runtime | 0.6B FP16 | 16GB unified memory | 权重加载阶段 OOM |

版本核验使用以下官方源码快照：

- Qwen3-ASR commit `7c6daf77a2421100f5fb066495372c00129d39ff`：
  `pyproject.toml` 的 vLLM extra 固定 0.14.0，并通过 `Qwen3ASRModel.LLM(...)`
  暴露官方接口。
- vLLM tag `v0.14.0`, commit `b17039bccc17b94a2ea107272ad7bc93508708df`：
  `pyproject.toml`、`requirements/build.txt` 和 `requirements/cuda.txt` 固定
  PyTorch 2.9.1；`CMakeLists.txt` 的 CUDA 支持架构包含 8.7。
- jetson-containers commit `70c149aea6126594153ef7690ec4890f8f396518`：
  vLLM build recipe 使用 `CUDA_HOME=/usr/local/cuda`、aarch64 低并发编译，
  并动态生成目标 SM patch；PyTorch recipe 包含 2.9.1 的 JetPack 6 变体。

对应上游资料：[Qwen3-ASR 官方仓库](https://github.com/QwenLM/Qwen3-ASR)、
[vLLM v0.14.0](https://github.com/vllm-project/vllm/tree/v0.14.0)、
[Jetson Containers](https://github.com/dusty-nv/jetson-containers) 和
[JetPack 6.2 组件说明](https://developer.nvidia.com/embedded/jetpack-sdk-62)。

## 隔离构建实际结果

本轮创建了仓库外的 `$VLLM_WORKDIR`，所有 Python/CUDA 编译依赖均位于该
目录或其 venv；`.venv-gpu`、系统 Python、系统 CUDA 均未修改。最终环境为：

```text
Python          3.10
PyTorch         2.9.1 (Jetson native, CUDA 12.6)
Triton          3.5.1
FlashInfer      0.5.3
Transformers    4.57.6
Qwen-ASR        0.0.6
vLLM            0.14.0+cu126
```

Jetson Containers 的 Docker 构建首先因宿主 BuildKit 0.18.2 不支持其 device entitlement
而停止。为核验该路线曾安装 docker-buildx 0.30.1 plugin，但 daemon 的 BuildKit backend
仍为 0.18.2；除此之外没有升级 JetPack/CUDA 或替换系统运行库。随后复用官方 Jetson
SM87 patch 思路做 native build。
上游默认 build graph 有 365 个对象，其中 287 个来自内置 FlashAttention 2，不适合这次
已知无需 FA2 的 dense batch-1 验证。实验性 minimal dense build 做了以下限定：

1. `CUDA_HOME=/usr/local/cuda`、`TORCH_CUDA_ARCH_LIST=8.7`、低并发编译；
2. 保留 `vllm._C`、cumem allocator 和 Triton kernels；
3. 省略 MoE 扩展和内置 FA2；SM87 不构建 Hopper FA3/FlashMLA；
4. `fa_utils` 在内置 FA2 不存在时允许回退，audio encoder 显式使用 Torch SDPA；
5. 原 checkpoint 不修改；仅建立 tokenizer metadata overlay，把 Transformers 5.13 的
   `model_specific_special_tokens` 映射为官方 vLLM/Transformers 4.57.6 读取的
   `extra_special_tokens`。

产物：

```text
vllm-0.14.0+cu126-cp310-cp310-linux_aarch64.whl
sha256 a40fd05732f7ef5f40c55bb5e82137b252ee0216f187979faf6e32bf585baa3b
vllm/_C.abi3.so 94910296 bytes
```

native `_C`、SM87 CUDA op、FlashInfer 和 Qwen backend import 均通过。构建清单保存在
[`vllm014_native_sm87_build.json`](../benchmarks/orin_nx_2026-08-10_runtime_optimizations/vllm014_native_sm87_build.json)。

## 运行时边界

引擎能够完成 config/model/processor 注册，输出如下关键状态后进入权重加载：

```text
Resolved architecture: Qwen3ASRForConditionalGeneration
dtype=torch.float16, max_seq_len=1024
Cudagraph is disabled under eager mode
Using TORCH_SDPA for MMEncoderAttention
Starting to load model .../Qwen3-ASR-0.6B-hf-vllm
```

以下四种配置均被 Linux OOM killer 终止：

| 进程模式 | GPU memory utilization | Max model len | 结果 |
|---|---:|---:|---|
| V1 engine subprocess | 0.5 | 2048 | load-time OOM |
| V1 engine subprocess | 0.2 | 2048 | load-time OOM |
| same process | 0.2 | 2048 | load-time OOM |
| same process + eager | 0.1 | 1024 | load-time OOM |

最后一次已将 context 与 KV 预算降至很小，仍在权重加载阶段失败，所以目前不是 KV cache
容量导致。内核日志中被杀进程的 virtual memory 约 22.9 GB，而运行前约有 11 GB
available RAM。没有为了 benchmark 停止桌面或机器人服务。完整失败摘要见
[`vllm014_fp16_runtime.failure.txt`](../benchmarks/orin_nx_2026-08-10_runtime_optimizations/vllm014_fp16_runtime.failure.txt)。

## 下一步可行路线

无需再证明 SM87 能否编译。下一步应先压低 load-time peak，再谈 CUDA Graph：

1. 在无机器人负载的专用启动配置或 32GB Orin 上复测同一 wheel，判断是否纯容量问题；
2. 检查 safetensors loading 期间是否同时持有重复 state dict / conversion buffer；
3. 研究 vLLM load-format、CPU offload 或已离线量化 checkpoint，所有方案仍在隔离环境；
4. 只有成功转写后，再对 `enforce_eager=true/false` 做同口径 A/B，确认 CUDA Graph
   实际捕获并比较首次 capture 与稳态 latency。

仓库已加入 [`benchmark_qwen3_vllm.py`](../tests/acceptance/benchmark_qwen3_vllm.py)，可在内存
问题解决后直接生成 JSON、逐次 CSV 和 Markdown，避免改变口径。

## 投入判断

native 编译风险已经解除，但 16GB load-time peak 是新的硬边界。若目标只是当前设备的
低延迟，已实测的 Transformers static-cache compiled decode（451.6–494.3 ms，
31.6–34.3 tokens/s）应作为近期路线。vLLM 仍值得在内存峰值解决后做时间盒验证；
在得到端到端数据前，既不能承诺收益，也不能宣布方案无效。即使成功，也应作为可选
runtime，不替换稳定 Transformers/SenseVoice 环境。
