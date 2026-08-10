# Qwen3-ASR 在 Orin NX 上的 generation runtime 优化实测

## 结论先行

前一轮 profiler 对瓶颈的判断成立，但“直接给 `model.generate()` 外层套
`torch.compile(mode="reduce-overhead")`”并不是可用实现。真正有效的是让
Transformers 用 static KV cache 驱动它自己的编译 decode 路径。

在固定 5.592 秒中文音频、相同 15-token 输出下，隔离的 Jetson-native PyTorch 2.9.1
和 Triton 3.5.1 得到：

| Variant | Mean (ms) | P95 (ms) | RTF | Generate tokens/s | 相对 2.9 eager |
|---|---:|---:|---:|---:|---:|
| PyTorch 2.9 eager, FP16 SDPA | 1394.7 | 1408.7 | 0.2494 | 10.89 | baseline |
| decoder compile, `default` | 567.2 | 570.8 | 0.1014 | 26.90 | **-59.3%** |
| static cache + auto compile | **451.6** | **460.1** | **0.0808** | **34.27** | **-67.6%** |
| static cache, fresh-process repeat (5 runs) | 494.3 | 572.4 | 0.0884 | 31.56 | **-64.6%** |
| decoder NF4 W4A16 | 1897.8 | 1970.8 | 0.3394 | 8.00 | +36.1% |
| NF4 + static-cache request | 2163.7 | 2219.7 | 0.3869 | 7.03 | +55.1% |

所有成功 variant 的转写完全相同。这里比较的是同一个 PyTorch 2.9.1 实验栈；已经封存
的 PyTorch 2.5 FP16 稳定 baseline 仍是 1218.3 ms，不因本实验被覆盖。PyTorch 2.9
eager 本身比它慢约 14.5%，但 static-cache 最优值仍比封存 baseline 低约 62.9%。

## CUDA Graph / `torch.compile`

隔离环境使用 Jetson-native wheel，而不是 generic aarch64 server wheel：

- PyTorch 2.9.1, CUDA 12.6
- Triton 3.5.1
- Orin SM87 kernel smoke test passed
- cuDSS 0.7.1 runtime 仅解压在隔离目录，没有安装到系统

简单 CUDA 函数的 `reduce-overhead` 编译和执行成功，证明基础环境可用。真实 ASR 上：

1. 编译完整 multimodal forward 会在 audio frontend 的 `.item()` / `tolist()` 产生 graph
   break，并触发 CUDA Graph 输出覆盖错误。
2. 只编译重复执行的 language decoder 可消除 audio graph break，但 KV-cache / 返回张量
   仍触发同一个输出覆盖错误。每次 forward 前调用
   `torch.compiler.cudagraph_mark_step_begin()` 仍不能跨一次调用内部的 graph partitions
   修复别名生命周期。
3. 改用 `mode="default"`、即不强求 CUDA Graph 的 Inductor decoder 编译成功，稳态
   567.2 ms。首次编译请求为 180.6 秒，不计入稳态。
4. `cache_implementation="static"` 让 Transformers 5.13 使用 compilable cache 和它的
   `get_compiled_call()` decode 路径。该路径默认也是 `reduce-overhead`，但由 generation
   runtime 管理 cache/step 生命周期，因此成功达到 451.6 ms。首次请求为 49.9 秒，
   不计入稳态。
5. 使用相同 Inductor cache 启动全新 Python 进程后，首次请求仍需 17.56 秒，随后 5 次
   稳态平均 494.3 ms、31.56 tokens/s。说明已编译 artifact 能缩短 cold start，但不能
   消除首次请求编译；稳态收益可跨进程复现。

所以 CUDA Graph 建议的方向正确，但集成边界很重要：**static cache + generation-aware
compiled decode 有效；手工编译整个 Qwen3-ASR forward 无效。**

### 实时可变长度语音的边界

上面的 451.6 ms 是固定 5.592 秒 WAV、固定输出 shape 在充分预热后的结果。实时麦克风
输入长度不断变化；如果仍使用默认的静态 shape 编译，新的 utterance shape 会触发新的
Inductor specialization。一次前台实测中，首个 1.16 秒输入因此耗时 81.4 秒，后续输入
又进入重新编译；同时运行环境没有把 CUDA SDK include 目录传给 Triton，最终以
`cuda.h: No such file or directory` 失败。这不是 ASR/VAD 停止工作，而是编译路径没有按
实时 workload 配置。

工程现已提供独立的 `qwen3_asr.compile_dynamic` 开关。配合 static cache 时，它把
`transformers.CompileConfig(dynamic=True)` 交给 generation runtime，使可变音频长度
共享动态编译图。以下是同一模型进程、FP16/SDPA、从测试 WAV 截取不同长度输入的受控
验证；第一次编译不计作稳态性能：

| 顺序 | Audio (s) | Total (ms) | Transcript |
|---:|---:|---:|---|
| 1（cold compile） | 1.2 | 135,133.1 | 開放。 |
| 2 | 2.0 | 505.6 | 開放時間。 |
| 3 | 3.0 | 2,440.7 | 開放時間：早上九點。 |
| 4 | 1.6 | 395.7 | 開放時間。 |
| 5 | 1.2（重复） | 337.2 | 開放。 |

这证明动态 shape 能消除“每种音频长度重新编译一分钟”的故障，但也显示实际延迟仍随
输出 token 数量变化。实时部署必须满足三个条件：

1. `cache_implementation="static"` 且 `compile_dynamic=true`；
2. 启动环境把 `$CUDA_HOME/include` 和
   `$CUDA_HOME/targets/aarch64-linux/include` 加入 `CPATH` / `CPLUS_INCLUDE_PATH`；
3. 上线前完成一次预热，并把最长约两分钟的首次编译与稳态 latency 分开观察。

实时配置可设置 `startup_warmup_seconds=1.2`。服务会用合成音频强制生成至少 8 个
token，在打开麦克风之前完成 dynamic generation graph 的冷编译；预热结果不会发布到
DDS/ROS 2。默认值是 0，因此 SenseVoice 和未选择此功能的 Qwen 部署没有额外启动成本。
在持久 Inductor cache 已存在的全新进程中，1.2 秒预热实测约 49.9 秒；同一进程随后
识别 6.08 秒真实语音为 682.7 ms（RTF 0.1123），证明真实输入复用了预热后的动态图。

项目的 systemd GPU runner 在选择隔离 Python 时会自动设置上述源码与 CUDA include
路径。`g1-speech-service status` 只报告四个 systemd unit；直接运行
`python -m g1_speech.cli serve` 的前台进程不会显示为 active。

Nsight Systems 对 warm-up 后单次请求的 scoped capture 给出了直接证据：

| CUDA API | PyTorch 2.5 eager baseline | PyTorch 2.9 static cache | 变化 |
|---|---:|---:|---:|
| `cudaLaunchKernel` calls | 26,135 | 2,832 | **-89.2%** |
| `cudaLaunchKernel` CPU total | 513.3 ms | 65.9 ms | **-87.2%** |
| `cudaGraphLaunch` calls | 0 | 14 | graph replay 已启用 |

static-cache capture 中普通 launch 仍包括 audio frontend/prefill 和 graph 外算子，因此不会
降到零。Nsight instrumentation 把该次 wall time 扰动到 779 ms，不能与非 profiler
latency 直接比较；表中的 launch/API 差异才是这个 capture 的用途。原始 CSV 保存在
benchmark evidence 目录。

## W4A16 / bitsandbytes

PyPI 的 generic aarch64 bitsandbytes 0.49.2 wheel 安装后，在第一个 4-bit kernel 上报
`named symbol not found`，不能据此判断量化性能。本实验随后从官方 0.49.2 源码针对
CUDA 12.6 / SM87 原生编译，4-bit `Linear4bit` kernel smoke test 成功。

量化范围仅为 `model.language_model` 的 Linear 权重；audio tower、multimodal projector
和 `lm_head` 保持 FP16，避免把 acoustic 精度变化混入 runtime 试验。NF4 输出与 FP16
样本一致，但 eager latency 增至 1897.8 ms。请求 static cache 后，Transformers 因该
quantizer/custom op 不满足有效自动编译路径，没有出现 FP16 static-cache 的首次编译，
并进一步回退到 2163.7 ms。

结论是：**bitsandbytes NF4 不是 Orin NX 上这个 batch-1 ASR workload 的低延迟方案。**
减少权重流量不足以抵消逐层反量化和通用 4-bit GEMM kernel 开销。这个结果不外推到
TensorRT-LLM Marlin/AWQ 等完全不同的融合 kernel。

## vLLM 0.14.0

Qwen 官方包明确固定 vLLM 0.14.0；该版本又固定 PyTorch 2.9.1。Jetson Containers
registry 没有适用于 JP6.2 的 vLLM 0.14.0 预构建镜像，因此实验使用全新环境和官方
Jetson Containers SM87 patch generator 源码构建。构建参数限定
`TORCH_CUDA_ARCH_LIST=8.7`、CUDA 12.6、`MAX_JOBS=2`，没有修改 `.venv-gpu`、系统
CUDA 或系统 Python。

原生构建最终成功，产物为 aarch64 / CUDA 12.6 wheel，`vllm._C` 为 94.9 MB，且
SM87 CUDA op、FlashInfer 0.5.3、Qwen3-ASR vLLM backend 的导入检查均通过。为了在
16GB 设备上完成构建，实验 wheel 保留主 `_C`、cumem allocator 和 Triton kernels，
省略 MoE、内置 FA2 与 Hopper FA3/FlashMLA；audio encoder 明确选择 Torch SDPA。

不过，端到端 benchmark 仍未跑通。引擎能够解析 `Qwen3ASRForConditionalGeneration`、
选择 FP16、初始化 NCCL，并进入模型权重加载，但随后被 Linux OOM killer 终止。以下
配置均重现同一结果：

- 默认独立 engine process，`gpu_memory_utilization=0.5`；
- 独立 engine process，`gpu_memory_utilization=0.2`；
- `VLLM_ENABLE_V1_MULTIPROCESSING=0` 同进程，utilization 0.2；
- 同进程、`max_model_len=1024`、utilization 0.1。

最后一个配置已把 KV 预算和 context 降到很小，仍在权重加载阶段被杀，因此当前阻塞是
vLLM/PyTorch/模型加载的统一内存峰值，而不是 KV cache 尺寸。内核日志记录被终止进程
约 22.9 GB virtual memory；运行前系统约有 11 GB available RAM。没有停止桌面或机器人
服务来换取内存。

所以本轮结论是：**JP6 / CUDA 12.6 / SM87 可以原生构建 vLLM 0.14.0，但当前构建在
Orin NX 16GB 上装载 Qwen3-ASR-0.6B 时超过可用统一内存，CUDA Graph 与 eager 都尚无
端到端数字。** 这属于“待试/待做内存优化”，不代表 vLLM 或 CUDA Graph 没有效果。
详细版本、patch 和复现边界见 [`vllm_orin_jp6.md`](vllm_orin_jp6.md)。

## TensorRT-LLM

TensorRT-LLM 1.2.1 源码支持普通 `Qwen3ForCausalLM`，但其模型注册表没有
`Qwen3ASRForConditionalGeneration`，Qwen converter 允许的 `model_type` 列表也没有
`qwen3_asr`。完整 ASR 还需要 audio tower、projector、audio position/embedding 注入和
decoder 的联合适配，不能把普通 Qwen3 decoder 转换器当成端到端 ASR converter。

此外，1.2.1 的官方 requirements 指定 TensorRT 10.14.1，而本机 JetPack 6.2 提供
TensorRT 10.3。Jetson Containers registry 对本机只返回 TensorRT-LLM 0.12 镜像；该
2024 年版本早于 Qwen3。源码导入探测也在未安装的 1.2.1 Python 依赖处停止。由于模型
适配器本身缺失，替换系统 TensorRT 或完整构建 1.2.1 不能产生可比较的 Qwen3-ASR
latency，故没有污染系统继续强装。

这条路线不是“证明 TensorRT-LLM 没效果”，而是结论更具体：**现成 TensorRT-LLM
方案尚不能接收 Qwen3-ASR checkpoint；需要先开发并验证模型转换/运行适配器。**

## 当前工程判断

| 方案 | 状态 | 判断 |
|---|---|---|
| PyTorch 2.9 + decoder compile default | 已实测 | 有效，但 cold compile 约 181 秒 |
| PyTorch 2.9 + static cache | 已实测 | 当前最优，34.3 tokens/s |
| 手工 `reduce-overhead` CUDA Graph | 已实测失败 | cache/output alias 生命周期不兼容 |
| native SM87 bitsandbytes NF4 | 已实测 | 文本正常但显著负优化 |
| vLLM 0.14.0 native SM87 | wheel/扩展已构建；runtime OOM | 可行性已前移到模型加载，性能仍待试 |
| TensorRT-LLM 1.2.1 | 支持边界已验证 | 缺 Qwen3-ASR adapter，不能端到端 benchmark |

原始证据位于
[`benchmarks/orin_nx_2026-08-10_runtime_optimizations`](../benchmarks/orin_nx_2026-08-10_runtime_optimizations/README.md)。
