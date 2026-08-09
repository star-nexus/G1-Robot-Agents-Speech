# Qwen3-ASR-0.6B 在 Jetson Orin NX 上的延迟定位

## 技术摘要

结论已经由实测支持：**1.2 秒级延迟几乎全部位于 Transformers `generate()`，
主要矛盾是自回归 decoder 的 GEMM、内存访问和大量逐 token CUDA launch；attention
不是主因，预处理也不是主因。** 固定 5.592 秒 WAV 的同步拆解为 1281.5 ms，其中
`generate()` 1266.5 ms（98.8%），processor、H2D、decode 合计仅 14.8 ms。

Nsight Systems 在一个 15-token 请求的 `generate` NVTX 范围内记录到 26,137 个 GPU
kernel 和 26,135 次 `cudaLaunchKernel`。GEMM 占 GPU kernel 时间约 65.2%，attention
kernel 约 2.35%；86.2% 的 kernel 实例平均运行时间低于 50 μs。这个结构与
batch-1 autoregressive decode 的 launch/memory-bound 特征一致。

当前 Orin 最佳安全配置改为 **GPU + SDPA + FP16**：同文本、同输入下 mean 1218.3 ms，
比 BF16 profiling baseline 快 4.9%。`torch.compile(mode="reduce-overhead")` 和 static
cache 都在首个请求失败，因为现有 Jetson PyTorch 没有可工作的 Triton；没有安装通用
aarch64 wheel，也没有掩盖失败。

## 1. 5.592 秒请求中，时间花在哪里

测试条件：Qwen3-ASR-0.6B、GPU、BF16、SDPA、warm-up 3、正式 10 次。每个 CUDA
segment 边界显式 synchronize；mean/median/P95 均来自稳态正式运行。

| Segment | Mean (ms) | Median (ms) | P95 (ms) | Mean 占 total |
|---|---:|---:|---:|---:|
| processor | 12.2 | 12.2 | 15.8 | 0.95% |
| H2D | 2.3 | 1.7 | 6.8 | 0.18% |
| generate | 1266.5 | 1266.2 | 1276.1 | **98.83%** |
| decode | 0.3 | 0.3 | 0.4 | 0.03% |
| total | **1281.5** | **1279.5** | **1295.5** | 100% |

生成 15 tokens，`generate` 吞吐 11.84 token/s，RTF 0.2292。首次 cold request 为
2065.1 ms，已经作为 warm-up 记录，没有混入稳态统计。分段之和与 total 的微小差异
来自 Python 控制流和计时器本身。

这组数据不能在不大改 Transformers generation 的情况下可靠拆开 audio encoder、
prefill 和 AR decode，因此报告不伪造三者的精确毫秒数。下一节的 scaling 和 profiler
给出可审计的归因证据。

## 2. 长度扫描显示延迟跟输出 token 数几乎线性

受控扫描把同一个 5.592 秒 PCM 样本无损拼接为 1×/2×/4×/8×。每组 warm-up 3、
正式 10 次；保持模型、precision、attention 和 generation 参数不变。

| 音频 | 时长 (s) | Mean latency (ms) | RTF | Generated tokens | Tokens/s |
|---|---:|---:|---:|---:|---:|
| 1× | 5.592 | 1257.9 | 0.2249 | 15 | 12.03 |
| 2× | 11.184 | 2155.8 | 0.1928 | 26 | 12.13 |
| 4× | 22.368 | 3955.8 | 0.1768 | 48 | 12.22 |
| 8× | 44.736 | 6158.3 | 0.1377 | 74 | 12.10 |

RTF 随长度下降 38.8%，说明短音频确实更难摊薄请求级成本。但更关键的是：四个长度的
生成吞吐都约 12 token/s。用 mean latency 对 generated tokens 做线性回归，斜率为
83.02 ms/token、截距 -1.06 ms、R²=0.99991；对 audio seconds 回归则是约 782 ms
截距和 124 ms/audio-second、R²=0.983。这里的 782 ms 只是描述性截距，包含最短请求
固有的输出 token 与模型执行，不能解释成纯 preprocessing。直接可测的
processor+H2D+decode 只有约 14.8 ms。

用户提供的自然音频扫描是另一组 workload，不能与重复拼接混为同一条曲线：

| 文件 | 解码时长 (s) | Runs | Mean latency (ms) | RTF | Tokens | Tokens/s |
|---|---:|---:|---:|---:|---:|---:|
| test_audio_6s | 6.079 | 10 | 1040.0 | 0.1711 | 12 | 11.65 |
| test_audio_30s | 30.313 | 10 | 7983.8 | 0.2634 | 98 | 12.31 |
| test_audio_60s | 60.436 | 10 | 16118.6 | 0.2667 | 202 | 12.57 |
| test_audio_120s | 120.425 | 5 | 20415.7 | 0.1695 | **256** | 12.60 |

120 秒样本正好达到 `max_new_tokens=256`，文本被上限截断，因此它较低的 RTF 不能用于
推断完整长音频吞吐。6/30/60 秒的差异也说明 RTF 强烈受语速、文本 token 密度和内容
影响；受控拼接表才是固定内容的长度效应证据。

## 3. FP16 有小收益，compile/cache 在当前栈不可用

| Variant | Mean (ms) | Median (ms) | P95 (ms) | RTF | 文本验证 | 结果 |
|---|---:|---:|---:|---:|---|---|
| SDPA + BF16 | 1281.5 | 1279.5 | 1295.5 | 0.2292 | 正常、稳定 | profiling baseline |
| SDPA + FP16 | **1218.3** | **1216.0** | **1244.7** | **0.2179** | 与 BF16 完全相同 | **快 4.9%** |
| BF16 + `torch.compile(reduce-overhead)` | — | — | — | — | 首次请求前失败 | 无可工作 Triton |
| BF16 + static cache | — | — | — | — | 首次请求前失败 | generate 自动进入 compile，同样缺 Triton |

为避免一个容易误判的问题，adapter 编译的是 `model.forward`，而不是只包装 module；
后者的 proxied `generate()` 可能继续调用原始 forward，看似“compile 成功”但实际上没有
编译。真实 forward compile 出现 graph break 后进入 Inductor，并明确报错
`Cannot find a working triton installation`。失败 trace 已原样保存；没有启用
`suppress_errors` 伪装成成功，也没有安装会替换 Jetson PyTorch 的 wheel。

后续 10 条、50.188 秒的带标注语料验收覆盖普通话、英语和粤语。忽略 Unicode 标点的
内容 CER 在 BF16 与 FP16 下均为 **6.22%（13/209）**；10/10 条内容转写一致，语言
识别 10/10 正确，每条重复三次均确定。严格 CER 分别为 14.22% 和 13.27%，差异来自
BF16 多输出两个逗号，不是词汇回退。因此 **GPU + SDPA + FP16 正式封存为 Orin NX
稳定 baseline**。社区仓库只保留聚合指标、判定门槛和模型/运行时指纹；探索性语料、
参考文本和逐条输出因不适合作为社区 benchmark 而不入库。见
[`orin_nx_2026-08-10_fp16_baseline`](../benchmarks/orin_nx_2026-08-10_fp16_baseline/README.md)。

## 4. Profiler 确认 GEMM 和 launch 主导

Torch profiler 成功给出 CPU/operator 与显存数据，但 CUDA activity 因 CUPTI
`INSUFFICIENT_PRIVILEGES` 缺失。没有修改系统 profiler 权限；改用已安装的 Nsight
Systems 在无 sudo 条件下完成 CUDA/NVTX trace。

| Nsight 指标（单次 generate） | 实测 |
|---|---:|
| NVTX generate wall time | 2130.1 ms（profiling overhead 下） |
| GPU kernel instances | **26,137** |
| `cudaLaunchKernel` calls | **26,135** |
| GPU kernel summed time | 1130.6 ms |
| GEMM kernel time share | **65.2%** |
| Attention kernel time share | **2.35%** |
| Copy/cat kernel time share | 14.1% |
| Avg < 50 μs 的 kernel 实例 | **22,523 / 86.2%** |
| `cudaLaunchKernel` CPU API time | 513.3 ms，平均 19.6 μs/call |
| Peak allocated / reserved VRAM | 1.62 / 1.68 GB |

前三类 BF16 GEMM 已占 kernel 时间约 56.9%，其中有 15 个单次约 10 ms 的 sliced GEMM。
相比之下，主要 SDPA flash kernel 只占约 2.1%，所有匹配 attention/flash forward 的
kernel 合计约 2.35%。这与 eager→SDPA 仅改善约 10%、FA2 反而回退的宏观结果一致。

26k 次 launch 和大量短 kernel 说明 ARM CPU→CUDA 调度开销不可忽略。kernel 累计时间
与 `generate` wall time 之间还有约 1 秒差额，其中包含 launch/API、同步、memcpy 和
GPU idle gap；未运行 Nsight Compute 内存计数器，所以“memory-bound”属于结合
batch-1 自回归架构和 copy/cat 数据作出的强推断，不冒充直接带宽测量。

## 5. 与官方 RTF 的比较边界

[Qwen3-ASR 技术报告](https://arxiv.org/abs/2601.21337)说明官方效率实验使用约 2 分钟
音频、vLLM 0.14.0、CUDA Graph 和 BF16；offline 使用 batch generation，online 使用
vLLM Serve。报告只写“single typical computing resource”，没有公开 GPU 型号。官方
0.6B concurrency=1 offline RTF 0.00923 对应约 1.11 秒/120 秒音频，而本机原始稳定
SDPA 基线是约 1.24 秒/5.592 秒音频。两者绝对请求时间相近，但 workload、硬件和
runtime 完全不同，不能把约 24× RTF gap 归因于一个 attention kernel。

模型架构也决定它不应与 SenseVoice 期待同一延迟级别。技术报告描述 Qwen3-ASR-0.6B
由约 180M AuT encoder、projector 和 Qwen3-0.6B LM 组成；后端仍要逐 token 生成。
SenseVoice 是专用 ASR ONNX 路径，本项目 GPU FP32 RTF 0.0138，仍是机器人低延迟口令
的更经济选择。

## 6. 最终归因与部署建议

| 候选来源 | 证据强度 | 判断 |
|---|---|---|
| Autoregressive decoder | 很强 | token 数解释 latency 的 R²=0.99991；约 12 token/s |
| MLP/GEMM | 很强 | Nsight kernel 时间约 65.2% |
| CUDA launch overhead | 很强 | 单请求约 26k launches，CUDA API 累计约 513 ms |
| Memory traffic/bandwidth | 中强（推断） | batch-1 AR、copy/cat 14.1%、大量瘦 GEMM；无带宽计数器 |
| Audio encoder/prefill | 未单独量化 | 包含在 generate；不能给出可靠精确毫秒数 |
| Attention | 强反证 | kernel 时间约 2.35%，FA2 不增反降 |
| Preprocessing/H2D/decode | 很强反证 | 合计约 14.8 ms / 1.15% |

建议：

1. 低延迟机器人交互继续优先 SenseVoice；需要 Qwen3-ASR 鲁棒性/语言知识时才切换。
2. Qwen3-ASR 在 Orin NX 上采用已通过 CER gate 的 SDPA + FP16 baseline，固定样本延迟
   比 BF16 低约 4.9%。
3. 不再投入 FA2 微调。下一轮优化应面向 generation runtime/CUDA Graph/流式首 token。
4. vLLM 只能在独立 Jetson PyTorch 2.9.1 容器栈中验证，详见
   [`vllm_orin_jp6.md`](vllm_orin_jp6.md)。

## 方法、限制与验证状态

- 这是单台 Orin NX、单模型、单 batch、有限音频的描述性实验，不是总体性能保证。
- 动态时钟未锁定；正式分布较紧，但不同独立进程仍有数个百分点漂移。
- 分段同步会引入少量测量扰动；所有 variant 使用同一口径，比较仍然成立。
- 120 秒自然音频达到 token 上限，已从完整长音频结论中排除。
- CER gate 只有 10 条、50.188 秒，适合验证 FP16 相对 BF16 无回退，不代表生产语料的
  绝对准确率；普通话、噪声、远场和机器人领域词仍需扩充人工标注集。
- Torch profiler 的 CUDA activity 权限失败已明确披露；CUDA 结论来自 Nsight CSV。
- 所有 headline 数字均从保存的 JSON/CSV 重新计算，并检查了单位、RTF 分母、run 数、
  token cap 和同文本条件。FP16 相对 BF16 的 baseline 判定为 **Ready to use on this
  Orin NX stack**；模型总体准确率结论仍为 **Share with caveats**。

完整原始数据、失败 trace、命令和来源说明位于
[`benchmarks/orin_nx_2026-08-09`](../benchmarks/orin_nx_2026-08-09/README.md)。
