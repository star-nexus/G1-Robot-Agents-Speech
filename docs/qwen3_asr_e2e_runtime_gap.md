# Qwen3-ASR controlled benchmark 与 E2E 表观吞吐差异归因

## 1. Executive conclusion

**结论：`31–34 tok/s` 与 `15.3 tok/s` 不是 autoregressive decode 真实下降约 2 倍。**
本轮严格控制实验测得动态编译 runtime 的增量 decode 成本为
`28.15 ms/token`，即 **35.53 decode tok/s**，线性拟合
`R²=0.9997`。`15.3 tok/s` 来自把 audio encoder、multimodal projector、LM
prefill、首 Token、decode 和 generation bookkeeping 全部放入分母，并叠加了新 audio
shape 首次 CUDA Graph 录制/热化；它是请求级 apparent throughput，不是 decode-only
throughput。

四项直接证据共同支持该判断：

1. 同一进程、同一模型对象、同一 PCM 的 VAD-fed 与 direct replay，`generate_ms` 差异
   只有 `+0.63%` 和 `-0.64%`，均在运行波动内；E2E/VAD pipeline 没有改变 ASR compute。
2. 5/10/15/20/30/50 Token 同音频实验满足
   `generate_ms = 157.22 + 28.147 × tokens`，`R²=0.9997`。表观吞吐自然从
   5 Token 的 `16.12 tok/s` 上升到 50 Token 的 `31.86 tok/s`。
3. E2E steady-state Nsight 仍有 `9` 次 `cudaGraphLaunch`；10 Token 正好包含 1 次
   prefill/首 Token和 9 次后续 decode。普通 kernel launch 为 `2,698`，与历史 controlled
   的 `2,832` 同级，不存在恢复到数万次 eager launch 的现象。
4. 当前 E2E runtime 确实是 Jetson-native Python 3.10.12 / PyTorch 2.9.1 /
   CUDA 12.6 / Transformers 5.13.0 / Triton 3.5.1 / SM87，而不是旧 `.venv-gpu`
   PyTorch 2.5。

历史 controlled 与实时 E2E 并非完全相同配置：历史结果使用 fixed-shape compiled
decode，实时配置使用 `CompileConfig(dynamic=True)`。当代同机重跑 fixed-shape 得到
`459.1ms / 33.97 apparent tok/s`，与历史 `451.6ms / 34.27 tok/s` 重合；同输入动态配置
为 `580.1ms / 26.58 tok/s`。这项差异确认贡献约 `121ms`，但仍不构成 decode 2 倍退化。

## 2. Runtime fingerprint comparison

### E2E 没有偷跑到旧 PyTorch 环境

| Field | Historical controlled | Contemporary fixed-shape rerun | E2E dynamic runtime |
|---|---|---|---|
| Python executable | 历史 JSON 未记录 | `.venv-torch291/bin/python` | `.venv-torch291/bin/python` |
| Python | 未记录 | 3.10.12 | 3.10.12 |
| PyTorch | 2.9.1 | 2.9.1 | 2.9.1 |
| CUDA | 12.6 | 12.6 | 12.6 |
| Transformers | 历史 JSON 未记录 | 5.13.0 | 5.13.0 |
| Triton | 文档记录 3.5.1 | 3.5.1 | 3.5.1 |
| GPU | Orin / SM87 | Orin / 8.7 | Orin / 8.7 |
| dtype / attention | FP16 / SDPA | FP16 / SDPA | FP16 / SDPA |
| KV cache | static | static | static |
| compile backend | Inductor | Inductor | Inductor |
| compile mode | generation-aware | generation-aware | `reduce-overhead` |
| dynamic shapes | false | false | **true** |
| max new tokens | 256 | 256 | 256 |
| sampling | greedy | greedy | greedy |

E2E 进程实际模型类为
`Qwen3ASRForConditionalGeneration`，language model 为 `Qwen3Model`。同进程 A/B
所有行记录了完全相同的 engine object id `281472724582352` 和 model object id
`281469735719296`，证明不是重复加载或不同常驻实例。

E2E 的实际 compile 配置为：

```text
backend=inductor
mode=reduce-overhead
dynamic=true
fullgraph=false
cache_implementation=static
```

启动方式也一致指向隔离环境：`deploy.env` 的 `SPEECH_PYTHON_GPU` 为
`.venv-torch291/bin/python`，生产入口为 `scripts/run-speech-service gpu dds`。本轮
benchmark 直接用同一解释器和相同 CUDA/Inductor 环境变量启动，不涉及 systemd 或麦克风。
完整 fingerprint 位于
[`runtime_fingerprint.json`](../benchmarks/orin_nx_2026-08-11_runtime_gap/runtime_fingerprint.json)，
三组对照位于
[`runtime_comparison.json`](../benchmarks/orin_nx_2026-08-11_runtime_gap/runtime_comparison.json)。

## 3. Same-input direct-vs-E2E A/B

### 相同 PCM 进入 ASR 后没有路径差异

实验在一个 Python 进程、一个 CUDA 线程、一个 warm model instance 中交错运行，每个
case warm-up 3 次、正式 10 次。`VAD-fed` 表示使用真实 Silero VAD 产生的
`Utterance`；`direct replay` 使用完全相同的 NumPy PCM buffer 重新构造
`Utterance` 后直接送入同一 engine。所有请求的 dtype、attention、cache、generation
config 和 synchronized stage profiling 均相同。

| Input/path | PCM samples | Tokens | Generate mean | Total mean | Apparent tok/s |
|---|---:|---:|---:|---:|---:|
| Controlled full direct | 89,472 | 15 | 564.4ms | 580.1ms | 26.58 |
| Controlled VAD-fed | 78,368 | 15 | 569.6ms | 582.9ms | 26.35 |
| Same controlled PCM direct replay | 78,368 | 15 | 566.0ms | 579.0ms | 26.51 |
| Real VAD-fed | 52,608 | 10 | 408.1ms | 419.5ms | 24.53 |
| Same real PCM direct replay | 52,608 | 10 | 410.7ms | 422.4ms | 24.38 |

配对 run 的 controlled VAD-fed 减 direct 为 `+3.57ms`（`+0.63%`），real VAD-fed
减 direct 为 `-2.61ms`（`-0.64%`）。差值方向相反，且小于 run-to-run 范围，因此没有
证据表明 VAD 或 pipeline metadata 会降低 generate 性能。生产 E2E 的 ASR queue 先前
实测仅 `0.1ms`；queue/VAD 会影响 speech-end-to-final，但不进入 `generate_ms`。

Controlled WAV 经 VAD 后从 5.592s 变为 4.898s，real 6s 文件的首个 utterance 为
3.288s。提取 WAV 仅用于本机 exact-PCM replay，没有提交音频内容；sample count 和
SHA-256 保存在
[`extracted_audio_manifest.json`](../benchmarks/orin_nx_2026-08-11_runtime_gap/extracted_audio_manifest.json)。

## 4. Fixed-cost vs decode-cost decomposition

### 15.3 tok/s 的分母混入了约 157ms 固定 generation 成本

精确的 first-token callback 会强制逐步 device-to-host 同步并扰动 compiled decode，故本轮
没有伪造 TTFT。替代实验固定同一份 30.313s PCM、同一 model/cache/config，仅改变
`max_new_tokens`；每个长度 warm-up 2 次、正式 10 次。低上限输出是预期截断，不用于
识别质量判断。

| Generated tokens | Generate mean | Apparent tokens/s | First request generate |
|---:|---:|---:|---:|
| 5 | 310.3ms | 16.12 | 612.4ms |
| 10 | 435.1ms | 22.99 | 526.9ms |
| 15 | 570.7ms | 26.28 | 664.9ms |
| 20 | 720.0ms | 27.78 | 807.9ms |
| 30 | 996.7ms | 30.10 | 1,100.6ms |
| 50 | 1,569.7ms | 31.86 | 1,647.6ms |

OLS 回归结果：

```text
T_generate = 157.216ms + 28.147ms × N_generated_tokens
R² = 0.999725
estimated incremental decode throughput = 35.53 tokens/s
```

这里的 `157.216ms` 是外推的固定 generation 成本，包含 audio encoder、multimodal
projector、LM prefill、首 Token 和固定 bookkeeping；它不是直接测得的 TTFT。斜率是
每增加一个生成 Token 的增量成本，因此是本轮最可靠的 decode-only 近似。

这个模型也直接解释指标错觉：10 Token 请求即使完全稳态，固定成本仍占明显比例，表观
吞吐约 23 tok/s；输出变长后固定成本被摊薄，50 Token 已接近 32 tok/s，而增量 decode
本身约 35.5 tok/s。原始 15.3 tok/s 还叠加了新 shape 第一次 graph 录制和单次测量，不能
代表逐 Token decoder。

## 5. CUDA Graph / Nsight comparison

### E2E steady-state 正常 replay CUDA Graph

Nsight 只在 warm-up 后用 CUDA profiler API 捕获一个 3.288s E2E-extracted 请求。Nsight
会显著扰动 wall time，因此本节只用 launch structure 判断 compiled path，不把 profiler
wall time当作正常 latency。

| Metric | Historical controlled, 15 tokens | E2E dynamic, 10 tokens |
|---|---:|---:|
| `cudaLaunchKernel` | 2,832 | 2,698 |
| ordinary launch CPU API | 65.87ms | 66.09ms |
| `cudaGraphLaunch` | 14 | **9** |
| GPU kernel count | 2,834 | 2,700 |
| GPU kernel total time | 170.45ms | 136.27ms |
| profiled generate NVTX | 746.22ms | 634.56ms |

`cudaGraphLaunch = generated_tokens - 1` 在两组数据中都成立：首 Token 由
audio/prefill 路径产生，后续 decode step replay graph。E2E 没有出现 `cudaGraphLaunch=0`
或 2.6 万普通 launch，因此“未命中 compiled decode / 回退 eager”已排除。

原始 API、kernel、NVTX CSV 和统一对比位于
[`nsight_summary.csv`](../benchmarks/orin_nx_2026-08-11_runtime_gap/nsight_summary.csv)。

## 6. Recompile / shape analysis

### 动态图没有 recompile 或 graph break，但新 audio shape 有一次性 graph 录制成本

处理器 shape 随音频变化：

| Input | Samples | `input_ids` | Audio features |
|---|---:|---:|---:|
| Controlled full 5.592s | 89,472 | `[1,89]` | `[1,128,600]` |
| Controlled VAD 4.898s | 78,368 | `[1,80]` | `[1,128,500]` |
| Real VAD 3.288s | 52,608 | `[1,59]` | `[1,128,400]` |
| Scaling 30.313s | 485,013 | `[1,410]` | `[1,128,3100]` |

对真实 3.288s 请求的诊断 hook 记录 10 次
`prepare_inputs_for_generation`：第一次为 prefill，`input_ids=[1,59]` 并携带
`input_features=[1,128,400]`；其后 9 次均为 `input_ids=[1,1]` decode。该模型的
prepared dict 没有暴露 `cache_position`，所以没有推测或伪造其 shape。hook 本身不参与
性能汇总。

独立进程开启 `TORCH_LOGS=recompiles,graph_breaks` 后：

- 日志中没有 recompile 或 graph-break 条目；
- `_compiled_call` 在所有请求中持续存在；
- 新的 3.288s、5.592s、30.313s shape 各使
  `cudagraph_recorded_non_static_inputs` 累计增加 5；
- 反向重复相同 shape 时该计数不再增加。

相应首次/重复 generate 为：3.288s `600.8 → 394.5ms`，5.592s
`802.6 → 548.1ms`，30.313s `3540.5 → 3174.6ms`。因此 dynamic shape 没有发生分钟级
recompile 或 eager fallback，但 startup synthetic 1.2s warm-up 并不能消除每种新 audio
shape 的首次 CUDA Graph 录制/allocator/cache 热化。这是原始单次 654.3ms 的重要来源。

## 7. Profiling overhead A/B

### synchronized stage profiling 没有可测的 steady-state penalty

对同一 3.288s PCM，在同一模型中交错执行 profiling off/on，各 warm-up 3、正式 10 次：

| Mode | Total mean | Median | P95 | Text |
|---|---:|---:|---:|---|
| `log_profile=false` equivalent | 435.12ms | 434.16ms | 449.96ms | 一致 |
| synchronized stage profile | 434.19ms | 431.60ms | 452.63ms | 一致 |

差值为 `-0.93ms`（`-0.21%`），属于噪声，不支持 profiling 导致 2 倍差异。profiling-off
case 故意不提供 `generate_ms`：在关闭 stage boundary synchronization 的控制组里重新
发明一个 generate boundary 会破坏该控制实验。

## 8. Root-cause ranking

| Candidate | Verdict | Evidence |
|---|---|---|
| 1. E2E 使用不同 PyTorch/Triton runtime | **Ruled out** | 实际 fingerprint 为 2.9.1 / 3.5.1 / CUDA 12.6 / SM87 |
| 2. E2E 未命中 static-cache compiled decode | **Ruled out** | `static`、`_compiled_call` 持续存在，launch 数约 2.7k |
| 3. CUDA Graph 没有 replay | **Ruled out** | 10 Token 请求有 9 次 graph launch |
| 4. Dynamic shape 触发 recompile / eager fallback | **Fallback ruled out; first-shape recording confirmed** | 无 recompile/graph-break log；新 shape graph counter +5，重复不再增加 |
| 5. 10-Token 固定 audio/prefill 成本占比高 | **Confirmed** | 固定成本估计 157.2ms；短输出表观 tok/s 显著较低 |
| 6. `tokens / entire generate` 指标误导 | **Confirmed, primary** | 增量 decode 35.53 tok/s，表观吞吐随 N 从 16.12 升至 31.86 |
| 7. synchronized profiling overhead | **Ruled out** | `-0.21%`，文本一致 |
| 8. VAD/E2E pipeline 影响 ASR compute | **Ruled out** | 同 PCM direct-vs-VAD-fed 差异绝对值小于 1% |
| 9. GPU frequency / thermal / concurrent workload | **Unlikely for 2×; not fully ruled out** | 无并发 ASR 服务，MAXN_SUPER，测试后 46–53°C；未锁频、未记录逐 run clocks |
| 10. model/cache 没有真正 warm | **Confirmed for first/new-shape requests** | first-shape 600–803ms，重复 394–548ms；稳态 A/B 显著更快 |
| 11. generation config 不一致 | **Confirmed, secondary** | historical fixed-shape vs E2E dynamic；同输入总延迟 459.1 vs 580.1ms |

根因按对原始 `34 → 15.3` 观测的重要性排序：

1. **指标定义与短输出固定成本摊薄**；
2. **单次请求尚未达到该 audio shape 的完整 steady state**；
3. **fixed-shape historical baseline 与 dynamic live config 不同**；
4. 其余 pipeline、profiling、CUDA Graph 丢失和 runtime 偷换均被数据排除。

## 9. Final answer: Is 15.3 tok/s a real decode regression?

**不是。** 15.3 tok/s 是 `10 / (audio encoder + projector + prefill + first token +
9 decode steps + bookkeeping)` 的请求级比值，并叠加新 shape 首次 graph 录制。相同
3.288s PCM 达到 steady state 后为约 `24.4–24.5 apparent tok/s`；固定音频的 Token
scaling 则估计真实增量 decode 为 **35.53 tok/s**。目前没有任何证据显示 autoregressive
decoder 退化约 2 倍。

本轮不提出新的性能优化。下一步如果要继续测量，只应把 `apparent generate tok/s` 改名为
request-level 指标，并把回归得到的 incremental decode tok/s 与 first-shape/steady-state
状态同时记录；是否改变 dynamic compile 或 warm-up 策略属于后续独立优化决策，不应混入
本次归因结论。

## Scope, methodology, limitations, and reproducibility

- 所有 steady-state A/B 正式样本均为至少 10 runs；cold compile、warm-up 和正式样本分开。
- 同输入 A/B 采用交错/反向顺序，降低温度与顺序偏差。
- regression 是诊断性 OLS，不是对不同音频内容的因果模型；它固定同一 PCM，仅改变输出上限。
- 无侵入 TTFT、audio encoder 与 LM prefill 的独立 wall-time 未可靠获得；报告只给合并固定成本，
  不把 profiler kernel 分类伪装成阶段时长。
- Nsight wall time受 instrumentation 扰动，仅用于 launch structure。
- 没有停止系统服务；检查时没有运行中的 speech service。未锁 GPU clocks，因此几十毫秒级
  波动仍可能包含 DVFS，但不能解释已经被同进程 A/B 排除的 2 倍路径差异。
- 没有绘制趋势图：本报告只有六个受控 Token anchor，精确值、回归公式和 `R²` 表格比视觉
  拟合更便于审计；原始 CSV 可直接用于后续作图。

复现命令、原始 JSON/CSV、recompile log 和 artifact inventory 见
[`benchmarks/orin_nx_2026-08-11_runtime_gap/README.md`](../benchmarks/orin_nx_2026-08-11_runtime_gap/README.md)。

## Recommended next steps and open questions

本轮归因完成后，建议只做测量治理：

1. 将现有 `generated_tokens_per_second` 在日志/报告中明确标注为
   `request_apparent_tokens_per_second`；保留原字段以避免 API 破坏可在后续单独设计。
2. 性能验收同时报告 first-shape、third-request 和 10-run steady-state，而不是单点。
3. 若必须获得真实 TTFT，再设计不引入逐 Token D2H 同步的 CUDA event/NVTX 专项实验；在此
   之前不要将回归 intercept 宣称为直接 TTFT。

尚未回答但不改变当前结论的问题：动态编译比 fixed-shape 多出的约 121ms 分别来自哪些
graph partition，以及能否在保持可变音频 shape 的前提下消除首次 graph 录制。这些属于
下一轮优化问题，不属于本轮“15.3 是否为 decode 退化”的归因范围。
