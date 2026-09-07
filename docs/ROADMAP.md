# STAR Runtime Roadmap

> 本文档是项目阶段、优先级和验收状态的唯一规划基准。
>
> 最后更新：2026-09-07
> 当前功能基线：`main`，包含至 `69a946c` 的实现提交
> 历史行动与实测结果：[PROJECT_HISTORY.md](PROJECT_HISTORY.md)

## 1. 如何使用本文档

重新开始工作前，先阅读“当前关卡”和“恢复工作检查表”。不要仅根据聊天记录、某次终端
输出或未合并分支判断项目状态。

状态只有以下六种：

| 状态 | 含义 |
|---|---|
| `DONE` | 实现、自动测试和该阶段要求的验收均已完成。 |
| `IN PROGRESS` | 已开始，仍有明确的未完成验收。 |
| `BLOCKED` | 已得到阻塞证据；解除阻塞前不应宣称完成。 |
| `PLANNED` | 已确定进入路线图，但尚未开始实现。 |
| `DEFERRED` | 有价值，但明确不属于当前阶段。 |
| `EXPERIMENTAL` | 只有实验或分支实现，不属于 `main` 的生产基线。 |

维护规则：

- `DONE` 必须链接代码、测试或硬件证据，不能只凭“看起来能用”。
- 软件正确性、硬件残余行为和真人体验必须分别记录。
- 新实验完成后，把行动和结果追加到 [PROJECT_HISTORY.md](PROJECT_HISTORY.md)。
- 阶段状态、优先级或下一步改变时，同一提交更新本文档。
- 失败实验不删除；记录其结论，避免以后重复尝试。
- 只有合入 `main` 的代码属于当前基线。其他分支必须标成 `EXPERIMENTAL`。

## 2. 项目目标与边界

STAR Runtime 的目标是在边缘机器人上提供低延迟、可打断、可组合的离线智能交互：

```text
感知/语音输入 -> Agent 推理 -> 语音/动作输出
                    |
             Role / Knowledge / Memory
```

当前核心原则：

- 单 Orin 优先采用模块化单体和进程内数据路径，避免无必要的序列化与中间服务。
- DDS 和 ROS 2 是跨进程、跨机器部署方式，不是强制的内部中间件。
- ASR、Agent、TTS 和未来推理提供方保持可替换。
- Session / Turn / Epoch 是所有可见输出的统一有效性边界。
- 实际扬声器 PCM 是 WebRTC AEC render reference 的时钟和内容权威。
- 先建立正确性和测量能力，再引入 GPU 调度、模型驻留和资源管理。

当前明确不做：

- 为架构统一而迁移 llama.cpp 或 vLLM-Omni；
- 在没有测量基线前实现统一推理调度；
- 用提高 VAD 阈值掩盖声学回声；
- 为单项优化进行新的全仓库重构。

## 3. 当前生产基线

| 领域 | 当前基线 | 证据 |
|---|---|---|
| Runtime 结构 | `star_runtime` 模块化单体；集成模式默认进程内传递；DDS/ROS 2 可选 | [runtime_architecture.md](runtime_architecture.md) |
| ASR | SenseVoice 与 Qwen3-ASR 可插拔；Orin NX 有受控延迟基线 | [README_ZH.md](../README_ZH.md)、[ASR 优化报告](qwen3_asr_orin_nx_runtime_optimization_report_en.md) |
| Agent | llama.cpp OpenAI-compatible 接口；流式 delta；可主动取消并以 epoch 校验兜底 | [runtime_architecture.md](runtime_architecture.md) |
| TTS | Qwen3-TTS 0.6B CustomVoice，经隔离的 vLLM-Omni 0.26 环境流式输出 | [qwen3_tts_vllm_omni_jp6.md](qwen3_tts_vllm_omni_jp6.md) |
| 播放 | persistent PortAudio/ALSA stream、10 ms PCM 时钟、单调 generation、最终 pre-write gate | [full_duplex_audio.md](full_duplex_audio.md)、[playback_metrics.md](playback_metrics.md) |
| AEC | WebRTC APM/AEC3；实际扬声器 PCM与显式静音共同维持连续 render timeline | [acoustic_path_diagnostic.md](acoustic_path_diagnostic.md) |
| 控制平面 | Session -> Turn -> Epoch；旧 epoch 的 Agent/TTS/PCM 不得重新可见 | [runtime_architecture.md](runtime_architecture.md) |
| 时序观测 | 每个终态 Turn 输出一条 `RUNTIME_TURN_TIMELINE` | [runtime_timing_audit.md](runtime_timing_audit.md) |
| Agent 基准 | 独立 Responses API 工具测量 input/cache/output、TTFT 和 decode rate | [agent_llm_latency_benchmark.md](agent_llm_latency_benchmark.md) |
| Role Package | 主线包含 Tifa、Olaf、Ada Wong、Babata、Link、Saber 等版本化角色包 | [role_packages.md](role_packages.md) |
| 主要硬件 | Jetson Orin NX 16 GB、JetPack 6.2、L4T R36.4.3、CUDA 12.6 | 各 Orin 实测报告 |

## 4. 里程碑总览

| ID | 里程碑 | 状态 | 已得到的结果 | 下一验收点 |
|---|---|---|---|---|
| M0 | 可移植离线语音基础 | `DONE` | CPU/GPU ASR、稳定 ALSA 输入、配置所有权、DDS/ROS 2 生命周期 | 仅做回归维护 |
| M1 | Orin ASR/TTS 生产基线 | `DONE` | ASR 受控优化完成；TTS 稳态低于实时 RTF，单用户内存下降约 2 GiB | 新模型或环境变化时重新封版 |
| M2 | Role-based Integrated Runtime | `DONE` | ASR -> Agent -> TTS 进程内流式路径和版本化 Role Package 已落地 | 维持接口兼容与角色内容测试 |
| M3 | Runtime Control Plane v1 | `DONE` | Session/Turn/Epoch、全链路失效保护、Provider cancellation、传输元数据和竞态测试完成 | 不得回退 stale-work 不变量 |
| M4 | Playback Recovery v1.1 | `DONE` | hard abort 根因被定位；persistent 策略完成 50/50 硬件恢复循环 | 保留 hard-abort 回滚与指标 |
| M5 | JP6.2 声学打断发布验收 | `BLOCKED` | 连续 render 时钟已消除约 1.25 s 缺口，但 robot-only 仍出现 1 次 VAD 边沿 | 见“当前关卡” |
| M6 | Runtime Measurement v1 | `PLANNED` | 已有 Turn timeline 和独立 Agent TTFT 工具作为基础 | M5 签字后确定指标基线和预算 |
| M7 | Robot Unified Inference Resource Manager | `DEFERRED` | 尚未开始；当前没有统一 GPU 调度或模型驻留管理 | 依赖 M6 的真实资源/延迟数据 |
| M8 | 视觉感知主线集成 | `EXPERIMENTAL` | `codex/vision-perception-port` 有有界视觉管线实现，但未合入 `main` | 独立评审范围、内存和交互契约 |

## 5. 当前关卡：M5 声学打断发布验收

这是恢复工作时唯一应优先处理的 P0。Control Plane 的 stale-work 正确性已经通过；当前
阻塞是机器人自身声音经过真实扬声器、空间和麦克风后，仍可能在句尾形成 post-AEC VAD
误触发。

### 已确认

- 物理 render -> echo 延迟约 100 ms，已观察窗口内没有显著漂移。
- 原先 WebRTC reverse stream 存在约 1.23--1.27 s 缺口。
- persistent speaker writer 现在每 10 ms 提交真实 PCM 或显式静音，最大 render 间隔已降至
  约 11.6 ms，原缺口消失。
- 在不改变 VAD、音量、TTS、Control Plane 和播放策略的情况下，robot-only 实测仍产生
  1 次 VAD edge/Control invalidation。
- fixed `stream_delay_ms` 的 60/80/100/120/140 ms A/B 均未达到零误触门槛。
- 因此“reverse-stream 缺口是唯一根因”的假设已被否定；继续盲调固定 delay 没有依据。

完整证据见 [acoustic_path_diagnostic.md](acoustic_path_diagnostic.md)。

### 完成条件

必须在同一 JP6.2 生产音频时序与固定硬件布置上同时满足：

1. `robot-only speech -> zero VAD speech edges / zero false barge-ins`；
2. `robot speech + real near-end human speech -> reliable early VAD edge`；
3. VAD edge 后立即推进 epoch、停止旧输出，旧答案不恢复；
4. replacement turn 被识别并正常播放；
5. 保存同步 render/raw/post-AEC trace、Runtime timeline 和硬件日志；
6. 不通过提高 VAD 阈值隐藏 residual echo。

### 恢复工作时的下一步

先用现有诊断工具重新建立可复现的 robot-only 失败样本，确认麦克风、扬声器、音量、房间
位置和 20 ms capture timing 与冻结实验一致。之后应针对 AEC 尾部抑制失效的实际信号证据
提出一个有界假设；如果需要改变架构或 VAD 策略，先把证据和新验收边界写入本文档，不要
直接调参。

## 6. 下一阶段：M6 Runtime Measurement v1

M6 只负责建立可以长期比较的运行时指标与采样方法，不负责 GPU 调度。建议范围：

- 固定 `speech_end -> first hardware write` 的用户等待时间定义；
- 分解 VAD tail、ASR、Agent TTFT、首个可合成文本段、TTS First PCM 和播放提交；
- 区分 Agent TTFT、完整生成时间和终端完整文本回显时间；
- 报告冷启动、热缓存、连续不同问题和并发 TTS 条件；
- 同步记录输入 token、缓存 token、输出 token、GPU/CPU/统一内存状态；
- 为正常回合、打断回合和恢复回合分别建立 P50/P95，而不是混成一个平均值；
- 所有指标链接原始 artifact、配置指纹和代码版本。

开始 M6 前需要单独确认目标预算；本文档不擅自把当前观测值变成产品 SLO。

## 7. 明确延期的工作

以下内容在 M6 之前保持 `DEFERRED`：

- GPU kernel/model 统一调度；
- 模型常驻、淘汰和预热策略管理；
- 统一内存预算、准入控制与跨模型抢占；
- llama.cpp、vLLM-Omni 或 ASR provider 的架构性迁移；
- 多机器人、多租户资源治理；
- 因视觉感知接入而进行的 Runtime 大重构。

## 8. 已知限制与风险

- JP6.2 robot-only 零误触和真人 double-talk 发布验收尚未签字。
- 已通过最终 pre-write gate 的一个硬件块属于 committed residual；它不是新的 stale hardware
  submission。边界和指标定义见 [playback_metrics.md](playback_metrics.md)。
- TTS 首次冷启动可能包含很大的 JIT 成本；稳态数字不能代表冷启动。
- Agent TTFT 对输入 token、prefix cache 和并发负载敏感；不能只比较终端完整回显时间。
- Qwen3-ASR 的 vLLM 路径在 16 GB Orin NX 上仍受模型加载内存压力限制；当前生产路径不依赖它。
- 视觉感知分支不是 `main` 基线，不应作为已发布能力引用。

## 9. 恢复工作检查表

隔几天或几周回来时，依次执行：

1. 阅读本文档的“当前关卡”和最近状态变化。
2. 阅读 [PROJECT_HISTORY.md](PROJECT_HISTORY.md) 最后两个日期段。
3. 检查 `git status`、当前分支和最近提交，不默认未提交文件属于同一任务。
4. 确认实际硬件、模型、配置文件和虚拟环境与证据基线一致。
5. 先复现最近一个成功/失败样本，再开始新的修改。
6. 新实验保存原始 artifact，并在历史账本中记录成功或失败。
7. 完成阶段后更新状态、验收证据和下一步，再提交代码。
