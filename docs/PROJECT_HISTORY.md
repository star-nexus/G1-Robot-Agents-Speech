# STAR Runtime Project History

> 本文档是项目行动、实验结果和架构决策的只追加账本。
>
> 当前计划与恢复入口：[ROADMAP.md](ROADMAP.md)
> 首次整理日期：2026-09-07

## 1. 记录规则

本文档解决的是“以前做了什么、为什么这么做、结果如何”。它不替代 Git，也不把每个小提交
逐条复制进来。

- 已发生的记录原则上只追加，不因后续方案改变而删除。
- 如果旧结论被推翻，新增一条“修正/取代”记录，并链接旧结论。
- 每条重要记录至少包含：问题或目标、行动、结果、决策、证据。
- `main` 提交和未合并实验分支分开记录。
- 受控基准、软件测试、自动硬件测试和真人验收是四种不同证据，不能互相替代。
- 本次首次整理根据 Git 历史和仓库内报告回填；无法从持久证据确认的聊天细节不写入。

## 2. 历史摘要

| 日期 | 阶段 | 结果 |
|---|---|---|
| 2026-07-11 | 项目起点 | 建立最初的 G1 语音/TTS 项目。 |
| 2026-08-02 | 可移植 Runtime 定位 | 转向 Jetson/Linux 机器人平台，移除核心对厂商 SDK 的绑定。 |
| 2026-08-03--11 | 输入、传输和 ASR 基线 | 完成稳定采集、DDS/ROS 2、可插拔 ASR及 Orin Qwen3-ASR 归因优化。 |
| 2026-08-13--17 | 全双工与 TTS 基线 | 接入 AEC、流式 TTS和 ALSA 播放；建立 vLLM-Omni JP6.2 基线并降低内存。 |
| 2026-08-18--19 | Role Agent 与集成 Runtime | 修复多轮 TTS，加入 Role Package，完成模块化单体和进程内主路径。 |
| 2026-08-22--09-07 | Control Plane 与播放恢复 | 建立 Session/Turn/Epoch；完成 persistent 播放恢复、时序审计与声学诊断。 |
| 2026-09-07 | 仓库整理 | 将两周内未提交成果拆成五个逻辑提交，并建立本规划/历史制度。 |

## 3. 详细行动账本

### 2026-07-11：初始语音项目

**目标**

建立机器人语音和 TTS 的最初可运行仓库。

**行动与结果**

- 创建项目和 README，加入最初的 G1 TTS 实现。
- 当时的代码仍以特定机器人和语音功能为中心，尚未形成通用 Runtime 边界。

**提交**

`09f8fea`、`6c2451d`、`ab6129a`

### 2026-08-02：从机器人专用代码转向可移植 Runtime

**问题**

核心能力如果直接依赖机器人厂商 SDK，将难以在不同 Jetson 设备、ROS 2 系统和未来机器人
本体之间复用。

**行动**

- 增加 Jetson GPU 语音后端并重写社区文档。
- 把 Jetson Orin NX 明确为主要性能平台，同时保留通用 Linux 目标。
- 移除核心 Runtime 对厂商实现的强绑定。

**结果与决策**

确立“可移植 Runtime + 可选本体适配器”的长期方向。角色和推理逻辑不再绑定 DDS、ROS 2
或某个机器人品牌。

**提交**

`cb4efc7` 至 `c62ca85`

### 2026-08-03：音频采集、配置所有权和 ROS 2 生命周期

**问题**

远场 VAD、音频 callback 堵塞、配置分散和 ROS 2 生命周期不完整会造成生产运行不稳定。

**行动**

- 调整远场 VAD 默认值并强化有界音频缓冲。
- 集中 Runtime 配置默认值。
- 加入 ROS 2 transport、生命周期和一键服务管理。

**结果与决策**

采集 callback 与推理工作解耦，配置责任更清晰；DDS 之外具备正式 ROS 2 部署方式。

**提交**

`610c587` 至 `84b6755`

### 2026-08-06--11：可插拔 ASR 与 Orin NX 性能归因

**问题**

需要稳定选择麦克风，并判断 SenseVoice、Qwen3-ASR及不同 attention/compile 路径在 Orin NX
上的真实延迟。

**行动**

- 固定并报告实际音频设备选择，禁止生产环境悄悄切换错误麦克风。
- 建立通用 `AsrEngine` 边界，接入 SenseVoice 和 Qwen3-ASR。
- 对 FP16/BF16、eager/SDPA/FA2、static KV cache 和 compiled decode 做受控测试。
- 在 E2E 路径加入 speech latency 观测，并用同模型对象 A/B 与 Nsight 分析吞吐差异。

**关键结果**

- 固定 5.592 s 样本上，保留的 PyTorch 2.5 Qwen3-ASR FP16/SDPA 基线约为 1218.3 ms。
- 隔离的 PyTorch 2.9.1 环境中，static KV cache + generation-aware compiled decode 达到
  451.6 ms mean、460.1 ms p95。
- 实时日志里的约 15.3 tok/s 是包含 encoder/prefill/首 shape warm-up 的 request-level
  apparent throughput；增量 decode 实测约 35.5 tok/s，并非 decoder 性能减半。
- vLLM 0.14 能在 aarch64/CUDA 12.6/SM87 构建和导入，但 16 GB Orin NX 在 Qwen3-ASR
  加载阶段被 OOM killer 终止。因此没有把 vLLM 设为 ASR 生产依赖。
- NF4 在该设备和工作负载上更慢，未采用。

**决策**

保留可复现的 Transformers/compiled 路径和生产稳定路径；性能报告必须区分请求级吞吐、
增量 decode 和 cold/new-shape 成本。

**证据**

- [Qwen3-ASR Runtime Optimization](qwen3_asr_orin_nx_runtime_optimization_report_en.md)
- [E2E runtime gap attribution](qwen3_asr_e2e_runtime_gap.md)
- `benchmarks/orin_nx_2026-08-09/`
- `benchmarks/orin_nx_2026-08-10_fp16_baseline/`
- `benchmarks/orin_nx_2026-08-10_runtime_optimizations/`
- `benchmarks/orin_nx_2026-08-11_e2e_latency/`
- `benchmarks/orin_nx_2026-08-11_runtime_gap/`

**提交**

`562cfa8`、`1973736`、`e97d62e` 至 `6d225b4`、`23c9b28`

### 2026-08-13：低延迟 ALSA、全双工 AEC 与流式 TTS

**问题**

需要从桌面音频原型进入机器人生产路径，并在机器人说话时继续听取用户语音。

**行动**

- 增加低延迟 ALSA 输入和显式 Pulse fallback。
- 定义 hardware DSP、WebRTC APM/AEC3 和半双工 `off` 三种音频处理契约。
- 接入流式 TTS controller、文本调度、PCM player 和实际扬声器 PCM render reference。
- 在隔离环境构建并固定 vLLM-Omni 0.26 + vLLM 0.26.x 的 JP6.2 TTS 组合。

**关键结果**

- 正式建立 ASR -> Agent -> TTS -> ALSA 的流式基础。
- vLLM-Omni 官方 v0.26.0 在 JetPack 6.2/SM87 上成为可运行基线。
- 初始 eager 稳态 First PCM 约 512--569 ms；后续选定 TRITON + Stage-0 FULL 路径后，
  六次稳态测试达到 244.8 ms mean First PCM、0.716 mean audio RTF。
- 冷启动 JIT 曾达到 103.09 s，明确与稳态性能分开报告。

**决策**

TTS 服务保持在独立虚拟环境，不污染稳定 ASR 环境；不把 provider WebSocket 变成 Runtime
内部的强制架构边界。

**证据**

- [qwen3_tts_vllm_omni_jp6.md](qwen3_tts_vllm_omni_jp6.md)
- [qwen3_tts_cuda_graph_orin.md](qwen3_tts_cuda_graph_orin.md)
- [full_duplex_audio.md](full_duplex_audio.md)

**提交**

`668b8d3`、`dcca548`、`a2742af`

### 2026-08-17：TTS 流式性能、内存与全栈容量

**问题**

通用 serving 配置为 Stage 0 预留了远超单机器人短回复所需的 KV cache，挤压 Orin NX
统一内存；同时需要验证 ASR、Agent、TTS 共存能力。

**行动**

- 测量短回复的 Stage-0/Stage-1 token envelope。
- 将 Stage-0 KV cache 固定为 64 MiB/576 tokens，并保留回滚 profile。
- 运行 SenseVoice + Qwen3-4B llama.cpp + Qwen3-TTS 的全栈容量实验。

**结果**

- TTS 增量 idle 内存从 6.68 GiB 降至 4.71 GiB，约节省 1.98 GiB。
- sampled workload peak 约节省 2.17 GiB。
- 五次稳态请求 First PCM 均低于 300 ms，mean RTF 从 0.717 变为 0.712，没有实质回退。
- 单用户部署采用 2048 context、单 slot 的 Agent 候选；统一内存的 PSS、GPU mapping 和
  whole-system 指标明确禁止相加。

**决策**

采用单并发、短回复约束下的生产 TTS profile；量化因改变质量/数值基线而延期。

**证据**

- [qwen3_tts_memory_orin_nx.md](qwen3_tts_memory_orin_nx.md)
- `benchmarks/orin_nx_2026-08-17_tts_memory/`
- `benchmarks/orin_nx_2026-08-17_full_stack_capacity/`

**提交**

`e4c7ae1`、`ec7a75a`

### 2026-08-18--19：多轮 Agent、Role Package 与集成 Runtime

**问题**

语音链路可以单轮工作，但需要修复多轮 TTS handoff，并把身份、知识和能力从传输/机器人
实现中分离。

**行动**

- 修复多轮流式 TTS handoff。
- 加入本地 Role Package Agent Runtime、Tifa 角色与关键词知识检索。
- 对无知识路径和 keyword lore 做小样本 A/B。
- 将正式命名空间收敛到 `star_runtime`，建立模块化单体和集成 in-process 主路径。

**结果**

- ASR、Agent、TTS 可在单 Orin 上通过内存对象流式连接，DDS/ROS 2 保持可选。
- Tifa A/B 中 keyword knowledge 的 mean first text 增加约 224 ms，但答案更短且事实更稳定；
  该结果被标记为探索性产品测试，不是严格性能基准。
- Role Package 成为身份、prompt、lore、voice 和 capability requirements 的版本化边界。

**决策**

保留模块化单体，不进行微服务化重构；同机优先 in-process，跨边界再使用 DDS/ROS 2。

**证据**

- [runtime_architecture.md](runtime_architecture.md)
- [role_packages.md](role_packages.md)
- `benchmarks/tifa_knowledge_ab_2026-08-18/`

**提交**

`b8e59e4`、`44c6f1f`、`47587b2`、`4f5a530`、`e283785`、`4b219de`

### 2026-08-19：视觉感知实验分支

**行动与结果**

分支 `codex/vision-perception-port` 的 `9b0a866` 实现过有界视觉感知管线、camera、
llama.cpp vision adapter 和测试。

**状态与决策**

该提交不在 `main`，因此状态是 `EXPERIMENTAL`，不是当前发布能力。合入前必须单独评审
Runtime 边界、内存占用和与语音交互的调度关系。

### 2026-08-22--09-07：Runtime Control Plane v1

**P0 问题**

VAD 可以停止 TTS 和 PCM，但旧 Agent 仍可能继续生成，旧 delta 有机会重新创建已经失效的
输出。局部 request ID 和 cancellation flag 不能提供端到端正确性。

**行动**

- 建立 Session -> Turn -> Epoch 的统一控制模型和不可变 `ControlStamp`。
- barge-in、turn replacement 和显式取消推进/失效旧 epoch。
- Agent、TTS queue、TTS generation、PCM ring、playback callback 全部执行 exact-stamp 校验。
- 在 provider 支持时主动取消 Agent；epoch 校验仍是最终正确性保护。
- DDS/ROS 2/in-process 保留相同控制元数据。
- 增加确定性竞态测试和 live provider/ALSA harness。

**结果与决策**

软件不变量确立为：epoch N 一旦失效，属于 N 的新 Agent 文本、TTS 请求或 PCM 不能重新进入
外部可见路径。ASR-final fallback 被限定为清理旧输出，不能失效刚创建的新 Turn。

**证据**

- [runtime_architecture.md](runtime_architecture.md)
- `tests/test_runtime_control_plane.py`
- `tests/test_speech_service_control.py`
- `tests/acceptance/runtime_control_plane_live.py`

**提交**

`2e1efab`

### 2026-08-22--09-07：Playback Recovery v1.1 与提交边界指标

**P0 问题**

旧 PCM 不会复活，但 `stream.abort()` 后 replacement turn 偶发无法恢复播放。与此同时，
`post_invalidation_stale_write_submissions` 指标必须真实区分丢弃、已提交残余和禁止提交。

**诊断**

- hard-abort 在约 0.272 ms 内返回并把 stream 标为 inactive。
- 旧 writer 仍阻塞在已运行的 `Pa_WriteStream` 超过 3002 ms。
- replacement PCM 在中断后约 0.58 ms 已准备、0.705 ms 已入队，却没有被 writer dequeue；
  因而不是 TTS 没有生成，也不是 abort 调用本身阻塞。

**行动**

- 选择服务生命周期内持续活跃的 persistent PortAudio/ALSA stream。
- 中断推进 player generation 并清空 ring，在最终 pre-write gate 再验证 generation。
- 用 commit token 标记越过硬件提交边界的块。
- 指标分别记录 stale pre-write drop、committed residual 和 forbidden stale submit attempt。

**硬件结果**

- hard abort A/B 在第 1 个 cycle 内复现三秒无恢复。
- persistent 模式完成 50/50 cycles：零 stall、零 stale hardware submission、零 underflow、
  零 write/restart error、零 inactive transition。
- interrupt -> successful write mean 18.459 ms、p95 18.821 ms、max 19.051 ms。
- live harness 中 replacement PCM ready -> successful hardware write 为 1.574 ms；主要等待发生在
  Agent/TTS provider 之前，而不是 player。

**决策**

persistent 是 JP6.2 生产策略；`hard_abort` 仅作为回滚和诊断 A/B。越过 final pre-write gate
的一个块属于 committed residual，不计为 correctness violation；gate 后仍错误提交的旧块必须
被 forbidden metric 和确定性测试观察到。

**证据**

- [runtime_architecture.md](runtime_architecture.md)
- [playback_metrics.md](playback_metrics.md)
- `tests/acceptance/playback_recovery_stress.py`
- `tests/test_audio_output.py`

**提交**

`2e1efab`

### 2026-08-22--09-07：声学路径诊断与连续 AEC render timeline

**问题**

robot-only 播放会在没有真人近端语音时触发 VAD。不能通过提高阈值隐藏问题，需要同步观察
render reference、raw microphone 和 post-AEC。

**诊断结果**

- 物理 echo delay 稳定在约 97--102 ms。
- capture 以 20 ms cadence 持续处理，但 TTS provider 的首批 PCM 后，WebRTC reverse calls
  曾缺失约 1.23--1.27 s。
- 60--140 ms fixed delay sweep 全部仍产生 robot-only VAD edge。

**行动**

persistent speaker writer 成为唯一 render-clock authority：每 10 ms 向 speaker path 提交真实
PCM或显式静音，并把完全相同的信号送入 WebRTC reverse processing。时钟静音不属于 stale
audible PCM。

**结果**

- render max interval 降到 11.601 ms，超过 15 ms 的 gap 为零。
- 约 1.25 s 的 reverse-stream 空洞消失，物理 echo delay 仍约 101 ms。
- 但 robot-only 仍在 utterance tail 产生 1 次 VAD edge/Control invalidation。

**决策**

连续 render clock 作为正确时基保留，但“缺口是唯一误触根因”的假设被否定。按照实验边界，
没有继续调整 VAD threshold 或 fixed delay，也没有宣称硬件发布通过。真人 double-talk 最终验收
仍未签字。

**证据**

- [acoustic_path_diagnostic.md](acoustic_path_diagnostic.md)
- `tests/acceptance/acoustic_path_diagnostic.py`
- `tests/test_acoustic_trace.py`

**提交**

`2e1efab`

### 2026-08-22--09-07：Runtime Timing Audit 与 Agent TTFT 基准

**问题**

终端完整 response 的出现时间、Agent 首个 delta、首个可合成语句、TTS First PCM 和真实硬件
发声被混在一起，无法解释“日志很快但体感很慢”。

**行动与结果**

- 每个终态 Turn 输出统一单调时钟上的 `RUNTIME_TURN_TIMELINE`。
- 时序覆盖 speech end、ASR、Agent first delta、first delta publish、TTS segment、First PCM、
  player enqueue/dequeue、final pre-write gate 和 hardware write。
- 新增独立 Responses API 基准，隔离 ASR/TTS/AEC，测量 input tokens、cached tokens、output
  tokens、TTFT、response complete 和 decode tok/s。

**决策**

用户等待体验以 speech end -> first hardware write 为完整边界；Agent 优化首先看 TTFT，但不能
脱离输入规模和缓存命中率比较。终端完整 response 只代表 generation complete，不代表流式
数据何时已经进入 TTS。

**证据**

- [runtime_timing_audit.md](runtime_timing_audit.md)
- [agent_llm_latency_benchmark.md](agent_llm_latency_benchmark.md)
- `scripts/benchmark-qwen3-agent-responses.py`

**提交**

`2e1efab`、`3fe220b`

### 2026-09-07：角色内容扩充与遗留工作区整理

**行动**

- 新增 Ada Wong、Babata、Link、Saber Role Package。
- 扩充 Olaf lore，并明确为 English/Aiden 角色配置。
- 增加 Tifa 服装、体型资料、来源和知识检索测试。
- 审计两周未提交工作区，运行完整测试并拆成逻辑提交。

**整理发现与决策**

- Olaf 测试仍期待中文，与角色包明确的英文输出不一致；测试更新为 English。
- Tifa `max_tokens=6400` 没有文档或语音产品依据，被识别为临时实验值并恢复为 64。
- TTS `codec_chunk_frames` 从 25 调至 5，作为独立提交保存，便于 A/B 或回滚。
- 整理后完整测试通过，工作区恢复干净。

**提交**

- `2e1efab` — Runtime Control Plane、playback、AEC与 timing
- `1cd3c40` — TTS codec streaming chunk size
- `3fe220b` — Agent Responses latency benchmark
- `6081aca` — Role packages and lore
- `69a946c` — ignore Cursor metadata

## 4. 当前未关闭事项

未关闭事项的权威优先级见 [ROADMAP.md](ROADMAP.md)，这里仅保留历史出口：

1. JP6.2 robot-only 零 VAD edge 验收仍失败/未签字。
2. 真人在机器人说话期间开口的 double-talk 最终验收仍未签字。
3. Runtime Measurement v1 尚未开始；现有 timeline 和 Agent benchmark 只是测量基础。
4. GPU scheduling、model residency、unified memory admission 和 Resource Manager 明确延期。
5. `codex/vision-perception-port` 尚未合入 `main`。
