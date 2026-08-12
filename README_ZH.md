# G1 Speech Service — 机器人 Agent 离线语音识别

[English](README.md) | 简体中文

面向机器人 Agent 的可插拔离线语音识别服务，针对 NVIDIA Jetson 优化。
为机器人提供快速、私密的语音输入，支持与人实时互动，无需依赖云端。

项目不依赖任何机器人厂商 SDK，通过标准音频设备、DDS 或 ROS 2 接入机器人系统。
当前主要性能验证平台为 NVIDIA Jetson Orin NX，同样适用于其他 Jetson 机器人计算平台、
NVIDIA DGX Spark 和通用 Linux 边缘计算设备。
仓库中的 `G1` 仅作为现有项目和协议命名空间，不表示绑定任何同名机器人品牌。

- **CPU/GPU 双后端**：支持 CPU INT8 与 Jetson CUDA FP32 两种部署模式
- **ASR 模型可插拔**：内置 SenseVoice 与 Qwen3-ASR，切换模型不影响 VAD、DDS/ROS 2 或 Agent
- **生产级全双工**：硬件 DSP / WebRTC APM(AEC3) 可插拔，支持 VAD 打断机器人发声
- **流式机器人语音客户端**：实验性 Qwen3-TTS/vLLM-Omni PCM 分块通过低延迟 ALSA 直连播放
- **完全离线**：语音识别在本地完成，音频和文本无需上传云端
- **可靠音频采集**：内置流心跳、麦克风自动重连和有界队列
- **DDS 与 ROS 2 双传输**：轻量 DDS 系统和 ROS 2 机器人都可直接订阅结构化事件
- **高性能**：识别速度 0.01–0.2 秒，准确度高

## 已验证平台

| 平台 | 计算设备 | 可用后端 | 验证状态 |
|---|---|---|---|
| Jetson Orin NX 边缘计算设备 | Jetson Linux | CPU INT8 / CUDA FP32 | ✅ 性能测试平台 |
| NVIDIA DGX Spark | ARM64 Linux | CPU INT8 | ✅ 最初部署平台 |
| 其他 Jetson/Linux 机器人 | Jetson 或 Linux 边缘计算设备 | CPU INT8；Jetson CUDA FP32 | 兼容目标 |

验证状态仅代表本项目实际使用的配置；机器人厂商提供的硬件配置可能有所不同。

## 性能

Jetson Orin NX，`MAXN_SUPER` 电源模式（动态频率、未锁频），同一段 5.592 秒中文 WAV，
预热 3 次、正式运行 10 次：

| 模型 | 设备 | 精度 | Attention | 平均耗时 | 中位数 | P95 | RTF |
|---|---|---|---|---:|---:|---:|---:|
| SenseVoice-Small | CPU | INT8 | 不适用 | 205.9 ms | 206.0 ms | 206.4 ms | **0.0368** |
| SenseVoice-Small | GPU | FP32 | 不适用 | 77.1 ms | 78.0 ms | 97.7 ms | **0.0138** |
| Qwen3-ASR-0.6B | CPU | FP32 | eager | 8489.4 ms | 8469.6 ms | 9537.6 ms | 1.5181 |
| Qwen3-ASR-0.6B | CPU | FP32 | SDPA | 17045.2 ms | 16967.9 ms | 19567.6 ms | 3.0481 |
| Qwen3-ASR-0.6B | CPU | — | FA2 | 不支持（仅 CUDA） | — | — | — |
| Qwen3-ASR-0.6B | GPU | BF16 | eager | 1378.2 ms | 1375.9 ms | 1391.5 ms | 0.2465 |
| Qwen3-ASR-0.6B | GPU | BF16 | SDPA | **1235.7 ms** | 1234.8 ms | 1241.7 ms | **0.2210** |
| Qwen3-ASR-0.6B | GPU | FP16 | SDPA | **1218.3 ms** | 1216.0 ms | 1244.7 ms | **0.2179** |
| Qwen3-ASR-0.6B | GPU | BF16 | FA2 | 1493.4 ms | 1494.3 ms | 1500.2 ms | 0.2671 |

SenseVoice 使用 ONNX 图，不存在 eager/SDPA/FA2 选择。在这台 Orin 的单请求场景中，
GPU SDPA 比 eager 快 10.3%；FA2 反而比 SDPA 慢 20.9%。CPU SDPA 更慢，是因为当前
Jetson PyTorch 缺少原生 GQA，兼容路径需要显式展开 KV heads。这里测量的是端到端
Transformers adapter 延迟，不是服务器 GPU 上的 vLLM 吞吐，也不是准确率基准。

后续在隔离的 PyTorch 2.9.1 / Triton 3.5.1 环境中，static KV cache 与
generation-aware compiled decode 将固定样本延迟降至 451.6 ms。严格归因实验进一步
确认：steady-state incremental decode throughput 约为 35.5 tok/s；此前实时路径记录的
15.3 tok/s 是被固定 encoder/prefill 成本和首次新-shape graph warm-up 拉低的
request-level apparent generation throughput，并非 decoder 性能回退。详见
[runtime 优化报告](docs/qwen3_asr_orin_nx_runtime_optimization_report_en.md)和
[E2E 差异归因](docs/qwen3_asr_e2e_runtime_gap.md)。

## 快速接入

### 运行语音服务

部署完成后，直接选择推理后端和传输方式。四种模式都是由 systemd 管理的后台服务，
并可随系统自动启动：

```bash
# 低资源、默认选择
sudo g1-speech-service cpu

# Jetson CUDA 加速
sudo g1-speech-service gpu

# ROS 2 传输（CPU/GPU 均可）
sudo g1-speech-service cpu ros2
sudo g1-speech-service gpu ros2

g1-speech-service status
g1-speech-service logs
```

机器人生产部署使用 ALSA 硬件直连；所选麦克风不可用时直接报错，不静默回退到
其他桌面麦克风。配置稳定的 ALSA card ID，不要写入 USB 重枚举后会变化的
`hw:1,0` 数字：

```bash
for path in /proc/asound/card*/id; do
  printf '%s: %s\n' "$path" "$(<"$path")"
done

# deploy.env——请使用本机输出的 ID
AUDIO_INPUT_BACKEND="alsa"
ALSA_INPUT_CARD="your-card-id"
AUDIO_INPUT_FALLBACK=""
AUDIO_INPUT_BLOCK_MS=20

sudo g1-speech-service restart
g1-speech-service status
```

服务用麦克风原生格式采集，并在实时 callback 之外转换成 ASR 所需的 16 kHz 单声道。
项目不根据厂商名猜测麦克风质量：安装时由用户选择 card ID，只有硬件候选唯一时才可留空自动选择。
`logs` 会报告实际打开的 backend、解析后的设备、原生格式和 PortAudio latency。独占设备要求与完整验证方法见
[低延迟输入指南](docs/audio_input.md)。

全双工交互需要明确选择一种音频处理契约，并固定扬声器的稳定 ALSA card ID：

```bash
# 普通 USB 麦克风：软件 AEC3/NS，使用实际扬声器 PCM 作为参考。
AUDIO_PROCESSING_MODE="webrtc"
ALSA_OUTPUT_CARD="your-speaker-card-id"
TTS_ENABLED=1

# 已验证带 DSP/AEC 的麦克风阵列（不得重复叠加软件 AEC）：
# AUDIO_PROCESSING_MODE="hardware"
```

`off` 保留安全的半双工播放门控；`hardware` 与 `webrtc` 在机器人说话时继续运行 VAD，
用户开口后会终止合成、清空排队文本并立即 abort ALSA 播放。完整说明见
[全双工音频指南](docs/full_duplex_audio.md)和
[JetPack 6 vLLM-Omni 隔离环境记录](docs/qwen3_tts_vllm_omni_jp6.md)。

DDS 是默认传输。识别结果发布到 `rt/g1/hri/speech/final`。CPU/GPU 后端使用
相同的消息，Agent 无需修改。

### Agent 订阅

```python
from g1_speech.dds import DdsSpeechSubscriber, initialize_dds

# 每个 Agent 进程只初始化一次 Cyclone DDS。
initialize_dds(domain_id=0)

subscriber = DdsSpeechSubscriber(lambda event: print(event.text))
subscriber.start()
```

Agent 退出时调用 `subscriber.close()`。回调运行在独立订阅线程中，正式 Agent 建议只在
回调里入队，再由业务线程调用 LLM、工具或机器人动作。

无需编写 Agent 代码也可以快速查看识别结果：

```bash
.venv/bin/g1-speech listen --config config.json --timeout 0
```

### 切换 SenseVoice / Qwen3-ASR

流水线只依赖稳定的 `AsrEngine` 契约，模型由 `asr.backend` 选择。Qwen3-ASR 使用本机
Hugging Face 模型目录；以 `.local.json` 结尾的配置和模型文件均不会提交到 Git：

```bash
bash scripts/setup-qwen3-asr.sh
cp config.qwen3-asr.example.json config.qwen3-asr.local.json
# 修改 model_dir 后验证 CUDA 模型加载
.venv-gpu/bin/g1-speech doctor \
  --config config.qwen3-asr.local.json --load-model --skip-audio
```

长期运行时在 `deploy.env` 选择配置，然后继续使用同一个 systemd 服务入口：

```bash
SPEECH_CONFIG_GPU="config.qwen3-asr.local.json"
sudo g1-speech-service restart
# 服务当前停止时：sudo g1-speech-service gpu dds
```

`SPEECH_CONFIG_GPU` 是部署时唯一的模型选择入口。JSON profile 负责
`asr.backend`、模型目录、dtype、attention 和 cache；`deploy.env` 不再重复这些字段。
切回 SenseVoice GPU 时只需选择 `config.gpu.json` 并重启。详见
[配置模型](docs/configuration.md)。

示例配置在 Orin NX 上采用已封存的 `SDPA + FP16` baseline：10 条、50.188 秒带标注
语料中，FP16 与 BF16 的忽略标点内容 CER 均为 6.22%，10/10 内容转写一致，且每条
重复三次均确定。其他 CUDA 平台仍应先测量，再决定是否从 BF16 切换。

默认 `attention_implementation` 为 `sdpa`。新 PyTorch 使用原生 GQA；Jetson 当前的
NVIDIA PyTorch 2.5 缺少 `enable_gqa` 参数，adapter 会自动展开 KV heads 后进入 SDPA，
无需修改 Transformers。需要排障或复现旧基线时可显式设为 `eager`。

FlashAttention 2 仍可用于受控实验，但不再作为部署 profile：在已测 Orin batch=1
场景中，它比 SDPA 慢 20.9%。实验变体及其完整参数应跟随 benchmark artifact 保存，
而不是继续在根目录增加 `config.qwen3-*` 文件。

每次识别日志都会报告音频时长和 RTF。固定 WAV 的可复现基准命令：

```bash
PYTHONPATH=src .venv-gpu/bin/python acceptance/benchmark_engine.py \
  --config config.qwen3-asr.local.json --wav /path/to/fixed.wav \
  --warmup 3 --runs 10
```

公平比较准确率时，使用 `acceptance/compare_asr.py` 对同一份人工标注 JSONL 串行测试
SenseVoice、Qwen3-ASR 0.6B、1.7B 或外部 `g1_speech.asr_backends` 插件。

### ROS 2

ROS 2 是可选能力，不影响默认 DDS 部署。首次准备一次，之后就可以像 DDS 一样选择，
无需用户手工 source 环境：

```bash
bash scripts/setup-ros2.sh
sudo g1-speech-service gpu ros2  # 也可以使用 cpu ros2
g1-speech-service status
```

识别结果发布到 `/hri/speech/final`。在已 source ROS 的终端中可以直接查看：

```bash
ros2 topic echo /hri/speech/final g1_speech_msgs/msg/SpeechEvent
```

原生生命周期节点既支持一条命令自动进入工作状态，也支持交给外部 Lifecycle Manager：

```bash
ros2 launch g1_speech_ros2 speech_lifecycle.launch.py \
  config_file:="$PWD/config.json" autostart:=true

# 或由外部 Lifecycle Manager 控制：
ros2 launch g1_speech_ros2 speech_lifecycle.launch.py \
  config_file:="$PWD/config.json" autostart:=false
ros2 lifecycle set /g1_speech configure
ros2 lifecycle set /g1_speech activate
```

ROS 2 使用 `g1_speech_msgs/msg/SpeechEvent`、
`g1_speech_msgs/msg/PlaybackState`，并在启用 TTS 时订阅
`g1_speech_msgs/msg/TtsTextChunk`，完整保留 DDS 事件契约。安装与生命周期说明见
[ROS 2 文档](ros2/README.md)。

## 工作方式

```text
Microphone
  → 16 kHz mono audio
  → Silero VAD
  → SenseVoice-Small
  → DDS or ROS 2 transport
  → Robot / Agent / application

TTS or playback
  → off：通过 playback state 暂停识别
  → hardware/webrtc：输入 AEC 参考，并由 VAD 打断播放
```

| Topic | 方向 | 用途 |
|---|---|---|
| `rt/g1/hri/speech/final` | Service → Agent | 最终语音识别事件 |
| `rt/g1/hri/playback/state` | Agent → Service | 播放期间暂停识别，避免机器人听到自己 |
| `rt/g1/hri/tts/request` | Agent → Service | 流式 TTS 的增量、可去重文本块 |
| `/hri/speech/final` | Service → ROS 2 Agent | 最终结构化 `SpeechEvent` |
| `/hri/playback/state` | ROS 2 Agent → Service | 结构化播放门控状态 |
| `/hri/tts/request` | ROS 2 Agent → Service | 增量 `TtsTextChunk` 请求 |

`SpeechEvent` 包含稳定的 `event_id`、文本、语言、音频时长、推理耗时、来源和时间戳。
发布端会在短暂 DDS 故障时重试，订阅端会按 `event_id` 去重。

## 部署

### CPU INT8

CPU 是默认后端，适合资源受限设备和不具备 CUDA 的 Linux 主机。

```bash
git clone https://github.com/star-nexus/G1-Robot-Agents-Speech.git
cd G1-Robot-Agents-Speech

cp deploy.env.example deploy.env
# DDS 网卡可选；麦克风留空时可在安装过程中选择
# DDS_NETWORK_INTERFACE="your-interface"

bash scripts/setup-cpu.sh
```

脚本会创建项目私有的 `.venv`、下载模型、运行 doctor 和固定 WAV 测试，并安装 CPU
systemd 服务。

### Jetson GPU FP32

GPU 环境与 CPU 环境完全隔离，不会覆盖 `.venv`、`config.json` 或 CPU INT8 模型。

```bash
bash scripts/setup-jetson-gpu.sh
sudo g1-speech-service gpu
```

GPU 部署会：

1. 创建独立的 `.venv-gpu`
2. 为 Jetson aarch64 编译 sherpa-onnx CUDA wheel
3. 安装 SenseVoice FP32 模型
4. 生成 `config.gpu.json`
5. 安装互斥的 `g1-speech-gpu.service`

当前 GPU 构建针对 CUDA 12.6、sherpa-onnx 1.13.4 和 ONNX Runtime 1.18.1。
如果 CUDA wheel 不可用，服务会明确失败，不会静默回退 CPU。

## 配置

配置明确分成两层：

1. JSON 模型 profile 选择 SenseVoice/Qwen3-ASR，并拥有全部模型专属参数。
2. `deploy.env` 描述本机部署，拥有 profile 路径、Python runtime、麦克风、通用
   VAD/播放门控、transport 与安装选项。

`deploy.env.example` 与真实 `deploy.env` 字段和顺序一致，仅使用与主机无关的示例值：

| 配置 | 说明 |
|---|---|
| `SPEECH_CONFIG_CPU` / `SPEECH_CONFIG_GPU` | CPU/GPU 服务选择的 JSON profile |
| `SPEECH_PYTHON_GPU` | GPU 服务可选的隔离 Python runtime |
| `SPEECH_GPU_LIBRARY_PATH` | 该 runtime 需要的冒号分隔 native library 路径 |
| `DDS_NETWORK_INTERFACE` | DDS 使用的本机网卡；留空时自动选择 |
| `AUDIO_INPUT_BACKEND` / `AUDIO_INPUT_FALLBACK` | 生产默认：`alsa` 直连，不静默回退 |
| `ALSA_INPUT_*` | 稳定 ALSA card ID 与硬件原生设备/采样率/声道/dtype |
| `PULSE_INPUT_DEVICE` | 主动选择或回退时使用的 PulseAudio 设备 |
| `PULSE_SOURCE` | 回退客户端可选的稳定 PulseAudio source |
| `AUDIO_INPUT_BLOCK_MS` / `AUDIO_INPUT_LATENCY` | callback 大小与 PortAudio latency 请求 |
| `AUDIO_PROCESSING_MODE` / `WEBRTC_*` | `off`、硬件 DSP 或原生 WebRTC APM/AEC3 |
| `ALSA_OUTPUT_*` / `TTS_*` | 物理扬声器与本机流式 TTS endpoint |
| `DDS_DOMAIN_ID` | 与订阅方一致的 DDS Domain |
| `SPEECH_TOPIC` | 最终识别结果 Topic |
| `PLAYBACK_TOPIC` | TTS/播放门控 Topic |
| `VAD_*` | 对全部 ASR backend 生效的通用分句参数 |
| `PLAYBACK_*` | 对全部 ASR backend 生效的通用识别门控 |
| `SPEECH_TRANSPORT` | `dds`（默认）或 `ros2` |
| `ROS2_SETUP` | 可选 ROS 2 `setup.bash`；通常可自动发现 |
| `ROS2_WORKSPACE_SETUP` | 可选自定义工作空间 overlay 路径 |
| `ROS2_DOMAIN_ID` | 可选 ROS Domain；安装时默认沿用当前终端的值 |

不要在 `deploy.env` 中配置 `ASR_BACKEND`、`SENSEVOICE_*` 或 `QWEN3_ASR_*`；它们属于
被选中的 JSON profile。backend-neutral 环境设置会在生成配置和服务启动时应用，因此
会一致地作用于 SenseVoice 与 Qwen3-ASR。相对模型路径按 JSON profile 所在目录解析。

也可以直接根据应用的统一默认值生成配置：

```bash
g1-speech config init --output config.json
```

安装脚本使用同一个命令，并且只应用 `deploy.env` 中明确设置的 backend-neutral
覆盖项。完整字段归属与优先级见 [`docs/configuration.md`](docs/configuration.md)。

## 验证与测试

检查 CPU 部署：

```bash
.venv/bin/g1-speech doctor --config config.json --load-model
```

检查 GPU 部署：

```bash
.venv-gpu/bin/g1-speech doctor \
  --config config.gpu.json --load-model --skip-audio
```

识别固定 WAV：

```bash
.venv/bin/g1-speech transcribe --config config.json path/to/audio.wav
```

复现性能测试：

```bash
.venv/bin/python acceptance/benchmark_engine.py \
  --config config.json \
  --wav models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17/test_wavs/zh.wav

.venv-gpu/bin/python acceptance/benchmark_engine.py \
  --config config.gpu.json \
  --wav models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/test_wavs/zh.wav
```

运行单元测试：

```bash
.venv/bin/python -m pytest
```

## 服务管理

```bash
g1-speech-service status
g1-speech-service logs
g1-speech-service logs cpu
g1-speech-service logs gpu
g1-speech-service logs cpu ros2
g1-speech-service logs gpu ros2

sudo g1-speech-service restart
sudo g1-speech-service stop
```

底层 systemd 单元：

- CPU + DDS：`g1-speech.service`
- GPU + DDS：`g1-speech-gpu.service`
- CPU + ROS 2：`g1-speech-ros2.service`
- GPU + ROS 2：`g1-speech-gpu-ros2.service`

四个单元彼此互斥，因此只有用户选择的模式会占用麦克风。切换时会停止原模式，并把
新模式设为开机启动。

## 引用

本项目基于以下开源工作构建，感谢原作者和社区贡献者：

- [SenseVoice](https://github.com/QwenAudio/SenseVoice)：多语言语音识别模型
- [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx)：SenseVoice ONNX 推理与跨平台部署
- [Silero VAD](https://github.com/snakers4/silero-vad)：语音活动检测
- [Eclipse Cyclone DDS](https://github.com/eclipse-cyclonedds/cyclonedds)：厂商无关的 DDS 通信
