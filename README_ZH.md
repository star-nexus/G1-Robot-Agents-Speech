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

Jetson ORIN NX，同一段 5.592 秒中文音频，预热 3 次、运行 30 次：

| 后端 | 模型 | 平均耗时 | 中位数 | P95 | RTF |
|---|---|---:|---:|---:|---:|
| CPU | INT8 | 206.9 ms | 206.8 ms | 208.6 ms | 0.0370 |
| GPU | FP32 | 69.5 ms | 64.5 ms | 94.3 ms | 0.0124 |

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

机器人有多个音频输入时，应当固定语音服务使用的麦克风。USB 摄像头经常也会枚举出
麦克风，PulseAudio 可能在摄像头接入时自动切换默认输入。先从设备列表中找到具有
辨识度的名称，再写入本机的 `deploy.env`：

```bash
.venv/bin/python -c 'import sounddevice as sd; print(sd.query_devices())'

# deploy.env——仅为示例，请填写自己设备列表中出现的名称
MICROPHONE_DEVICE="USB Microphone"

sudo g1-speech-service restart
g1-speech-service status
```

推荐使用稳定的名称，而不是重启或 USB 重新枚举后可能变化的数字序号。`status` 和
`logs` 都会显示当前配置的麦克风；使用 `pulse` 或 PortAudio 默认输入时，还会显示
当前 PulseAudio 输入源，并提示热插拔可能改变它。

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

流水线只依赖稳定的 `AsrEngine` 契约，具体模型由 `asr.backend` 选择：

| `asr.backend` | 推理实现 | 模型配置段 |
|---|---|---|
| `sensevoice` | sherpa-onnx | `sensevoice` |
| `qwen3_asr` | Transformers 5.13 原生 Qwen3-ASR | `qwen3_asr` |

从版本库中的 `config.qwen3-asr.example.json` 复制一份本机配置并修改 `model_dir`。
配置文件以 `.local.json` 结尾时会被 Git 忽略，不会把本机绝对路径提交到仓库。
然后给 Jetson GPU 环境补齐运行时；脚本保留 NVIDIA PyTorch，并把 NumPy 固定在与其
ABI 兼容的 1.x：

```bash
bash scripts/setup-qwen3-asr.sh

cp config.qwen3-asr.example.json config.qwen3-asr.local.json
# 编辑 config.qwen3-asr.local.json 中的 model_dir

.venv-gpu/bin/g1-speech doctor \
  --config config.qwen3-asr.local.json --load-model --skip-audio

.venv-gpu/bin/g1-speech transcribe \
  --config config.qwen3-asr.local.json \
  models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17/test_wavs/zh.wav
```

前台试运行 DDS 或 ROS 2：

```bash
.venv-gpu/bin/g1-speech serve --config config.qwen3-asr.local.json
.venv-gpu/bin/g1-speech serve \
  --config config.qwen3-asr.local.json --transport ros2
```

长期运行时，只需在 `deploy.env` 选择 GPU 服务使用的配置并重启；topic、消息结构和
订阅端完全不变：

```bash
SPEECH_CONFIG_GPU="config.qwen3-asr.local.json"
sudo g1-speech-service gpu
```

`qwen3_asr.language` 设为 `null` 可自动识别语言，设为 `zh` 可强制中文。
`qwen3_asr.prompt` 可提供机器人和场景专有词，例如
`"词汇：实验室名称、机器人型号、开放世界自主交互。"`。
JetPack 自带 PyTorch 的 SDPA 尚不支持 Qwen3 使用的 GQA 参数，因此本机配置使用
`attention_implementation: "eager"`；以后升级到支持 `enable_gqa` 的 PyTorch 后可再测试 `sdpa`。

公平比较速度和准确率时，应使用同一批真实机器人录音及人工标注。JSONL 每行包含
`audio`、`text` 和可选 `language`；仓库示例只用于冒烟测试，不代表真实准确率：

```bash
PYTHONPATH=src .venv-gpu/bin/python acceptance/compare_asr.py \
  --config sensevoice=config.gpu.json \
  --config qwen-0.6b=config.qwen3-asr-0.6b.local.json \
  --config qwen-1.7b=config.qwen3-asr-1.7b.local.json \
  --manifest acceptance/asr_manifest.example.jsonl \
  --warmup 2 --runs 5
```

输出包含加载耗时、均值/中位数/P95、RTF、语料级 CER 和逐条转写。程序串行加载并卸载
模型，避免 Orin NX 同时驻留多套权重。生产选型建议至少录制 100 条，覆盖远近场、噪声、
口音、打断和专有词；先以 CER 为质量门槛，再在合格模型中选 P95 延迟最低者。

以后接其他模型时，可在独立 Python 包中注册 `g1_speech.asr_backends` entry point，
也可把 `asr.backend` 写成 `your_module:create_engine`。工厂接收 `ServiceConfig`，返回实现
`load / transcribe / close` 的引擎即可；采集、VAD、生命周期和传输代码无需修改。

### 本地 Qwen 语音对话

把语音识别结果直接交给本地 Qwen3，回答以流式文本显示在终端，保留有限对话历史，
暂不使用 TTS。默认使用 llama.cpp 的 OpenAI-compatible 服务：

```bash
# 如果 `llama serve` 已包含所需加速后端，可直接使用；Jetson 建议启动
# 通过 GGML_CUDA=ON 编译的独立 llama-server。
llama-server -m /path/to/Qwen3-8B-Q5_K_M.gguf --alias qwen3-8b-q5 \
  --host 127.0.0.1 --port 8080 -c 4096 -ngl all -np 1 \
  --cache-ram 0 --no-cache-prompt --no-cache-idle-slots \
  --flash-attn on --reasoning off --no-webui

# 自动跟随当前启用的 DDS 或 ROS 2 语音服务
scripts/run-qwen-voice-chat --model qwen3-8b-q5
```

如果 llama.cpp 中运行的是视觉语言模型，可指定 V4L2 摄像头。默认 `see` 策略会在
主回答之前运行一个独立分类 session：它使用专用 system prompt，保留四轮路由历史，
但不会读取或修改角色对话。分类器只输出 `[SEE]` 或 `[TEXT]`；只有 `[SEE]` 才附加
当前画面，之后再由主对话 session 生成给用户的回答：

```bash
# 适用于 16 GB Jetson 的 Qwen3-VL 内存安全配置。仅当 llama-server
# 不在 PATH 中时才需要设置 LLAMA_SERVER_BIN。
LLAMA_SERVER_BIN=/path/to/llama-server \
scripts/run-qwen-vl-server \
  /path/to/Qwen3VL-4B-Instruct-Q4_K_M.gguf \
  /path/to/mmproj-Qwen3VL-4B-Instruct-Q8_0.gguf

scripts/run-qwen-voice-chat \
  --model qwen3-vl-4b-q4 \
  --role robot-assistant \
  --camera /dev/video0 \
  --camera-width 1280 \
  --camera-height 720 \
  --camera-fps 5 \
  --vision-strategy see
```

对 Qwen3-VL，客户端会自动使用模型卡推荐的采样配置：纯文本请求使用
`temperature=1.0`、`top_p=1.0`、`top_k=40`、`presence_penalty=2.0`；
带图请求使用 `temperature=0.7`、`top_p=0.8`、`top_k=20`、
`presence_penalty=1.5`。两者的 `repeat_penalty` 均为 `1.0`。
`--temperature` 仅用于明确覆盖默认温度。
服务启动脚本还会限制为单 slot、使用 1536 token 上下文和 Q8 KV cache，并关闭
llama.cpp 默认上限 8 GiB 的全局 prompt cache。这组有界默认值用于保证 16 GB
统一内存 Jetson 的长期运行稳定性；可通过 `scripts/run-qwen-vl-server --help`
查看环境变量覆盖项。

角色配置可以在不重新编译、也不重启模型服务的情况下切换应用场景。带版本号的 JSON
`.role` 文件集中保存 system prompt、有界对话设置以及独立的文本/视觉采样参数：

```bash
scripts/run-qwen-voice-chat --list-roles

scripts/run-qwen-voice-chat \
  --model qwen3-vl-4b-q4 \
  --role snowball \
  --camera /dev/video0 \
  --vision-strategy see
```

也可以通过 `--role /path/to/custom.role` 加载外部角色。显式命令行参数优先于角色
配置，角色配置优先于内置默认值。角色文件只能设置经过校验的 prompt、对话和采样
字段，不能注入 `model`、`messages`、`stream` 等协议核心字段。

视觉策略已经与 ASR、摄像头、消息队列、对话历史和通信传输解耦，可以直接切换：

| 策略 | 行为 |
|---|---|
| `see` | 主回答前运行独立的四轮 `[SEE]`/`[TEXT]` 分类 session |
| `qwen` | 保留此前独立调用 Qwen 输出 `T`/`V`/`U` 后再回答的方案 |
| `always` | 每句话都附图 |
| `off` | 不启动也不使用摄像头 |

通过 `--vision-strategy see|qwen|always|off` 选择。摄像头只在内存中保留最新 JPEG；
纯文本路径不会做 Base64 或视觉编码。可用 `--vision-router-history-turns` 调整独立
分类 session 的历史长度，默认四轮。该功能需要 FFmpeg，可用 `v4l2-ctl --list-devices`
查找设备。

交互记录默认关闭。需要做受控策略对比时，可指定
`--vision-log /path/to/results.jsonl`。权限为 `0600` 的 JSONL 会记录策略、路由与耗时、
是否附图、端到端回答指标和纠错信号，但不会保存图片。

在 Jetson 上应确认服务启动日志显示 `CUDA0`；Vulkan 版本也能运行，但在已测
Orin NX 上速度较慢。Ollama 仍作为可选后端保留：

```bash
# 首次导入本地 GGUF 模型
bash scripts/import-ollama-gguf.sh /path/to/Qwen3-8B-Q5_K_M.gguf qwen3-8b-q5

scripts/run-qwen-voice-chat --provider ollama --model qwen3-8b-q5
```

语音回调只负责把事件放入队列，不会被 LLM 推理阻塞。说“清空对话”可以清除
当前对话历史。

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

ROS 2 使用 `g1_speech_msgs/msg/SpeechEvent` 和
`g1_speech_msgs/msg/PlaybackState`，完整保留 DDS 事件契约。安装与生命周期说明见
[ROS 2 文档](ros2/README.md)。

## 工作方式

```text
Microphone
  → 16 kHz mono audio
  → Silero VAD
  → selected ASR adapter (SenseVoice / Qwen3-ASR / external plugin)
  → DDS or ROS 2 transport
  → Robot / Agent / application

TTS or playback
  → rt/g1/hri/playback/state
  → temporarily pause recognition
```

| Topic | 方向 | 用途 |
|---|---|---|
| `rt/g1/hri/speech/final` | Service → Agent | 最终语音识别事件 |
| `rt/g1/hri/playback/state` | Agent → Service | 播放期间暂停识别，避免机器人听到自己 |
| `/hri/speech/final` | Service → ROS 2 Agent | 最终结构化 `SpeechEvent` |
| `/hri/playback/state` | ROS 2 Agent → Service | 结构化播放门控状态 |

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

复制 `deploy.env.example` 后只填写自己的设备信息，不需要固定 IP：

| 配置 | 说明 |
|---|---|
| `DDS_NETWORK_INTERFACE` | DDS 使用的本机网卡；留空时自动选择 |
| `MICROPHONE_DEVICE` | PortAudio 输入编号或设备名；留空可交互选择 |
| `DDS_DOMAIN_ID` | 与订阅方一致的 DDS Domain |
| `SPEECH_TOPIC` | 最终识别结果 Topic |
| `PLAYBACK_TOPIC` | TTS/播放门控 Topic |
| `SENSEVOICE_THREADS` | CPU 推理线程数 |
| `SPEECH_TRANSPORT` | `dds`（默认）或 `ros2` |
| `ROS2_SETUP` | 可选 ROS 2 `setup.bash`；通常可自动发现 |
| `ROS2_WORKSPACE_SETUP` | 可选自定义工作空间 overlay 路径 |
| `ROS2_DOMAIN_ID` | 可选 ROS Domain；安装时默认沿用当前终端的值 |

运行时参数位于 `config.json`；GPU 部署使用独立的 `config.gpu.json`。相对模型路径
按配置文件所在目录解析。

也可以直接根据应用的统一默认值生成配置：

```bash
g1-speech config init --output config.json
```

安装脚本使用同一个命令，并且只应用 `deploy.env` 中明确设置的覆盖项，因此 CPU、
GPU、示例配置和代码调用不会再分别维护默认参数。

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
