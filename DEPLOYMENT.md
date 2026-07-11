# G1 Speech Service 双机部署与 Agent 接入（旧方案参考）

> 当前项目已经改为 DGX 单机采集、识别、发布和订阅。新部署请使用
> [DGX_DEPLOYMENT.md](./DGX_DEPLOYMENT.md)。本文及 `setup-orin.sh`、
> `setup-spark.sh`、`verify-dual.sh` 只保留用于参考原有参数与双机实现。

本文面向以下比赛环境：

```text
Unitree G1 板载 Orin NX
  ├── RealSense D435i：图像/视觉输入
  ├── 麦克风：语音输入
  └── speech_service：VAD + SenseVoice + DDS

NVIDIA Spark 上位机
  ├── 团队已有 Agent
  ├── OpenAI REST API / LLM
  ├── Agent 工具系统
  └── unitree_sdk2_python：执行 G1 预置动作和 TTS
```

两台机器位于同一个 `192.168.123.x` 网段。Orin 负责感知侧语音识别，Spark 负责 Agent 决策和机器人控制。

## 比赛推荐：优先使用三个自动化脚本

先在两台机器的 `speech_service` 目录复制并填写配置：

```bash
cp deploy.env.example deploy.env
```

然后按顺序执行：

```bash
# Orin
bash scripts/setup-orin.sh --config deploy.env

# Spark
bash scripts/setup-spark.sh --config deploy.env

# Spark，双端均已安装且 Orin 服务已启动后
bash scripts/verify-dual.sh --config deploy.env
```

变量说明、团队 Agent Observation/TTS 接入和最终人工放权流程见 [PIPELINE.md](./PIPELINE.md)。下文保留详细原理和手工故障恢复步骤，正常部署不需要逐条手工执行。

## 1. 最重要的部署结论

| 设备 | 安装内容 | 运行内容 |
|---|---|---|
| Orin NX | 完整 `g1-speech[orin]`、SenseVoice、Silero VAD、Unitree SDK2 Python | 麦克风采集、VAD、ASR、DDS 文本发布、播放门控订阅 |
| NVIDIA Spark | `g1_speech` 基础包、已有 Unitree SDK2 Python | DDS 文本订阅、事件去重、播放门控发布 |

两端使用同一份 `g1_speech` 源码，是为了共享完全一致的 DDS 消息类型；它们不会运行相同的业务：

```text
Orin  ── rt/g1/hri/speech/final ──► Spark Agent
Spark ─ rt/g1/hri/playback/state ─► Orin
```

Spark 不安装 SenseVoice 模型，也不运行 ASR。

## 2. 为什么源码包排除 `.venv`、`models` 和 `config.json`

### `.venv`

`.venv` 包含操作系统、CPU 架构和绝对路径相关内容。Mac 创建的虚拟环境不能在 Orin aarch64 Linux 或 Spark Linux 上运行，因此源码分发包必须排除 `.venv`，并在目标机器本地创建环境。

### `models`

SenseVoice 模型约 230MB，只在 Orin 使用，Spark 不需要。源码包排除模型是为了减小传输体积。

比赛现场不应依赖网络。正确做法是今晚在 Orin 完成模型下载和全链路验收，然后备份一份 Orin 可运行目录。

### `config.json`

它包含 Orin 专属配置：

- 麦克风设备编号；
- Orin 的 DDS 网卡名称；
- CPU/CUDA provider；
- 模型目录和 VAD 参数。

Spark 不读取这份配置，而且 Spark 网卡名称通常与 Orin 不同，所以不应把同一个 `config.json` 分发到两端。

## 3. 推荐准备两个安装包

### 通用源码包

在开发机执行：

```bash
cd /Users/liyang/Downloads/moyun_zhihe

tar \
  --exclude='speech_service/.venv' \
  --exclude='speech_service/models' \
  --exclude='speech_service/config.json' \
  -czf speech_service-src.tar.gz \
  speech_service
```

该包可以同时传给 Orin 和 Spark：

```bash
scp speech_service-src.tar.gz <ORIN_USER>@<ORIN_IP>:~/
scp speech_service-src.tar.gz <SPARK_USER>@<SPARK_IP>:~/
```

两端拿到同一套源码，但安装 extras 和运行入口不同。

### Orin 离线恢复包

在 Orin 完成安装、模型下载、配置和验收后执行：

```bash
cd ~
tar -czf g1-speech-orin-ready.tar.gz speech_service
```

它可以包含 Orin 自己创建的 `.venv`、模型和 `config.json`。该包用于恢复同一台或兼容环境的 Orin，不能复制给 Spark。

## 4. 网络确认

同一个 `192.168.123.x` 网段是正确前提，还要确认网卡和双向连通。

Orin：

```bash
ip -br addr
ping -c 4 <SPARK_IP>
```

Spark：

```bash
ip -br addr
ping -c 4 <ORIN_IP>
```

记录：

```text
ORIN_IP=
ORIN_DDS_IFACE=
SPARK_IP=
SPARK_DDS_IFACE=
DDS_DOMAIN_ID=0
```

Orin `config.json` 使用 `ORIN_DDS_IFACE`；Spark 已有 Agent 的 `ChannelFactoryInitialize` 使用 `SPARK_DDS_IFACE`。不要因为两台机器处于同一网段就把同一个网卡名称写到两端。

## 5. Orin 正常安装路径

### 5.1 解压源码

```bash
cd ~
tar xzf speech_service-src.tar.gz
cd speech_service
```

### 5.2 安装一次系统依赖

```bash
sudo apt update
sudo apt install -y \
  python3-venv \
  python3-dev \
  build-essential \
  libportaudio2 \
  portaudio19-dev \
  alsa-utils \
  curl \
  bzip2
```

`setup-orin.sh` 已经自动执行上述 apt 安装。本节命令仅用于手工故障恢复。

### 5.3 确认 Unitree SDK2 Python

```bash
python3 -c "import unitree_sdk2py, cyclonedds; print('Unitree DDS OK')"
```

如果能够导入，正常情况下只运行一次自动安装脚本：

```bash
bash scripts/setup-orin.sh --config deploy.env
```

该脚本会：

```text
创建 .venv
→ 安装 g1-speech[orin]
→ 检查 unitree_sdk2py/cyclonedds
→ 下载 SenseVoice 与 Silero VAD
→ 首次生成 config.json
```

不需要成功运行两次。

## 6. Orin 缺少 Unitree SDK 时的补救路径

本节是 `AUTO_INSTALL_UNITREE_SDK=0` 且 SDK 缺失时的手工补救路径，不是正常流程。

更清晰的做法是先创建虚拟环境：

```bash
cd ~/speech_service
python3 -m venv --system-site-packages .venv
```

按照宇树官方方法准备 CycloneDDS 0.10.x，并把 SDK 安装到同一个 `.venv`：

```bash
export CYCLONEDDS_HOME="$HOME/cyclonedds/install"

cd ~/unitree_sdk2_python
~/speech_service/.venv/bin/pip install -e .
```

验证：

```bash
~/speech_service/.venv/bin/python -c \
  "import unitree_sdk2py, cyclonedds; print('Unitree DDS OK')"
```

然后执行一次完整安装：

```bash
cd ~/speech_service
bash scripts/setup-orin.sh --config deploy.env
```

脚本是幂等的；如果此前已经运行到一半，再运行不会重复破坏已安装内容。

## 7. 麦克风与 Orin 配置

查看输入设备：

```bash
arecord -l

cd ~/speech_service
.venv/bin/python -c \
  "import sounddevice as sd; print(sd.query_devices())"
```

在 `config.json` 中填实际麦克风编号和 Orin DDS 网卡：

```json
{
  "audio": {
    "sample_rate": 16000,
    "block_ms": 100,
    "device": 3,
    "queue_seconds": 5.0
  },
  "sensevoice": {
    "model_dir": "models",
    "device": "cpu",
    "language": "zh",
    "use_itn": true,
    "num_threads": 6
  },
  "dds": {
    "domain_id": 0,
    "network_interface": "eth0",
    "speech_topic": "rt/g1/hri/speech/final",
    "playback_topic": "rt/g1/hri/playback/state",
    "delivery_ttl_seconds": 120.0
  }
}
```

这只是关键字段示例。请在安装脚本生成的完整 `config.json` 上修改，不要用上述片段覆盖整个文件。

## 8. 与 `moyun_zhihe` 实测参数对照

| 参数 | `moyun_zhihe` 当前实测配置 | `speech_service` 建议 |
|---|---:|---:|
| 模型 | SenseVoice int8 | 同一个模型 |
| 采样率 | 16000 | 16000 |
| provider | CPU | CPU |
| 语言 | 中文 | 中文 |
| ITN | 开启 | 开启 |
| CPU 线程 | 6 | 改为6 |
| 录音结束方式 | 按键 | Silero VAD |
| 最长语音 | 配置10秒 | 建议10秒 |
| VAD pre-roll | 无 | 300ms |

为了尽量复现已经验证过的效果，建议：

```json
"sensevoice": {
  "device": "cpu",
  "language": "zh",
  "use_itn": true,
  "num_threads": 6
}
```

以及：

```json
"vad": {
  "threshold": 0.5,
  "speech_pre_roll_seconds": 0.3,
  "min_silence_seconds": 0.35,
  "min_speech_seconds": 0.25,
  "max_speech_seconds": 10.0
}
```

SenseVoice 核心参数一致，但“按键停止”改成了“VAD 自动断句”，因此实际麦克风和现场噪声测试不可省略。优先调整麦克风位置和 VAD，不要先修改已经验证过的 SenseVoice 参数。

## 9. Orin 验收

检查环境和模型：

```bash
cd ~/speech_service
.venv/bin/g1-speech doctor --config config.json --load-model
```

识别官方测试 WAV：

```bash
.venv/bin/g1-speech transcribe \
  --config config.json \
  models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17/test_wavs/zh.wav
```

800ms 延迟验收：

```bash
.venv/bin/python acceptance/e2e_latency.py \
  --config config.json \
  --wav models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17/test_wavs/zh.wav \
  --target-ms 800
```

CPU 已达到目标时不要在比赛前临时切 CUDA。

## 10. 在 Spark 安装基础包不会覆盖团队 Agent

进入团队 Agent 实际使用的 Python/Conda 环境：

```bash
conda activate <AGENT_ENV>
python -c "import unitree_sdk2py, cyclonedds; print('Agent DDS OK')"
```

如果 NumPy 已经是 1.24 或更高：

```bash
python -c "import numpy; print(numpy.__version__)"
python -m pip install --no-deps -e ~/speech_service
```

该命令只注册一个名为 `g1_speech` 的新 Python 包，不会：

- 删除或覆盖团队 Agent 源码；
- 修改 Prompt；
- 修改 OpenAI REST API 调用；
- 替换工具系统；
- 替换现有 Unitree 动作控制代码；
- 自动启动任何进程。

Spark 不使用 `[orin]` extras，因此不会安装 SenseVoice、sounddevice 或模型。

## 11. `agent_subscriber.py` 只用于独立测试

`examples/agent_subscriber.py` 验证：

```text
Orin SenseVoice → DDS → Spark → 打印文本
```

测试时，在团队正式 Agent 停止或动作执行器处于 dry-run 的情况下运行：

```bash
python ~/speech_service/examples/agent_subscriber.py <SPARK_DDS_IFACE>
```

看到文本后关闭它。它不会调用 OpenAI、不会运行团队 Agent、不会执行工具，也不应作为比赛时的正式入口。

## 12. “正式 Agent”就是团队比赛 Agent

这里的正式 Agent 指团队已经开发的系统：

```text
接收 Observation
→ 构建上下文
→ 调用 OpenAI REST API / LLM
→ 解析 tool calls
→ 调用团队工具
→ unitree_sdk2_python 执行 G1 动作
```

`speech_service` 只为它增加一种 Observation：

```text
Observation
├── 机器人状态
├── RealSense 图像
├── 视觉算法结果
└── 语音识别文本  ← 新增
```

## 13. 接入同步队列型 Agent

团队 Agent 应该已经调用过：

```python
ChannelFactoryInitialize(0, spark_network_interface)
```

在此之后创建语音订阅器，同一 Python 进程不要再次初始化 DDS：

```python
from queue import Full
from g1_speech.dds import DdsSpeechSubscriber


def on_speech(event):
    observation = {
        "type": "speech",
        "source": "g1_microphone",
        "event_id": event.event_id,
        "session_id": event.session_id,
        "sequence": event.sequence,
        "timestamp_ns": event.created_unix_ns,
        "text": event.text,
        "is_final": event.is_final,
    }
    try:
        observation_queue.put_nowait(observation)
    except Full:
        logger.error("Observation queue full; drop %s", event.event_id)


speech_subscriber = DdsSpeechSubscriber(on_speech)
speech_subscriber.start()
```

Agent 原有主循环只需认识 `speech` 类型：

```python
observation = observation_queue.get()

if observation["type"] == "speech":
    agent_messages.append({
        "role": "user",
        "content": observation["text"],
    })

response = call_openai_rest_api(agent_messages, tools=tools)
execute_tool_calls(response)
```

DDS 回调只负责入队，不要直接在回调线程调用 LLM、视觉推理或机器人动作。

Agent 退出时：

```python
speech_subscriber.close()
```

## 14. 接入 asyncio Agent

DDS 回调运行在线程中，不能直接 `await` 异步队列。应使用 Agent 事件循环的线程安全入口：

```python
def on_speech(event):
    observation = {
        "type": "speech",
        "event_id": event.event_id,
        "text": event.text,
        "timestamp_ns": event.created_unix_ns,
    }
    event_loop.call_soon_threadsafe(
        async_observation_queue.put_nowait,
        observation,
    )
```

## 15. 接入 Agent 的 TTS/音频工具

动作工具不需要修改。只包装会让 G1 扬声器发声的工具，例如 TTS、播放音乐或 WAV。

Agent 启动时：

```python
from g1_speech.dds import DdsPlaybackPublisher

playback_gate = DdsPlaybackPublisher()
playback_gate.start()
```

播放工具：

```python
import uuid


def speak(text):
    request_id = str(uuid.uuid4())
    playback_gate.set_active_reliably(True, request_id=request_id)
    try:
        audio_client.TtsMaker(text, 0)
        wait_until_audio_really_finishes()
    finally:
        playback_gate.set_active_reliably(False, request_id=request_id)
```

`active=false` 必须在扬声器实际播放结束后发送。没有完成回调时，固定台词应提前测量时长并增加安全余量。

如果表演完全不使用 G1 扬声器，可以先只接入 `DdsSpeechSubscriber`。

## 16. 推荐联调顺序

1. 双机互相 `ping`。
2. Orin 完成 `doctor`、测试 WAV 和800ms验收。
3. Spark 单独运行 `agent_subscriber.py`，团队 Agent 保持关闭或 dry-run。
4. Orin 启动 `g1-speech serve`，真人说话，Spark 应打印一次文本。
5. 关闭测试订阅器。
6. 在团队 Agent 的 Observation 层接入 `DdsSpeechSubscriber`。
7. Agent 先只记录 Observation，不执行动作。
8. 确认同一 `event_id` 只入队一次。
9. 开启一个低风险动作工具进行端到端测试。
10. 最后接入 TTS 播放门控并验证没有自触发。

## 17. 仍需结合团队 Agent 代码确认的部分

本文使用通用的 `observation_queue` 示例。精确接入位置取决于团队 Agent 的：

- 入口文件；
- Observation 数据结构；
- 同步还是 asyncio 主循环；
- OpenAI REST 请求封装；
- tool call 调度器；
- TTS/音频工具实现。

要把集成代码精确落到团队 Agent 中，需要提供这些文件或接口定义。`speech_service` 不应该猜测或替换团队已有 Agent 架构。
