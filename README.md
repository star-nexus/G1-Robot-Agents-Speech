# G1 Speech Service

> 把任意麦克风变成机器人 Agent 可直接订阅的离线语音输入。

G1 Speech Service 持续完成 **麦克风采集 → VAD 断句 → SenseVoice 识别 → DDS 发布**。
Agent 只需要订阅一个 Topic，不需要了解模型、音频驱动、线程或推理后端。

- **接入简单**：几行 Python 即可获得最终识别文本
- **通用解耦**：服务与 Agent 通过 DDS 通信，可独立部署、独立升级
- **CPU/GPU 可选**：默认 CPU INT8；Jetson 可选 CUDA FP32，Agent 代码无需变化
- **完全离线**：音频和文本不离开本地设备
- **面向机器人**：内置播放门控、DDS 重试、事件去重和有界队列

## 性能

Jetson ORIN NX，同一段 5.592 秒中文音频，预热 3 次、运行 30 次：

| 后端 | 模型 | 平均耗时 | 中位数 | P95 | RTF |
|---|---|---:|---:|---:|---:|
| CPU | INT8 | 206.9 ms | 206.8 ms | 208.6 ms | 0.0370 |
| GPU | FP32 | 69.5 ms | 64.5 ms | 94.3 ms | 0.0124 |

GPU 平均推理约快 **3 倍**，而服务接口和 DDS 消息完全一致。以上是速度测试；
准确率应使用自己的业务语料单独评测。

## 一分钟接入

### 运行语音服务

部署完成后，选择一个后台后端：

```bash
# 低资源、默认选择
sudo g1-speech-service cpu

# Jetson CUDA 加速
sudo g1-speech-service gpu

g1-speech-service status
g1-speech-service logs
```

选择器会停止另一后端，并把当前选择设为开机启动。同一时间只有一个服务读取麦克风
并向 `rt/g1/hri/speech/final` 发布结果。

### Agent 订阅

```python
from g1_speech.dds import DdsSpeechSubscriber, initialize_unitree_dds

# 每个进程只初始化一次；如果 Agent 已初始化 Unitree DDS，请跳过这一行。
initialize_unitree_dds(domain_id=0)

subscriber = DdsSpeechSubscriber(lambda event: print(event.text))
subscriber.start()
```

Agent 退出时调用 `subscriber.close()`。回调运行在 DDS 线程中，正式 Agent 建议只在
回调里入队，再由业务线程调用 LLM、工具或机器人动作。

无需编写 Agent 代码也可以快速查看识别结果：

```bash
.venv/bin/g1-speech listen --config config.json --timeout 0
```

## 工作方式

```text
Microphone
  → 16 kHz mono audio
  → Silero VAD
  → SenseVoice-Small
  → rt/g1/hri/speech/final
  → Robot / Agent / ROS bridge / application

TTS or playback
  → rt/g1/hri/playback/state
  → temporarily pause recognition
```

| Topic | 方向 | 用途 |
|---|---|---|
| `rt/g1/hri/speech/final` | Service → Agent | 最终语音识别事件 |
| `rt/g1/hri/playback/state` | Agent → Service | 播放期间暂停识别，避免机器人听到自己 |

`SpeechEvent` 包含稳定的 `event_id`、文本、语言、音频时长、推理耗时、来源和时间戳。
发布端会在短暂 DDS 故障时重试，订阅端会按 `event_id` 去重。

## 部署

### CPU INT8

CPU 是默认后端，适合资源受限设备和不具备 CUDA 的 Linux 主机。

```bash
git clone https://github.com/star-nexus/Unitree_G1_Voice.git
cd Unitree_G1_Voice

cp deploy.env.example deploy.env
# 设置实际 DDS 网卡；麦克风留空时可在安装过程中选择
# ORIN_DDS_IFACE="your-interface"

bash scripts/setup-orin.sh
```

脚本会创建项目私有的 `.venv`、下载模型、运行 doctor 和固定 WAV 测试，并安装 CPU
systemd 服务。

### Jetson GPU FP32

GPU 环境与 CPU 环境完全隔离，不会覆盖 `.venv`、`config.json` 或 CPU INT8 模型。

```bash
bash scripts/setup-orin-gpu.sh
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
| `ORIN_DDS_IFACE` | DDS 使用的本机网卡，可用 `ip -br link` 查看 |
| `ORIN_MIC_DEVICE` | PortAudio 输入编号或设备名；留空可交互选择 |
| `DDS_DOMAIN_ID` | 与订阅方一致的 DDS Domain |
| `SPEECH_TOPIC` | 最终识别结果 Topic |
| `PLAYBACK_TOPIC` | TTS/播放门控 Topic |
| `SENSEVOICE_THREADS` | CPU 推理线程数 |

运行时参数位于 `config.json`；GPU 部署使用独立的 `config.gpu.json`。相对模型路径
按配置文件所在目录解析。

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

sudo g1-speech-service restart
sudo g1-speech-service stop
```

底层 systemd 单元：

- CPU：`g1-speech.service`
- GPU：`g1-speech-gpu.service`
