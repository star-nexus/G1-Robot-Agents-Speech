# G1 Speech Service

面向 Unitree G1 的离线 ASR 服务。默认使用 CPU INT8，Jetson 可选 CUDA FP32；识别结果
统一发布到 DDS，Agent 不需要感知 SenseVoice、VAD、音频设备或推理后端。

## 数据流

```text
无线/USB
  → sounddevice / 16 kHz mono
  → Silero VAD
  → SenseVoice-Small
  → 发布 rt/g1/hri/speech/final
  → Agent 订阅

Agent/TTS
  → 发布 rt/g1/hri/playback/state
  → speech_service 本机订阅并暂停识别
```

G1 固定地址 `192.168.123.164` 不影响单机拓扑。DDS 绑定 DGX 的机器人网卡 `enP7s7`
（当前为 `192.168.123.100/24`），以便 Agent 在同一个 DDS 初始化中同时使用语音 Topic
和 Unitree SDK。

## 部署

脚本不需要 sudo，会安装以下项目私有组件：

- Python 3.11 虚拟环境；
- PortAudio；
- CycloneDDS 0.10.x 和 Unitree SDK2 Python；
- sherpa-onnx、SenseVoice 与 Silero VAD；
- `~/.config/systemd/user/g1-speech.service`。

仓库中的 C++ `unitree_sdk2` 不包含 `unitree_sdk2py`，因此 Python DDS SDK 是单独的私有依赖。

## 验收

不依赖说话的自动测试：

```bash
bash scripts/verify-local.sh --config deploy-dgx.env
```

它会运行全部单元测试，并用合成中文 `SpeechEvent` 验证本机 DDS 发布与订阅。

真人语音闭环：

```bash
bash scripts/verify-local.sh --config deploy-dgx.env --live
```

在 20 秒内对麦克风说一句话。也可以持续查看识别事件：

```bash
.venv/bin/g1-speech listen --config config.json
```

## Jetson ORIN CPU/GPU 服务

CPU INT8 是默认部署，GPU 版本使用独立的 `.venv-gpu`、`config.gpu.json` 和
`g1-speech-gpu.service`，不会覆盖 CPU 环境。首次部署 CPU：

```bash
cp deploy.env.example deploy.env
# 按现场网卡和麦克风修改 deploy.env
bash scripts/setup-orin.sh
```

安装可选 GPU 后端：

```bash
bash scripts/setup-orin-gpu.sh
```

CPU 和 GPU 都作为 systemd 后台服务运行。使用统一命令选择后端：

```bash
sudo g1-speech-service cpu
sudo g1-speech-service gpu

g1-speech-service status
g1-speech-service logs
sudo g1-speech-service restart
sudo g1-speech-service stop
```

选择器会停止并禁用另一后端，只保留一个服务读取麦克风和发布 DDS；所选后端会设为
开机启动。两种服务发布相同 Topic，Agent 接入代码不需要变化。

GPU 脚本固定编译 sherpa-onnx 1.13.4、CUDA Execution Provider 和 Jetson aarch64
ONNX Runtime 1.18.1。若 CUDA wheel 不可用，服务会报错退出，不会静默回退 CPU。

### 性能测试

Jetson ORIN NX 上使用同一段 5.592 秒中文音频，预热 3 次、运行 30 次：

| 后端 | 模型 | 平均耗时 | 中位数 | P95 | RTF |
|---|---|---:|---:|---:|---:|
| CPU | INT8 | 206.9 ms | 206.8 ms | 208.6 ms | 0.0370 |
| GPU | FP32 | 69.5 ms | 64.5 ms | 94.3 ms | 0.0124 |

复现命令：

```bash
.venv/bin/python acceptance/benchmark_engine.py \
  --config config.json --wav models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17/test_wavs/zh.wav

.venv-gpu/bin/python acceptance/benchmark_engine.py \
  --config config.gpu.json --wav models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/test_wavs/zh.wav
```

以上只测试速度，不代表准确率结论。GPU 使用 FP32 模型，可避免 INT8 量化造成的潜在
精度损失；准确率应使用业务语料单独评测。

## Agent 接入

Agent 进程应只初始化一次 DDS，然后启动订阅器。DDS 回调只负责入队，不应直接调用
LLM、工具或机器人动作：

```python
from g1_speech.dds import DdsSpeechSubscriber


def on_speech(event):
    observation_queue.put_nowait({
        "type": "speech",
        "event_id": event.event_id,
        "text": event.text,
        "timestamp_ns": event.created_unix_ns,
    })


speech_subscriber = DdsSpeechSubscriber(on_speech)
speech_subscriber.start()
```

Agent 退出时执行 `speech_subscriber.close()`。TTS 播放门控和更完整的 Observation 接入方式
见 [PIPELINE.md](./PIPELINE.md)。

G1 TTS 的独立测试步骤和成功判据见 [TTS_TESTING.md](./TTS_TESTING.md)。

## 历史双机脚本

`setup-orin.sh`、`setup-spark.sh` 和 `verify-dual.sh` 保留作旧版参数与故障排查参考，
不用于当前 DGX 单机部署。
