# G1 Speech Service

G1 上的离线快速 ASR 语音服务

## 当前拓扑

```text
无线/USB
  → sounddevice / 16 kHz mono
  → Silero VAD
  → SenseVoice-Small
  → 发布 rt/g1/hri/speech/final
  → DGX Agent 本机订阅

DGX Agent/TTS
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

## 服务管理

```bash
systemctl --user status g1-speech.service
journalctl --user -u g1-speech.service -f
systemctl --user restart g1-speech.service
```

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
