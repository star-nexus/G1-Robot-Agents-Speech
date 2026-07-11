# DGX 单机语音服务部署

当前拓扑不再使用 G1 板载 Orin NX：

```text
DGX USB 麦克风
  → Silero VAD
  → SenseVoice-Small
  → 本机发布 rt/g1/hri/speech/final
  → DGX Agent 本机订阅
```

DDS 使用与 Agent/机器人 SDK 相同的 Domain 和网卡。当前 DGX 连接 G1 的网卡为
`enP7s7`，地址为 `192.168.123.100/24`。本机发布和订阅无需访问 G1 Orin 的
`192.168.123.164`。

## 安装

```bash
cd /home/dgx/moonbot/speech_service
cp deploy-dgx.env.example deploy-dgx.env
bash scripts/setup-dgx.sh --config deploy-dgx.env
```

脚本创建独立 `.venv`、私有 PortAudio、私有 CycloneDDS/Unitree SDK2 Python、下载模型、
生成 `config.json`、运行离线识别和延迟验收，并安装用户级 systemd unit。整个过程不需要
sudo。`unitree_sdk2` C++ SDK 与 `unitree_sdk2py` 是两个不同依赖；语音服务使用后者。

## 无麦克风自动验收

```bash
bash scripts/verify-local.sh --config deploy-dgx.env
```

该命令会运行单元测试，并用合成 `SpeechEvent` 验证 DGX 本机 DDS 发布与订阅。

## 插入麦克风后启用服务

先确认设备：

```bash
source deploy-dgx.env
export LD_LIBRARY_PATH="$PWD/.deps/portaudio/root/usr/lib/aarch64-linux-gnu:$CYCLONEDDS_HOME/lib"
.venv/bin/python -c 'import sounddevice as sd; print(sd.query_devices())'
```

把 `deploy-dgx.env` 中的 `DGX_MIC_DEVICE` 改成输入设备编号或名称。当前无线 USB
麦克风硬件只支持 48 kHz，因此使用 `DGX_MIC_DEVICE="pipewire"`，由 PipeWire 重采样
到识别算法要求的 16 kHz。然后设置：

```text
REQUIRE_MIC=1
START_SERVICE=1
```

重新运行安装脚本（幂等），然后做真人语音验收：

```bash
bash scripts/setup-dgx.sh --config deploy-dgx.env
bash scripts/verify-local.sh --config deploy-dgx.env --live
```

查看服务日志：

```bash
journalctl --user -u g1-speech.service -f
```

## Agent 订阅

Agent 进程完成一次 `ChannelFactoryInitialize(0, "enP7s7")` 后，创建
`DdsSpeechSubscriber`。不要在同一进程重复初始化 DDS：

```python
from g1_speech.dds import DdsSpeechSubscriber

speech = DdsSpeechSubscriber(lambda event: observation_queue.put_nowait({
    "type": "speech",
    "event_id": event.event_id,
    "text": event.text,
}))
speech.start()
```

也可以先用独立命令查看本机识别事件：

```bash
.venv/bin/g1-speech listen --config config.json
```
