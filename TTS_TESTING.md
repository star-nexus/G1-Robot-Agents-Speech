# Unitree G1 TTS 测试方法

本文说明如何从 DGX 调用 Unitree G1 的 TTS，让机器人说出指定文字，并在播放期间暂停
`speech_service` 的麦克风识别，避免机器人把自己的声音识别成用户指令。

## 1. 通信流程

```text
DGX 测试程序
  ├─ 发布 playback/state active=true
  ├─ 调用 G1 AudioClient.TtsMaker(text, speaker_id)
  ├─ 等待机器人播放结束
  └─ 发布 playback/state active=false

G1
  └─ 扬声器播放 TTS

DGX g1-speech.service
  └─ active=true 期间暂停麦克风识别
```

TTS 使用 Unitree DDS/RPC，不需要 SSH 登录 G1，也不需要在脚本中保存 G1 或 DGX 密码。

## 2. 前置条件

- DGX 的机器人网卡为 `enP7s7`，地址为 `192.168.123.100/24`；
- G1 可通过 `192.168.123.164` 访问；
- `speech_service` 已安装在 `/home/dgx/moonbot/speech_service`；
- `g1-speech.service` 正在运行；
- G1 已开机，扬声器可用，现场允许播放声音。

## 3. 测试前检查

在 DGX 执行：

```bash
cd /home/dgx/moonbot/speech_service

ip link show enP7s7
ping -I enP7s7 -c 2 -W 2 192.168.123.164
systemctl --user is-active g1-speech.service
```

预期 G1 ping 无丢包，且 `g1-speech.service` 返回 `active`。

## 4. 执行 TTS 测试

下面的命令让 G1 说“你们好”：

```bash
cd /home/dgx/moonbot/speech_service

export LD_LIBRARY_PATH="$PWD/.deps/portaudio/root/usr/lib/aarch64-linux-gnu:$PWD/.deps/cyclonedds/install/lib"
export LIBRARY_PATH="$LD_LIBRARY_PATH"

.venv/bin/python - <<'PY'
import time
import uuid

from g1_speech.dds import DdsPlaybackPublisher, initialize_unitree_dds
from unitree_sdk2py.g1.audio.g1_audio_client import AudioClient


# 本测试进程只初始化一次 DDS。
initialize_unitree_dds(0, "enP7s7")

audio = AudioClient()
audio.SetTimeout(10.0)
audio.Init()

volume_code, volume_data = audio.GetVolume()
print(f"GetVolume code={volume_code} data={volume_data}")
if volume_code != 0:
    raise SystemExit(f"G1 audio service unavailable, code={volume_code}")

gate = DdsPlaybackPublisher(source="tts-test")
gate.start()
request_id = "tts-test-" + str(uuid.uuid4())
gate_active = False

try:
    # 确认 speech_service 已收到暂停识别消息后才开始播放。
    gate.set_active_reliably(True, request_id=request_id)
    gate_active = True

    code = audio.TtsMaker("你们好", 0)
    print(f"TtsMaker code={code} request_id={request_id}")
    if code != 0:
        raise SystemExit(f"TTS failed, code={code}")

    # TtsMaker 返回表示请求已接受，不表示扬声器已经播放完毕。
    time.sleep(4.0)
finally:
    if gate_active:
        gate.set_active_reliably(False, request_id=request_id)
    gate.close()

print("TTS test completed")
PY
```

测试其他内容时，只修改：

```python
audio.TtsMaker("你们好", 0)
```

第一个参数是文本，第二个参数是 `speaker_id`，当前使用 `0`。

## 5. 成功判据

同时满足以下条件才算通过：

1. `GetVolume code=0`；
2. `TtsMaker code=0`；
3. 现场能听到 G1 完整说出目标文本；
4. 播放期间没有把 G1 自己的声音识别成新的用户指令；
5. 测试结束后 `g1-speech.service` 仍为 `active`。

本次已验证结果：

```text
文本：你们好
音量：100
GetVolume：0
TtsMaker：0
现场听音：通过
播放门控：active=true → active=false
```

## 6. 检查播放门控日志

```bash
journalctl --user -u g1-speech.service --since '-5 min' --no-pager \
  | grep '播放门控'
```

预期看到同一个 `request_id` 的两条日志：

```text
播放门控 active=True  request_id=tts-test-...
播放门控 active=False request_id=tts-test-...
```

如果只有 `active=True`，麦克风门控最多会在配置的 30 秒后自动解除，但仍需检查测试程序为何
没有发送 `active=False`。

## 7. 常见问题

### G1 无法连接

```bash
ping -I enP7s7 192.168.123.164
```

检查网线、网卡地址以及 G1 是否开机。TTS 不依赖 SSH，ping 成功后仍应继续检查 DDS。

### `GetVolume` 或 `TtsMaker` 返回非零

表示 G1 Audio 服务没有接受请求。确认 DDS 使用 Domain `0` 和网卡 `enP7s7`，并确认没有
其他程序错误初始化 Unitree DDS。

### 返回 0 但听不到声音

先检查 `GetVolume` 返回的音量，并确认 G1 扬声器没有被静音。不要在无人确认的情况下突然
把音量调到最大。

### G1 的声音被再次识别

检查测试代码是否在 `TtsMaker` 之前发送 `active=True`，并检查
`g1-speech.service` 日志是否收到门控消息。播放结束前不能提前发送 `active=False`。

### 连续播放多句话

每句话必须等待实际播放完成后再发送下一句。Agent 正式工具应使用互斥锁串行执行 TTS，避免
多个 tool call 同时播放；较长文本应限制长度或分句，并为播放时长保留余量。

## 8. Agent 正式接入注意事项

- Agent 进程只调用一次 `ChannelFactoryInitialize(0, "enP7s7")`；
- `AudioClient` 和 `DdsPlaybackPublisher` 在 Agent 启动时创建并复用；
- 每次播放使用新的 `request_id`；
- 使用 `try/finally` 保证发送 `active=False`；
- TTS 工具应串行执行；
- Agent 退出时关闭 `DdsPlaybackPublisher`。

