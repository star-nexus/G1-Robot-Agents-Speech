# 团队 Agent 手工接入 Pipeline

> 当前语音服务和 Agent 都运行在 DGX 上。下文原有的 Spark/Orin 名称应分别理解为
> “DGX Agent 进程”和“DGX speech_service 进程”；DDS Topic、去重、Observation 入队和
> TTS 播放门控的接入方式保持不变。部署步骤以 [DGX_DEPLOYMENT.md](./DGX_DEPLOYMENT.md)
> 为准。

三个自动化脚本负责安装和双机通信验证，但不会修改团队比赛 Agent 的源码。本文只描述真正需要人工结合 Agent 架构完成的部分。

## 自动化与手工工作的边界

自动完成：

```text
Orin 系统/Python依赖、模型、config、doctor、延迟测试
Spark Agent Python 环境中的 g1_speech 包注册
双机 ping、环境、speech/final、playback/state 联测
```

必须人工确认：

```text
Agent 入口文件
DDS 初始化位置
Observation 数据结构和队列
同步或 asyncio 主循环
OpenAI REST 请求入口
tool call 调度位置
TTS/音频工具的实际播放结束条件
```

## 1. 部署前填写 `deploy.env`

两端都从模板复制：

```bash
cp deploy.env.example deploy.env
```

必须填写：

| 变量 | 示例 | 如何确认 |
|---|---|---|
| `ORIN_HOST` | `192.168.123.161` | Orin 上运行 `ip -br addr` |
| `SPARK_HOST` | `192.168.123.99` | Spark 上运行 `ip -br addr` |
| `ORIN_DDS_IFACE` | `eth0` | Orin 到 Spark 的实际网卡 |
| `SPARK_DDS_IFACE` | `enp3s0` | Spark 到 Orin 的实际网卡 |
| `ORIN_USER` | `unitree` | Orin 上运行 `whoami` |
| `SPARK_AGENT_PYTHON` | `/home/spark/agent/.venv/bin/python` | 在 Agent 环境运行 `which python` |
| `ORIN_INSTALL_DIR` | `/home/unitree/speech_service` | Orin 上 `cd speech_service && pwd` |

可选人工填写：

- `ORIN_MIC_DEVICE`：空值时 Orin 安装脚本会交互式列出并询问；只有一个正确的默认麦克风时可以直接回车。
- `AUTO_INSTALL_UNITREE_SDK`：已有 SDK 保持 `0`；确实缺失且有网络时才改为 `1`。
- `INSTALL_SYSTEMD`：默认 `1`，安装通过后自动启用并启动 Orin 服务；只有需要手工调试时才改为 `0`。
- `MODEL_SOURCE_DIR`：现场离线安装时指向提前准备的模型目录。

## 2. 三个脚本的执行顺序

### Orin

```bash
cd ~/speech_service
bash scripts/setup-orin.sh --config deploy.env
```

### Spark

```bash
cd ~/speech_service
bash scripts/setup-spark.sh --config deploy.env
```

### 双机验收（在 Spark 执行）

```bash
cd ~/speech_service
bash scripts/verify-dual.sh --config deploy.env
```

双机验收会要求你在15秒窗口内对 G1 麦克风说一句话。

默认 `INSTALL_SYSTEMD=1` 时，Orin 安装脚本已启动服务。如果设置为 `0`，请先在 Orin 另开终端运行：

```bash
.venv/bin/g1-speech serve --config config.json
```

## 3. 找到团队 Agent 的接入位置

先在 Agent 项目中找到四处代码：

```text
A. ChannelFactoryInitialize(...) 的位置
B. Observation/Perception 队列定义
C. Agent 主循环读取 Observation 的位置
D. Agent 退出时释放资源的位置
```

不要在 `agent_subscriber.py` 上继续开发。它只是独立 DDS 验收程序，比赛时不运行。

## 4. 在现有 DDS 初始化之后创建订阅器

团队 Agent 通常已经执行：

```python
ChannelFactoryInitialize(0, spark_network_interface)
```

在同一进程中只初始化一次 DDS。紧接着增加：

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

DDS 回调线程只负责入队，不能直接调用 OpenAI、视觉推理、tool call 或机器人动作。

## 5. 如果 Agent 使用 asyncio

使用 Agent 事件循环的线程安全入口：

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

不要在 DDS 回调线程直接操作只属于 asyncio 主线程的对象。

## 6. 将语音 Observation 接入现有 LLM 流程

在主循环读取 Observation 的位置增加一种类型，不改变现有 OpenAI REST 和工具架构：

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

如果团队 Agent 使用自定义 `Observation` dataclass，请把 `SpeechEvent` 转换成该类型，而不是把 Agent 全部改成字典。

## 7. 在 Agent 退出时关闭订阅器

```python
speech_subscriber.close()
```

## 8. 包装会让 G1 发声的工具

只有 TTS、播放 WAV、音乐等扬声器工具需要播放门控；走路、挥手、跳舞等动作工具不需要修改。

Agent 启动时创建一次 publisher：

```python
from g1_speech.dds import DdsPlaybackPublisher

playback_gate = DdsPlaybackPublisher()
playback_gate.start()
```

包装已有 TTS 工具：

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

退出时：

```python
playback_gate.close()
```

`active=false` 必须在扬声器实际播放结束之后发送。没有完成回调时，提前测量固定台词时长并增加安全余量。

## 9. 最终人工验收

按以下顺序逐步放开权限：

```text
1. 语音事件只记录日志，不调用 LLM
2. 语音进入 Observation，但 tool executor 保持 dry-run
3. 允许 OpenAI REST 返回 tool calls，但只打印
4. 只开放一个低风险 G1 预置动作
5. 验证相同 event_id 不会执行两次
6. 接入 TTS 门控并验证没有自触发
7. 最后开放完整表演动作白名单
```

## 10. 精确接入仍需要的团队代码

如果需要把上述代码直接落到团队 Agent，至少需要提供：

- Agent 入口文件；
- DDS/Unitree SDK 初始化文件；
- Observation 类型定义；
- Observation 队列或事件总线；
- OpenAI REST 调用封装；
- tool call dispatcher；
- TTS/音频工具。

在没有这些代码时，安装脚本不会猜测或修改团队 Agent。
