# G1 Speech Service — Offline ASR for Robot Agents

English | [简体中文](README_ZH.md)

Offline speech recognition for robot Agents, optimized for NVIDIA Jetson.
Give robots fast, private speech input for responsive, real-time interaction with people—without relying on the cloud.

**Tested on [Unitree G1](https://www.unitree.com/mobile/g1/) and
[Galbot G1](https://www.galbot.com/g1). Benchmarked on NVIDIA Jetson Orin NX.**
It also runs on other Jetson-powered robots and Linux edge computers such as NVIDIA DGX Spark.

- **CPU and GPU backends**: CPU INT8 and Jetson CUDA FP32 deployment modes
- **Fully offline**: Speech recognition runs locally—audio and text never need to leave the device
- **Resilient audio capture**: Stream heartbeat, automatic microphone reconnection, and bounded queues
- **DDS and ROS 2 transports**: Native structured topics for lightweight DDS systems and ROS 2 robots
- **High performance**: 0.01–0.2 s recognition latency with high accuracy

## Verified Platforms

| Platform | Compute | Available backends | Validation |
|---|---|---|---|
| Unitree G1 | Jetson Orin NX test configuration | CPU INT8 / CUDA FP32 | ✅ Verified |
| Galbot G1 | Jetson Orin test configuration | CPU INT8 / CUDA FP32 | ✅ Verified |
| Jetson Orin NX edge systems | Jetson Linux | CPU INT8 / CUDA FP32 | ✅ Benchmark platform |
| NVIDIA DGX Spark | ARM64 Linux | CPU INT8 | ✅ Original deployment |
| Other Jetson/Linux robots | Jetson or Linux edge computer | CPU INT8; CUDA FP32 on Jetson | Compatibility target |

Validation refers to configurations tested by this project; vendor hardware configurations may vary.

## Performance

Benchmarked on a Jetson Orin NX using the same 5.592-second Chinese audio sample, with 3 warm-up runs and 30 measured runs:

| Backend | Model | Mean latency | Median | P95 | RTF |
|---|---|---:|---:|---:|---:|
| CPU | INT8 | 206.9 ms | 206.8 ms | 208.6 ms | 0.0370 |
| GPU | FP32 | 69.5 ms | 64.5 ms | 94.3 ms | 0.0124 |

## Quick Start

### Start the Speech Service

After deployment, select the inference backend and transport. Every mode is a
managed background service and starts again automatically after reboot:

```bash
# Low-resource, default option
sudo g1-speech-service cpu

# Jetson CUDA acceleration
sudo g1-speech-service gpu

# ROS 2 transport (CPU or GPU)
sudo g1-speech-service cpu ros2
sudo g1-speech-service gpu ros2

g1-speech-service status
g1-speech-service logs
```

Pin the microphone on robots that have more than one audio input. USB cameras
often expose a microphone, and PulseAudio may automatically switch its default
input when the camera is connected. Choose a distinctive device-name substring
from the device list and set it in the host-local `deploy.env`:

```bash
.venv/bin/python -c 'import sounddevice as sd; print(sd.query_devices())'

# deploy.env — example only; use a name shown on your own machine
MICROPHONE_DEVICE="USB Microphone"

sudo g1-speech-service restart
g1-speech-service status
```

Prefer a stable name over a numeric index, which may change after reboot or USB
re-enumeration. `status` and `logs` show the configured microphone; when set to
`pulse` or the PortAudio default, they also show the current PulseAudio source
and warn that hot-plugging can change it.

DDS is the default transport. Recognition results are published to
`rt/g1/hri/speech/final`. Both CPU and GPU backends use the same messages, so
no Agent-side changes are required.

### Subscribe from an Agent

```python
from g1_speech.dds import DdsSpeechSubscriber, initialize_dds

# Initialize Cyclone DDS once per Agent process.
initialize_dds(domain_id=0)

subscriber = DdsSpeechSubscriber(lambda event: print(event.text))
subscriber.start()
```

Call `subscriber.close()` when the Agent exits. The callback runs on a dedicated subscriber thread; in production Agents, enqueue the event in the callback and let a worker thread invoke the LLM, tools, or robot actions.

You can also inspect recognition results without writing Agent code:

```bash
.venv/bin/g1-speech listen --config config.json --timeout 0
```

### Local Qwen Voice Chat

Use speech recognition as input to a local Qwen3 model. The demo streams text
replies in the terminal and keeps a bounded conversation history; it does not
use TTS. A llama.cpp OpenAI-compatible server is the default backend:

```bash
# Use `llama serve` if that installation has the desired accelerator backend,
# or launch a standalone llama-server built with GGML_CUDA=ON on Jetson.
llama-server -m /path/to/Qwen3-8B-Q5_K_M.gguf --alias qwen3-8b-q5 \
  --host 127.0.0.1 --port 8080 -c 4096 -ngl all -np 1 \
  --cache-ram 0 --no-cache-prompt --no-cache-idle-slots \
  --flash-attn on --reasoning off --no-webui

# Follow the active DDS or ROS 2 speech service automatically
scripts/run-qwen-voice-chat --model qwen3-8b-q5
```

For a vision-language model served by llama.cpp, attach a V4L2 camera. The
default `see` strategy lets Qwen answer text-only questions in its first normal
generation. When the current camera view is required, Qwen emits an internal
`[SEE]` marker; the marker is hidden, the latest frame is attached, and the
multimodal answer is generated. Explicit visual requests bypass the probe and
attach a frame immediately:

```bash
# 16 GB Jetson memory-safe Qwen3-VL server profile. Set LLAMA_SERVER_BIN
# only when llama-server is not available on PATH.
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

For Qwen3-VL, the client automatically uses the model-card sampling profiles:
text requests use `temperature=1.0`, `top_p=1.0`, `top_k=40`,
`presence_penalty=2.0`; requests carrying an image use `temperature=0.7`,
`top_p=0.8`, `top_k=20`, `presence_penalty=1.5`. Both use
`repeat_penalty=1.0`. `--temperature` is available only as an explicit override.
The server launcher also limits the service to one slot, uses a 1,536-token
context and Q8 KV cache, and disables llama.cpp's default 8 GiB global prompt
cache. These bounded defaults are intentional for long-running use on a 16 GB
unified-memory Jetson; environment overrides are listed by
`scripts/run-qwen-vl-server --help`.

Role profiles make application scenes switchable without rebuilding or
restarting the model server. A versioned JSON `.role` file contains the system
prompt, bounded conversation settings, and separate text/vision sampling
fields. Bundled profiles can be inspected and selected by name:

```bash
scripts/run-qwen-voice-chat --list-roles

scripts/run-qwen-voice-chat \
  --model qwen3-vl-4b-q4 \
  --role snowball \
  --camera /dev/video0 \
  --vision-strategy see
```

Pass a file path to load an external profile, for example
`--role /path/to/custom.role`. Explicit command-line generation options take
precedence over the selected role, while the role takes precedence over
built-in defaults. Role files can only set validated prompt, conversation, and
sampling fields; protocol fields such as `model`, `messages`, and `stream`
cannot be injected.

Vision policies are interchangeable without changing the ASR, camera, queue,
conversation, or transport code:

| Strategy | Behavior |
|---|---|
| `see` | Reuse the first Qwen generation as the answer, or intercept `[SEE]` and attach a frame |
| `qwen` | Run the earlier separate `T`/`V`/`U` Qwen classification request before answering |
| `always` | Attach a frame to every utterance |
| `off` | Never start or use the camera |

Select one with `--vision-strategy see|qwen|always|off`. The camera keeps only
its newest JPEG in memory; text-routed questions never Base64-encode or evaluate
that frame. FFmpeg is required. Use `v4l2-ctl --list-devices` to find the capture
device.

Interaction recording is off by default. For controlled strategy comparisons,
pass `--vision-log /path/to/results.jsonl`. The private `0600` JSONL records the
selected strategy, route and latency, frame use, end-to-end response metrics,
and correction signals, but never stores images.

On Jetson, confirm that the server startup log reports `CUDA0`; a Vulkan build
works but is slower on the tested Orin NX. Ollama remains available as an
alternative backend:

```bash
# One-time import of a local GGUF model
bash scripts/import-ollama-gguf.sh /path/to/Qwen3-8B-Q5_K_M.gguf qwen3-8b-q5

scripts/run-qwen-voice-chat --provider ollama --model qwen3-8b-q5
```

The speech callback only enqueues events, so LLM inference never blocks the
subscriber. Say “清空对话” to reset the current history.

### ROS 2

ROS 2 is optional and does not affect the default DDS deployment. Prepare it
once, then select it exactly like DDS—without manually sourcing environments:

```bash
bash scripts/setup-ros2.sh
sudo g1-speech-service gpu ros2  # or: cpu ros2
g1-speech-service status
```

Recognition results are published to `/hri/speech/final`. In a ROS-sourced
terminal, inspect them with:

```bash
ros2 topic echo /hri/speech/final g1_speech_msgs/msg/SpeechEvent
```

The native lifecycle node supports both one-command startup and external
lifecycle orchestration:

```bash
ros2 launch g1_speech_ros2 speech_lifecycle.launch.py \
  config_file:="$PWD/config.json" autostart:=true

# Or let an external lifecycle manager control it:
ros2 launch g1_speech_ros2 speech_lifecycle.launch.py \
  config_file:="$PWD/config.json" autostart:=false
ros2 lifecycle set /g1_speech configure
ros2 lifecycle set /g1_speech activate
```

ROS 2 publishes `g1_speech_msgs/msg/SpeechEvent` and subscribes to
`g1_speech_msgs/msg/PlaybackState`, preserving the complete DDS event contract.
See [ROS 2 setup and lifecycle details](ros2/README.md).

## How It Works

```text
Microphone
  → 16 kHz mono audio
  → Silero VAD
  → SenseVoice-Small
  → DDS or ROS 2 transport
  → Robot / Agent / application

TTS or playback
  → rt/g1/hri/playback/state
  → temporarily pause recognition
```

| Topic | Direction | Purpose |
|---|---|---|
| `rt/g1/hri/speech/final` | Service → Agent | Final speech recognition events |
| `rt/g1/hri/playback/state` | Agent → Service | Pause recognition during playback so the robot does not hear itself |
| `/hri/speech/final` | Service → ROS 2 Agent | Final structured `SpeechEvent` |
| `/hri/playback/state` | ROS 2 Agent → Service | Structured playback gate state |

Each `SpeechEvent` includes a stable `event_id`, recognized text, language, audio duration, inference latency, source, and timestamp. The publisher retries after transient DDS failures, while subscribers deduplicate events by `event_id`.

## Deployment

### CPU INT8

CPU is the default backend. It is suitable for resource-constrained devices and Linux hosts without CUDA.

```bash
git clone https://github.com/star-nexus/G1-Robot-Agents-Speech.git
cd G1-Robot-Agents-Speech

cp deploy.env.example deploy.env
# Optionally set a DDS network interface. Leave the microphone empty to select
# one interactively during installation.
# DDS_NETWORK_INTERFACE="your-interface"

bash scripts/setup-cpu.sh
```

The script creates a project-local `.venv`, downloads the model, runs the doctor and fixed-WAV checks, and installs the CPU systemd service.

### Jetson GPU FP32

The GPU environment is completely isolated from the CPU environment and does not overwrite `.venv`, `config.json`, or the CPU INT8 model.

```bash
bash scripts/setup-jetson-gpu.sh
sudo g1-speech-service gpu
```

The GPU setup will:

1. Create a separate `.venv-gpu`
2. Build a sherpa-onnx CUDA wheel for Jetson aarch64
3. Install the SenseVoice FP32 model
4. Generate `config.gpu.json`
5. Install the mutually exclusive `g1-speech-gpu.service`

The current GPU build targets CUDA 12.6, sherpa-onnx 1.13.4, and ONNX Runtime 1.18.1. If the CUDA wheel is unavailable, the service fails explicitly instead of silently falling back to the CPU.

## Configuration

Copy `deploy.env.example` and enter your own device settings. No fixed IP address is required:

| Setting | Description |
|---|---|
| `DDS_NETWORK_INTERFACE` | Optional local interface used by DDS; leave empty for automatic selection |
| `MICROPHONE_DEVICE` | PortAudio input index or device name; leave empty for interactive selection |
| `DDS_DOMAIN_ID` | DDS domain shared with subscribers |
| `SPEECH_TOPIC` | Topic for final recognition results |
| `PLAYBACK_TOPIC` | Topic used to gate recognition during TTS or playback |
| `SENSEVOICE_THREADS` | Number of CPU inference threads |
| `SPEECH_TRANSPORT` | `dds` (default) or `ros2` |
| `ROS2_SETUP` | Optional ROS 2 `setup.bash`; normally discovered automatically |
| `ROS2_WORKSPACE_SETUP` | Optional custom workspace overlay path |
| `ROS2_DOMAIN_ID` | Optional ROS graph domain; setup preserves the current shell value |

Runtime settings are stored in `config.json`; GPU deployments use a separate `config.gpu.json`. Relative model paths are resolved from the directory containing the configuration file.

Generate a configuration directly from the application's canonical defaults:

```bash
g1-speech config init --output config.json
```

The setup scripts use the same command and apply only variables explicitly set in `deploy.env`. This keeps CPU, GPU, example, and programmatic defaults consistent.

## Validation and Testing

Check the CPU deployment:

```bash
.venv/bin/g1-speech doctor --config config.json --load-model
```

Check the GPU deployment:

```bash
.venv-gpu/bin/g1-speech doctor \
  --config config.gpu.json --load-model --skip-audio
```

Transcribe a WAV file:

```bash
.venv/bin/g1-speech transcribe --config config.json path/to/audio.wav
```

Reproduce the performance benchmark:

```bash
.venv/bin/python acceptance/benchmark_engine.py \
  --config config.json \
  --wav models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17/test_wavs/zh.wav

.venv-gpu/bin/python acceptance/benchmark_engine.py \
  --config config.gpu.json \
  --wav models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/test_wavs/zh.wav
```

Run the unit tests:

```bash
.venv/bin/python -m pytest
```

## Service Management

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

Underlying systemd units:

- CPU + DDS: `g1-speech.service`
- GPU + DDS: `g1-speech-gpu.service`
- CPU + ROS 2: `g1-speech-ros2.service`
- GPU + ROS 2: `g1-speech-gpu-ros2.service`

All four units conflict with each other, so only the selected mode can own the
microphone. The selector stops the previous mode and enables the new one at boot.

## References

This project builds on the following open-source work. Many thanks to their authors and communities:

- [SenseVoice](https://github.com/QwenAudio/SenseVoice): Multilingual speech recognition model
- [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx): SenseVoice ONNX inference and cross-platform deployment
- [Silero VAD](https://github.com/snakers4/silero-vad): Voice activity detection
- [Eclipse Cyclone DDS](https://github.com/eclipse-cyclonedds/cyclonedds): Vendor-neutral DDS transport
