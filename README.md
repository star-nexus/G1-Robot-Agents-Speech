# G1 Speech Service — Offline ASR for Robot Agents

English | [简体中文](README_ZH.md)

Offline speech recognition for robot Agents, optimized for NVIDIA Jetson.
Give robots fast, private speech input for responsive, real-time interaction with people—without relying on the cloud.

**Tested on [Unitree G1](https://www.unitree.com/mobile/g1/) and
[Galbot G1](https://www.galbot.com/g1). Benchmarked on NVIDIA Jetson Orin NX.**
It also runs on other Jetson-powered robots and Linux edge computers such as NVIDIA DGX Spark.

- **CPU and GPU backends**: CPU INT8 and Jetson CUDA FP32 deployment modes
- **Fully offline**: Speech recognition runs locally—audio and text never need to leave the device
- **Robot-independent DDS interface**: Unified speech events are published to `rt/g1/hri/speech/final`
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

After deployment, select a backend to run as a background service:

```bash
# Low-resource, default option
sudo g1-speech-service cpu

# Jetson CUDA acceleration
sudo g1-speech-service gpu

g1-speech-service status
g1-speech-service logs
```

Recognition results are published to `rt/g1/hri/speech/final`. Both CPU and GPU backends use the same DDS messages, so no Agent-side changes are required.

### Subscribe from an Agent

```python
from g1_speech.dds import DdsSpeechSubscriber, initialize_unitree_dds

# Initialize only once per process. Skip this line if your Agent has already
# initialized Unitree DDS.
initialize_unitree_dds(domain_id=0)

subscriber = DdsSpeechSubscriber(lambda event: print(event.text))
subscriber.start()
```

Call `subscriber.close()` when the Agent exits. The callback runs on the DDS thread; in production Agents, enqueue the event in the callback and let a worker thread invoke the LLM, tools, or robot actions.

You can also inspect recognition results without writing Agent code:

```bash
.venv/bin/g1-speech listen --config config.json --timeout 0
```

## How It Works

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

| Topic | Direction | Purpose |
|---|---|---|
| `rt/g1/hri/speech/final` | Service → Agent | Final speech recognition events |
| `rt/g1/hri/playback/state` | Agent → Service | Pause recognition during playback so the robot does not hear itself |

Each `SpeechEvent` includes a stable `event_id`, recognized text, language, audio duration, inference latency, source, and timestamp. The publisher retries after transient DDS failures, while subscribers deduplicate events by `event_id`.

## Deployment

### CPU INT8

CPU is the default backend. It is suitable for resource-constrained devices and Linux hosts without CUDA.

```bash
git clone https://github.com/star-nexus/Unitree_G1_Voice.git
cd Unitree_G1_Voice

cp deploy.env.example deploy.env
# Set the DDS network interface for your machine. Leave the microphone empty
# to select one interactively during installation.
# ORIN_DDS_IFACE="your-interface"

bash scripts/setup-orin.sh
```

The script creates a project-local `.venv`, downloads the model, runs the doctor and fixed-WAV checks, and installs the CPU systemd service.

### Jetson GPU FP32

The GPU environment is completely isolated from the CPU environment and does not overwrite `.venv`, `config.json`, or the CPU INT8 model.

```bash
bash scripts/setup-orin-gpu.sh
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
| `ORIN_DDS_IFACE` | Local network interface used by DDS; inspect available interfaces with `ip -br link` |
| `ORIN_MIC_DEVICE` | PortAudio input index or device name; leave empty for interactive selection |
| `DDS_DOMAIN_ID` | DDS domain shared with subscribers |
| `SPEECH_TOPIC` | Topic for final recognition results |
| `PLAYBACK_TOPIC` | Topic used to gate recognition during TTS or playback |
| `SENSEVOICE_THREADS` | Number of CPU inference threads |

Runtime settings are stored in `config.json`; GPU deployments use a separate `config.gpu.json`. Relative model paths are resolved from the directory containing the configuration file.

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

sudo g1-speech-service restart
sudo g1-speech-service stop
```

Underlying systemd units:

- CPU: `g1-speech.service`
- GPU: `g1-speech-gpu.service`

## References

This project builds on the following open-source work. Many thanks to their authors and communities:

- [SenseVoice](https://github.com/QwenAudio/SenseVoice): Multilingual speech recognition model
- [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx): SenseVoice ONNX inference and cross-platform deployment
- [Silero VAD](https://github.com/snakers4/silero-vad): Voice activity detection
