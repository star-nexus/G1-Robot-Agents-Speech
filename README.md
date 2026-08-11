# G1 Speech Service — Offline ASR for Robot Agents

English | [简体中文](README_ZH.md)

Pluggable offline speech recognition for robot Agents, optimized for NVIDIA Jetson.
Give robots fast, private speech input for responsive, real-time interaction with people—without relying on the cloud.

The project has no robot-vendor SDK dependency. It integrates through standard audio devices,
DDS, or ROS 2. NVIDIA Jetson Orin NX is the primary benchmark platform; other Jetson robot
computers, NVIDIA DGX Spark, and general Linux edge systems are also supported targets.
`G1` is retained only as the existing project and protocol namespace; it does not bind the
service to any robot brand with that name.

- **CPU and GPU backends**: CPU INT8 and Jetson CUDA FP32 deployment modes
- **Pluggable ASR models**: built-in SenseVoice and Qwen3-ASR adapters keep transport consumers unchanged
- **Fully offline**: Speech recognition runs locally—audio and text never need to leave the device
- **Resilient audio capture**: Stream heartbeat, automatic microphone reconnection, and bounded queues
- **DDS and ROS 2 transports**: Native structured topics for lightweight DDS systems and ROS 2 robots
- **Measured performance**: backend-specific latency and RTF benchmarks on Jetson Orin NX

## Verified Platforms

| Platform | Compute | Available backends | Validation |
|---|---|---|---|
| Jetson Orin NX edge systems | Jetson Linux | CPU INT8 / CUDA FP32 | ✅ Benchmark platform |
| NVIDIA DGX Spark | ARM64 Linux | CPU INT8 | ✅ Original deployment |
| Other Jetson/Linux robots | Jetson or Linux edge computer | CPU INT8; CUDA FP32 on Jetson | Compatibility target |

Validation refers to configurations tested by this project; vendor hardware configurations may vary.

## Performance

Measured on a Jetson Orin NX in `MAXN_SUPER` mode (dynamic clocks, not locked), using
the same 5.592-second Chinese WAV, 3 warm-ups, and 10 measured runs:

| Model | Device | Precision | Attention | Mean | Median | P95 | RTF |
|---|---|---|---|---:|---:|---:|---:|
| SenseVoice-Small | CPU | INT8 | N/A | 205.9 ms | 206.0 ms | 206.4 ms | **0.0368** |
| SenseVoice-Small | GPU | FP32 | N/A | 77.1 ms | 78.0 ms | 97.7 ms | **0.0138** |
| Qwen3-ASR-0.6B | CPU | FP32 | eager | 8489.4 ms | 8469.6 ms | 9537.6 ms | 1.5181 |
| Qwen3-ASR-0.6B | CPU | FP32 | SDPA | 17045.2 ms | 16967.9 ms | 19567.6 ms | 3.0481 |
| Qwen3-ASR-0.6B | CPU | — | FA2 | unsupported (CUDA only) | — | — | — |
| Qwen3-ASR-0.6B | GPU | BF16 | eager | 1378.2 ms | 1375.9 ms | 1391.5 ms | 0.2465 |
| Qwen3-ASR-0.6B | GPU | BF16 | SDPA | **1235.7 ms** | 1234.8 ms | 1241.7 ms | **0.2210** |
| Qwen3-ASR-0.6B | GPU | FP16 | SDPA | **1218.3 ms** | 1216.0 ms | 1244.7 ms | **0.2179** |
| Qwen3-ASR-0.6B | GPU | BF16 | FA2 | 1493.4 ms | 1494.3 ms | 1500.2 ms | 0.2671 |

Attention selection does not apply to SenseVoice's ONNX graph. On this Orin and
single-request workload, GPU SDPA is 10.3% faster than eager; FA2 is 20.9% slower
than SDPA. CPU SDPA is also slower because the installed Jetson PyTorch lacks native
GQA and the compatibility path explicitly expands KV heads. These are end-to-end
Transformers adapter measurements, not server-GPU vLLM throughput or an accuracy
benchmark.

### Why is Qwen3-ASR slower on Jetson Orin NX?

The latency gap should not be attributed primarily to the attention backend.
Qwen3-ASR is a large audio-language model that combines an audio encoder with an
autoregressive Qwen3 decoder. On Jetson Orin NX, single-request decoding is dominated
by repeated decoder execution, GEMM work, memory traffic, and kernel-launch overhead
rather than attention alone.

Our measurements support this interpretation. Switching from eager attention to SDPA
reduces Qwen3-ASR-0.6B latency by 10.3%, while FlashAttention 2 is 20.9% slower than
SDPA on the same short, batch-1 workload. A synchronized stage breakdown attributes
98.8% of the 1.28-second request to `generate()`. Nsight Systems records about 26,000
CUDA kernel launches in one 15-token request: GEMM kernels account for about 65% of
GPU kernel time, while attention kernels account for about 2.3%.

The official efficiency results are not directly comparable. They use approximately
two-minute audio, vLLM 0.14.0, CUDA Graphs, BF16, and server-oriented batch or
asynchronous serving. This project measures a 5.592-second utterance through native
Transformers `generate()` at batch 1. Consequently, GPU SDPA + FP16 is the sealed
Orin NX baseline. A 10-utterance, 50.188-second labeled check found identical
punctuation-insensitive content CER for BF16 and FP16 (6.22%), 10/10 matching content
transcripts, deterministic output over three runs per utterance, and 100% language
identification. FP16 also provides a measured 4.9% fixed-sample latency reduction.
The sealed service environment still has no Triton and remains unchanged. A later
isolated Jetson-native PyTorch 2.9.1 / Triton 3.5.1 experiment made compiled decode
measurable: Transformers static-cache generation reached 451.6 ms and 34.3 tokens/s
of request-level apparent generation throughput, while decoder-only bitsandbytes NF4
was slower than FP16. A controlled token-scaling regression estimates steady-state
incremental decode throughput at 35.5 tokens/s; the earlier 15.3 tokens/s live value
was depressed by fixed encoder/prefill cost and first-new-shape graph warm-up, not a
decoder regression.

See the [full Orin NX latency report](docs/qwen3_asr_orin_nx_latency.md), the
[raw benchmark data](benchmarks/orin_nx_2026-08-09/README.md), the
[sealed FP16 baseline](benchmarks/orin_nx_2026-08-10_fp16_baseline/README.md), and the
[runtime optimization follow-up](docs/qwen3_asr_orin_nx_runtime_optimizations.md),
the [English runtime optimization report](docs/qwen3_asr_orin_nx_runtime_optimization_report_en.md),
the [controlled E2E runtime-gap attribution](docs/qwen3_asr_e2e_runtime_gap.md),
the [JetPack 6 vLLM feasibility note](docs/vllm_orin_jp6.md), and the
[end-to-end latency instrumentation and streaming plan](docs/speech_end_to_end_latency.md).

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

### Select SenseVoice or Qwen3-ASR

The pipeline depends only on the `AsrEngine` contract. Prepare a host-local Qwen config,
validate the CUDA load, and select it for the existing GPU service:

```bash
bash scripts/setup-qwen3-asr.sh
cp config.qwen3-asr.example.json config.qwen3-asr.local.json
# Edit model_dir first.
.venv-gpu/bin/g1-speech doctor \
  --config config.qwen3-asr.local.json --load-model --skip-audio

# deploy.env
SPEECH_CONFIG_GPU="config.qwen3-asr.local.json"
# Restart the selected service, or select GPU + DDS when currently stopped.
sudo g1-speech-service restart
# sudo g1-speech-service gpu dds
```

`SPEECH_CONFIG_GPU` is the single deployment-time model selector. The JSON
profile owns `asr.backend`, checkpoint, dtype, attention, and cache settings;
`deploy.env` does not duplicate them. To return to SenseVoice GPU, select
`config.gpu.json` and restart. See the [configuration model](docs/configuration.md).

SDPA is the default attention backend. On the Jetson PyTorch 2.5 build, which lacks
the `enable_gqa` argument, the adapter expands KV heads and then uses PyTorch SDPA.
The Qwen example selects FP16 because it passed the Orin NX CER gate; BF16 remains
the conservative fallback for other CUDA platforms until measured there.
Set `attention_implementation` to `eager` only for fallback or baseline reproduction.
FlashAttention 2 remains available for controlled experiments, but it is not a
deployment profile: it was 20.9% slower than SDPA on the measured Orin batch-1
workload. Experimental variants and their exact settings belong with benchmark
artifacts rather than as additional root-level `config.qwen3-*` files.
Recognition logs include audio duration and RTF. Use `acceptance/benchmark_engine.py`
for fixed-WAV latency benchmarks and `acceptance/compare_asr.py` for labeled-corpus CER.

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
  → selected ASR adapter (SenseVoice / Qwen3-ASR / external plugin)
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

Configuration has two explicit layers:

1. A JSON model profile selects SenseVoice or Qwen3-ASR and owns every
   model-specific setting.
2. `deploy.env` describes this host and owns the selected profile path, Python
   runtime, microphone, common VAD/playback behavior, transport, and installer.

Copy `deploy.env.example`; it has the same fields and ordering as a real
`deploy.env`, with host-neutral values:

| Setting | Description |
|---|---|
| `SPEECH_CONFIG_CPU` / `SPEECH_CONFIG_GPU` | JSON profile selected by each service mode |
| `SPEECH_PYTHON_GPU` | Optional isolated Python used by the GPU service |
| `SPEECH_GPU_LIBRARY_PATH` | Optional colon-separated native-library paths for that runtime |
| `DDS_NETWORK_INTERFACE` | Optional local interface used by DDS; leave empty for automatic selection |
| `MICROPHONE_DEVICE` | PortAudio input index or device name; leave empty for interactive selection |
| `DDS_DOMAIN_ID` | DDS domain shared with subscribers |
| `SPEECH_TOPIC` | Topic for final recognition results |
| `PLAYBACK_TOPIC` | Topic used to gate recognition during TTS or playback |
| `VAD_*` | Shared utterance segmentation applied to every ASR backend |
| `PLAYBACK_*` | Shared recognition gate applied to every ASR backend |
| `SPEECH_TRANSPORT` | `dds` (default) or `ros2` |
| `ROS2_SETUP` | Optional ROS 2 `setup.bash`; normally discovered automatically |
| `ROS2_WORKSPACE_SETUP` | Optional custom workspace overlay path |
| `ROS2_DOMAIN_ID` | Optional ROS graph domain; setup preserves the current shell value |

Do not put `ASR_BACKEND`, `SENSEVOICE_*`, or `QWEN3_ASR_*` in `deploy.env`.
Those values belong to the selected JSON profile. Backend-neutral environment
settings are applied both during configuration generation and when the service
starts, so they consistently affect SenseVoice and Qwen3-ASR. Relative model paths
are resolved from the directory containing the JSON profile.

Generate a configuration directly from the application's canonical defaults:

```bash
g1-speech config init --output config.json
```

The setup scripts use the same command and apply only backend-neutral variables
explicitly set in `deploy.env`. Full ownership and precedence are documented in
[`docs/configuration.md`](docs/configuration.md).

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
