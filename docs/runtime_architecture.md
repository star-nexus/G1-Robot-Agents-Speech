# STAR Runtime architecture

STAR Runtime is a modular monolith with an in-process data path as the preferred edge
composition. Microphone,
VAD, ASR, Agent orchestration, streaming TTS scheduling, playback gating, and robot
capability dispatch can share one process without serializing internal events.

The current llama.cpp LLM and vLLM-Omni TTS engines are still external inference
providers. Their adapters are boundaries of the Runtime, not a reason to turn every
internal stage into a service. DDS and ROS 2 are optional deployment adapters for a
second GPU, another machine, or a vendor ecosystem that already uses middleware.

```text
Role Package
  identity / prompt / memory / knowledge / capability policy
                         |
                         v
microphone/camera -> perception -> Agent Runtime -> action/audio commands
                            |              |
                            v              v
                    Role + capabilities  robot adapter
                            |
               +------------+-------------+
               |                          |
       in-process callbacks        DDS / ROS 2 adapters
       (default edge path)         (distributed path)
```

## Package ownership

- `star_runtime.agent`: role loading, model loop, memory, knowledge, identity, and
  the transport-independent voice bridge.
- `star_runtime.perception`: camera/sensor contracts, bounded latest-frame capture,
  and visual-turn routing. It does not own an Agent or a transport.
- `star_runtime.capabilities`: semantic capability schemas, deny-by-default policy,
  registration, invocation, and compatibility reports.
- `star_runtime.robots`: lightweight adapter manifests and lazy activation. A
  factory's metadata is safe to inspect without loading a vendor SDK.
- `star_runtime.speech`: audio, VAD, ASR, TTS, lifecycle, and the speech service.
- `star_runtime.core`: small canonical event objects shared by all subsystems.
- `star_runtime.transports`: transport-neutral contracts, the zero-serialization
  in-process path, and concrete DDS/ROS 2 adapters. Optional dependencies are lazy.
- `star_runtime.apps`: composition roots that choose concrete providers and transports.
- `star_runtime.cli`: the canonical command-line entrypoint.
- `g1_speech`: compatibility aliases for deployed callers; it owns no implementation.

## Embodiment activation

```text
Role Package declares required semantic capabilities
                         |
Runtime reads cheap RobotAdapterSpec metadata
                         |
permissions + missing capabilities are checked
                         |
exactly one compatible factory creates its vendor adapter
                         |
providers are registered atomically and the adapter starts
                         |
LLM/Agent may invoke capabilities through CapabilityRegistry
```

Compatibility is checked before `RobotAdapterFactory.create()`. Implementations
should keep vendor SDK imports inside `create()` or `RobotAdapter.start()` so unused
adapters do not occupy memory on Jetson Orin NX.

## Repository layout

```text
src/
├── star_runtime/
│   ├── agent/                 identity, role, memory, knowledge, LLM loop
│   ├── core/
│   │   └── events.py          canonical speech/TTS/playback events
│   ├── capabilities/
│   │   ├── contracts.py      semantic capability provider API
│   │   ├── registry.py       provider registration and invocation
│   │   └── policy.py         permission and compatibility checks
│   ├── robots/
│   │   ├── contracts.py      vendor-neutral adapter/factory contracts
│   │   └── catalog.py        lazy adapter discovery and activation
│   ├── perception/
│   │   ├── contracts.py      image-source and optional visual-turn ports
│   │   └── vision/
│   │       ├── camera.py     latest-frame V4L2 MJPEG capture
│   │       ├── routing.py    rule/semantic visual-turn selection
│   │       └── llama_cpp.py constrained llama.cpp route classifier
│   ├── speech/
│   │   ├── audio/            capture, playback, and processing
│   │   ├── asr/              ASR contracts/factories/backends
│   │   ├── tts/
│   │   │   ├── contracts.py  synthesis/audio provider ports
│   │   │   ├── scheduler.py  incremental text segmentation
│   │   │   ├── vllm_omni.py provider WebSocket adapter
│   │   │   └── controller.py playback lifecycle and barge-in
│   │   ├── settings/         validated models and JSON/env loading
│   │   ├── config.py         compatibility facade for settings
│   │   └── service.py        ASR/TTS application service
│   ├── transports/
│   │   ├── config.py         middleware-only configuration
│   │   ├── contracts.py      lifecycle, delivery, and voice ports
│   │   ├── inprocess.py      direct ASR ↔ Agent ↔ TTS callbacks
│   │   ├── dds/
│   │   │   ├── runtime.py    participant and low-level reader/writer
│   │   │   ├── types.py      wire IDL types
│   │   │   ├── codec.py      domain/wire conversion
│   │   │   ├── reliability.py bounded retry and deduplication
│   │   │   ├── speech.py     speech-service DDS adapter
│   │   │   └── voice.py      Agent ears/mouth DDS adapter
│   │   └── ros2/
│   │       ├── speech.py     speech-service ROS 2 adapter
│   │       └── voice.py      Agent ears/mouth ROS 2 adapter
│   ├── apps/
│   │   ├── integrated_runtime.py single-process Runtime owner
│   │   ├── speech_runtime.py distributed speech composition
│   │   └── local_voice_agent.py distributed Agent composition
│   └── cli/                  canonical command-line entrypoint
└── g1_speech/                backward-compatible import aliases

roles/                        versioned Soul/Role Packages
configs/examples/             portable model profiles
deploy/{systemd,profiles}/    deployment assets
integrations/ros2/            ROS 2 packages
scripts/                      stable operator entrypoints
tests/{,acceptance}/          automated and hardware acceptance tests
benchmarks/                   retained reproducible evidence
```

`transports/contracts.py` deliberately contains no implementation. Backend-specific
ears and mouth implementations live under `dds/` and `ros2/`; the local fast path is
named `inprocess.py` because it is a transport policy, not a voice domain module.

## Deployment modes

| Mode | Command | Internal event path | Intended use |
|---|---|---|---|
| Integrated | `star-runtime runtime ...` | direct callbacks | one Orin NX; lowest memory and latency |
| Distributed speech | `star-runtime serve ...` | DDS or ROS 2 | ASR/TTS on another GPU or machine |
| Distributed Agent | `star-runtime agent ...` | DDS or ROS 2 | independent Agent process or debugging |

The integrated path passes the same immutable `SpeechEvent` object to the Agent and
only allocates a `TtsTextChunk` at the Agent/TTS semantic boundary. It starts the Agent
consumer before microphone capture and stops capture before closing reasoning, which
prevents startup loss and shutdown races.

Visual input follows the same composition rule. The camera source retains one encoded
JPEG, and base64 conversion occurs only for a turn routed to vision. Completed
conversation memory stores the user's text rather than image data, so visual tokens and
base64 payloads do not accumulate across turns. The OpenAI-compatible model URL may be
loopback or a remote server; no Agent-core change is needed when the larger VLM runs on
a second Orin NX or server.

Benchmark code and raw evidence remain separate from runtime imports. Large future
artifacts should be published outside Git or under the ignored `artifacts/` tree.
