# STAR Runtime architecture

STAR Runtime is a modular monolith with an in-process data path as the preferred edge
composition. Microphone,
VAD, ASR, Agent orchestration, streaming TTS scheduling, playback gating, and robot
capability dispatch can share one process without serializing internal events.

The current llama.cpp LLM and vLLM-Omni TTS engines are still external inference
providers. Their adapters are boundaries of the Runtime, not a reason to turn every
internal stage into a service. DDS and ROS 2 are optional deployment adapters for a
second GPU, another machine, or a vendor ecosystem that already uses middleware.

## Runtime Control Plane v1

The speech-side `RuntimeControlPlane` is the single validity authority because VAD
is the first stage that observes barge-in. Its ownership model is:

- **Session**: one running robot interaction, created with `SpeechService` and
  stable across normal turns. The speech side owns `session_id`.
- **Turn**: one final user ASR result and the work derived from it. Final ASR starts
  a turn with a new `turn_id`; the event ID is reused as the turn ID to avoid a
  second identity allocation.
- **Epoch**: the monotonic generation within a turn. A turn starts at epoch 1.
  Barge-in or explicit cancellation atomically advances the active turn.

An immutable `ControlStamp(session_id, turn_id, epoch)` travels only at semantic
boundaries. A stamp is current iff it exactly equals the authority's active stamp.
As soon as an epoch advances or a newer turn starts, every older stamp is stale.
Provider cancellation improves latency and resource release, but this equality
check—not provider cooperation—is the correctness rule.

The integrated composition also shares a provider-neutral timing observer across
ASR, Agent, TTS, and PCM playback. It emits one structured record per terminal
exact stamp without changing control decisions or hot-path data. See
[`runtime_timing_audit.md`](runtime_timing_audit.md) for the observed stage and
hardware commit-point definitions.

```text
                         speech-side authority
                    Session S / current=(T2, E1)
                                  |
              +-------------------+-------------------+
              |                                       |
       Control Plane                             Data Plane
  speech-start / invalidation        ASR final [S,T,E] -> Agent
  provider cancel / queue drain      text delta [S,T,E] -> TTS
  playback start / stop / abort      audio request [S,T,E] -> PCM
  playback ack validation            future observations [S,T,E]
              |                                       |
              +-------- exact-stamp validity ---------+
```

The lifecycle state machine is deliberately small:

```text
NO_TURN -- final ASR --> T1/E1 -- normal completion --> T2/E1
                           |
                     barge-in/cancel
                           v
                        T1/E2
                  (all T1/E1 work stale)
                           |
                       final ASR
                           v
                        T2/E1

Any delayed delta, audio chunk, completion, or ack carrying T1/E1 is dropped.
The invalidated answer can neither resume playback nor mutate T2 state.
```

Validity is checked at five defense-in-depth points: Agent delta publication and
memory commit use the Agent's prompt-cancellation replica; authoritative checks run
when TTS text is accepted, when a synthesis request leaves the queue, before every
generated audio chunk enters the PCM ring, and before playback completion changes
state. `PlaybackGate` also binds its active edge to a stamp, so a late inactive ack
for an old response cannot deactivate a newer response.

On barge-in, the authority advances the epoch first, publishes one immutable
`EpochInvalidated` event, asks the Agent provider to cancel, resets the sentence
assembler, cancels active TTS, drains queued synthesis, and aborts the continuous
PCM ring timeline. The PortAudio stream remains active; the player advances its
local timeline generation and validates it immediately before hardware submission.
llama.cpp cancellation closes the active HTTP response; a provider that
cannot be stopped may continue computing, but all of its output remains stale and
is ignored. vLLM-Omni WebSocket cancellation may surface a local receive exception
while the socket closes; this is expected cancellation, and validity is still
independently checked.

The acoustic trigger is sherpa-onnx's early `is_speech_detected()` rising edge,
not ASR completion. Runtime startup fails if neither that API nor the legacy
`is_detected()` compatibility method exists; a missing method must never silently
disable barge-in. The synchronous preemption order is epoch invalidation, local
TTS/PCM cutoff, Agent cancellation publication, then TTS provider teardown. Local
cutoff advances the persistent player's generation and clears its ring before any
provider operation can block.

ASR Final is a second, later safety net for a missed acoustic edge. `begin_turn`
makes the replacement stamp authoritative first, then invokes output-only stale
cleanup before publishing the new SpeechEvent. That cleanup never calls control
invalidation, so it cannot advance or invalidate the newly created Turn. Exact
stamp checks reject concurrent old deltas during the cleanup window.

The in-process composition passes these immutable objects directly. DDS IDL and
ROS 2 messages preserve `session_id`, `turn_id`, and `epoch` on speech, TTS, and
playback data, plus a small control topic (`*/control/epoch`) for immediate Agent
cancellation. DDS and ROS 2 remain optional edge adapters, never mandatory internal
middleware.

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
│   │   ├── control.py         Session/Turn/Epoch authority and Agent replica
│   │   └── events.py          stamped speech/TTS/playback events
│   ├── capabilities/
│   │   ├── contracts.py      semantic capability provider API
│   │   ├── registry.py       provider registration and invocation
│   │   └── policy.py         permission and compatibility checks
│   ├── robots/
│   │   ├── contracts.py      vendor-neutral adapter/factory contracts
│   │   └── catalog.py        lazy adapter discovery and activation
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

## Validation and deferred work

Deterministic control-plane races live in `tests/test_runtime_control_plane.py`.
They cover Agent streaming, TTS generation, PCM playback, delayed old deltas and
completions, queue draining, and uninterrupted consecutive turns. Transport tests
cover in-process identity and DDS/ROS 2 metadata preservation. The unattended live
provider/ALSA check is:

```bash
TTS_ENABLED=1 AUDIO_PROCESSING_MODE=webrtc \
  AUDIO_INPUT_BACKEND=alsa ALSA_INPUT_CARD=Microphone \
  ALSA_OUTPUT_CARD=Audio AUDIO_OUTPUT_INTERRUPT_STRATEGY=persistent \
  SPEECH_TRANSPORT=inprocess PYTHONPATH=src \
  .venv/bin/python tests/acceptance/runtime_control_plane_live.py \
  --config config.tts-demo.local.json --force-cpu-asr \
  --role-package roles/tifa-lockhart
```

That unattended check injects final ASR events, so release acceptance must also run
one human acoustic sequence: ask a question, interrupt while PCM is audible, then
ask a follow-up. Logs must contain `Control epoch invalidated`, at least one
`Control stale work discarded`, and completion of the replacement turn, with no
resumption of the old answer.

Before its injected interruption sequence, the same harness performs a real
acoustic negative-control case: it plays a complete robot-only answer while the
production microphone/AEC/VAD path remains open. Keep the near-end area silent.
The harness requires both `control_invalidations` and
`audio_output_interruptions` to remain unchanged and emits:

```text
LIVE_MARKER robot-only speech produced zero false VAD barge-ins and zero playback interrupts
```

This uses the existing VAD thresholds; it is not a threshold-tuning experiment.

The current JP6.2 negative-control run remains **UNSIGNED / FAILED** with those
unchanged parameters. Two clean single-sentence runs produced a robot-echo VAD
edge about 3.17--3.26 seconds after first hardware audio and ASR decoded the
residual as `这不是。` / `不会。`. The diagnostic run processed 500 capture and
199 render-reference frames with zero WebRTC capture/render errors, so the AEC
path was active rather than missing. VAD-edge-to-local-output-cutoff remained
bounded at 0.823 ms and 1.509 ms; the failure is false acoustic triggering, not
Control Plane cutoff latency. No VAD threshold, AEC delay, Agent/TTS timing, or
sentence chunking parameter was changed in this P0 patch.

The synchronized render/raw/post-AEC follow-up found a stable physical echo delay
but a roughly 1.25-second hole in the WebRTC reverse-stream timebase during the
TTS provider gap. All bounded fixed-delay candidates still produced a robot-only
VAD edge. See [`acoustic_path_diagnostic.md`](acoustic_path_diagnostic.md) for the
captured levels, per-window correlations, 20 ms versus 100 ms timing comparison,
and fixed-delay A/B evidence.

### Playback recovery on JetPack 6.2

The v1.1 hardware failure was not an epoch-validity failure. On `KT USB Audio`
(`hw:1,0` during the test), the final hard-abort reproduction returned from abort
in 0.272 ms and the stream became inactive, while playback generation 0 remained
inside its already-running `Pa_WriteStream` for more than 3002 ms. Replacement PCM
was ready at +0.580 ms and enqueued at +0.705 ms but was never dequeued; restart
therefore never ran, and shutdown could not join the writer. An earlier reproduction
showed the same state, so the failure is repeatable rather than a visible-timeout
inference.

The selected strategy keeps one PortAudio/ALSA stream active for its full service
lifetime. Interruption advances a monotonic player generation and clears the ring.
A block is admitted to hardware only after the generation is rechecked. A write
that began before invalidation is explicitly counted as committed residual audio;
its completion cannot authorize another old block. This preserves the AEC render
reference because it is still fed only for blocks submitted to the speaker path.

Recovery observability spans `PLAYBACK_RECOVERY_PCM_READY`,
`PLAYBACK_RECOVERY_ENQUEUE`, `PLAYBACK_RECOVERY_DEQUEUE`,
`PLAYBACK_HW_WRITE_BEGIN`, and `PLAYBACK_RECOVERED`. Metrics expose stream state
transitions, abort/restart counts, underflows, write/restart errors, residual writes,
stale pre-write drops, forbidden stale write attempts, and recovery latency. Exact
definitions and update paths for every player metric are listed in
[`playback_metrics.md`](playback_metrics.md). The isolated hardware A/B is
reproducible with:

```bash
PYTHONPATH=src .venv/bin/python \
  tests/acceptance/playback_recovery_stress.py \
  --card Audio --strategy hard_abort --cycles 3 --timeout 3

PYTHONPATH=src .venv/bin/python \
  tests/acceptance/playback_recovery_stress.py \
  --card Audio --strategy persistent --cycles 50 --timeout 3
```

On L4T R36.4.3, hard abort stalled on cycle 1 with no recovery inside three
seconds. Persistent mode completed 50/50 cycles with zero stalls, stale hardware
submissions, underflows, write/restart errors, or inactive-stream transitions.
The final commit-audit rerun measured interrupt-to-successful-write latency at
18.459 ms mean, 18.821 ms p95, and 19.051 ms maximum. The nominal residual bound
is one already-committed 10 ms
software block plus the stream/device's reported 10 ms latency; it is physically
different from allowing stale PCM into the ring after invalidation.

The final live llama.cpp/vLLM-Omni/WebRTC/ALSA harness additionally recorded
replacement TTS PCM at +2084.549 ms, enqueue at +2084.911 ms, dequeue at
+2085.496 ms, and successful write at +2086.123 ms. Thus provider/Agent/TTS time
dominated; playback added 1.574 ms from PCM-ready to successful write. It also
completed an uninterrupted third turn with 741 render-reference frames and no AEC,
write, restart, or underflow errors.

A provider-only three-run check retained normal streaming behavior: First PCM was
270.620 ms versus the retained 249.681 ms baseline (+8.4%), while audio RTF was
0.684 versus 0.694 (-1.3%). The player is not on that benchmark path, and neither
TTS model code nor its request path changed. The remaining human acoustic VAD
sequence is intentionally not signed by the unattended harness.

Run that final sequence with a person at the robot:

```bash
TTS_ENABLED=1 AUDIO_PROCESSING_MODE=webrtc \
  AUDIO_INPUT_BACKEND=alsa ALSA_INPUT_CARD=Microphone \
  ALSA_OUTPUT_CARD=Audio AUDIO_OUTPUT_INTERRUPT_STRATEGY=persistent \
  SPEECH_TRANSPORT=inprocess PYTHONPATH=src \
  .venv/bin/g1-speech runtime \
  --config config.tts-demo.local.json \
  --role-package roles/tifa-lockhart \
  2>&1 | tee /tmp/runtime-control-human-barge-in.log
```

Ask a long question, wait for audible robot PCM, speak a follow-up over it, then
wait for the follow-up answer. The log must show the causal sequence `Control epoch
invalidated` → `PLAYBACK_INTERRUPT` → `PLAYBACK_RECOVERY_PCM_READY` →
`PLAYBACK_RECOVERY_ENQUEUE` →
`PLAYBACK_RECOVERY_DEQUEUE`, `PLAYBACK_HW_WRITE_BEGIN`, and
`PLAYBACK_RECOVERED`, followed by replacement-turn completion. Delayed `Control
stale work discarded` markers may appear between those milestones. A human must also
confirm that the old answer did not resume. Until that observation is recorded,
**JP6.2 human acoustic release acceptance is UNSIGNED**.

Control Plane v1 intentionally defers inference scheduling, GPU/model residency,
unified memory management, admission control, and latency/resource measurement.
Those belong to Runtime Measurement v1 and the later inference resource manager;
no llama.cpp or vLLM-Omni migration is required by this design.

Benchmark code and raw evidence remain separate from runtime imports. Large future
artifacts should be published outside Git or under the ignored `artifacts/` tree.
