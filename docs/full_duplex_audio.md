# Full-duplex robot audio and pluggable AEC

This project supports three explicit microphone-processing contracts:

| `AUDIO_PROCESSING_MODE` | Capture processing | Robot speech policy | Use when |
|---|---|---|---|
| `off` | None | Half duplex; recognition is gated during playback | Bring-up, headphones, or no echo path |
| `hardware` | Pass-through | Full duplex and VAD barge-in | The selected microphone/DSP already provides AEC, beamforming, and noise suppression |
| `webrtc` | Native WebRTC AudioProcessing 2.1 / AEC3, NS, optional AGC | Full duplex and VAD barge-in | Ordinary USB microphones without reliable hardware AEC |

`hardware` is an operator assertion, not automatic hardware detection. Do not run
software AEC after a microphone array's own AEC: the two adaptive filters can fight
each other and damage near-end speech. Conversely, selecting `hardware` for a plain
microphone leaves speaker echo untreated.

## Signal path

```text
Agent text chunks ──DDS/ROS 2──► sentence scheduler ─► vLLM-Omni WebSocket
                                                        │ 24 kHz mono PCM
                                                        ▼
                                              resample/volume/channel map
                                                        │
                              ┌─────────────────────────┴──────────────┐
                              ▼                                        ▼
                    ALSA hardware speaker                    AEC render reference
                                                                       │
USB microphone ─► ALSA hardware capture ─► 16 kHz mono ─► WebRTC APM ◄─┘
                                                            │
                                                            ▼
                                                      VAD ─► ASR
```

The render reference is the exact PCM after software volume, resampling, and channel
mapping, immediately before it is written to ALSA. PulseAudio monitor audio is not
used. Both capture and render are fed to APM in strict 10 ms frames; TTS playback uses
a persistent 10 ms ALSA output stream. When VAD detects near-end speech, the TTS
controller advances the Control Plane epoch, flushes the software PCM timeline,
cancels and clears TTS work, and publishes an inactive playback state. It does not
tear down the PortAudio stream during an ordinary interruption.

The persistent stream is a measured JP6.2 liveness requirement. Calling
`Pa_AbortStream` concurrently with a blocking `Pa_WriteStream` on the current USB
ALSA endpoint can leave the writer stuck even though abort returns and reports the
stream inactive. A software timeline generation is therefore checked immediately
before every hardware submission. PCM already submitted before invalidation is
bounded residual audio, not stale work newly admitted afterward. Its AEC reference
was already sent with that same speaker block. The `hard_abort` strategy remains an
explicit rollback and diagnostic mode.

The final write gate, residual classification, and every exported player metric are
defined in [`playback_metrics.md`](playback_metrics.md).

In `webrtc` mode, every signal that can reach the robot speaker must pass through
this player (or another component must feed the same samples into the APM render
bridge). Audio played independently by PulseAudio, a browser, or another process has
no render reference and therefore cannot be echo-cancelled reliably. System sounds
should be disabled or routed through the speech audio service in production.

## JetPack 6 setup

The Ubuntu 22.04 `libwebrtc-audio-processing1` package is the old 0.3 series and is
not this project's software-AEC runtime. Build the maintained AEC3 binding into each
Python environment that will run speech:

```bash
bash scripts/setup-webrtc-apm.sh .venv/bin/python
bash scripts/setup-webrtc-apm.sh .venv-gpu/bin/python
```

The script uses a temporary build environment, compiles bundled WebRTC
AudioProcessing 2.1 for ARM64, installs only the resulting Python wheel into the
selected runtime, and runs reverse/capture 10 ms smoke tests. It does not replace
PyTorch, CUDA, JetPack, or system Python.

For TTS, install the small client dependency in the speech runtime:

```bash
.venv-gpu/bin/python -m pip install websocket-client
```

The regular setup scripts perform both steps automatically when `deploy.env` selects
`AUDIO_PROCESSING_MODE="webrtc"` and `TTS_ENABLED=1`.

## Production configuration

First identify stable ALSA card IDs. Never copy `hw:1,0`; USB enumeration numbers can
change after reboot:

```bash
for path in /proc/asound/card*/id; do
  printf '%s: %s\n' "$path" "$(<"$path")"
done
```

Then configure this host in `deploy.env`:

```bash
AUDIO_INPUT_BACKEND="alsa"
AUDIO_INPUT_FALLBACK=""
ALSA_INPUT_CARD="your-microphone-card-id"
ALSA_OUTPUT_CARD="your-speaker-card-id"

AUDIO_PROCESSING_MODE="webrtc"
WEBRTC_AEC_STREAM_DELAY_MS=80
WEBRTC_NS_ENABLED=1
WEBRTC_AGC_ENABLED=0

TTS_ENABLED=1
TTS_WEBSOCKET_URL="ws://127.0.0.1:8091/v1/audio/speech/stream"
AUDIO_OUTPUT_INTERRUPT_STRATEGY="persistent"
```

Create a host-local model profile from the ASR profile you already use. This keeps
SenseVoice/Qwen3-ASR selection unchanged while assigning the exact model identifier
served by vLLM-Omni:

```bash
.venv-gpu/bin/g1-speech config init \
  --base config.gpu.json \
  --output config.full-duplex.local.json \
  --set tts.model=/absolute/path/to/Qwen3-TTS-12Hz-0.6B-CustomVoice \
  --set tts.task_type=CustomVoice \
  --force

# deploy.env
SPEECH_CONFIG_GPU="config.full-duplex.local.json"
```

For Qwen3-ASR, replace `--base config.gpu.json` with the existing
`config.qwen3-asr.local.json`. The ASR and TTS model paths remain profile data;
physical cards and the local server endpoint remain host deployment data.

Use `AUDIO_PROCESSING_MODE="hardware"` instead for a verified DSP microphone array.
Keep ALSA fail-closed in production so a busy or missing microphone does not silently
switch the robot to a laptop, camera, or PulseAudio default source. PulseAudio remains
an explicit diagnostic fallback only.

`WEBRTC_AEC_STREAM_DELAY_MS` describes the approximate delay from the render-reference
write to the corresponding echo at the microphone. Start at 80 ms, then tune on the
actual speaker, enclosure, USB devices, and power mode. A wrong value reduces AEC
convergence. Leave AGC disabled initially for a loud public environment; enable it
only after checking clipping, VAD stability, and ASR accuracy.

## Agent TTS contract

Agents stream text to `rt/g1/hri/tts/request` (ROS 2:
`hri/tts/request`) using `TtsTextChunk`:

- `request_id` identifies one answer;
- monotonically increasing `sequence` makes DDS retries idempotent;
- `text` may be an arbitrary LLM token fragment;
- `is_final` flushes the remaining sentence;
- `interrupt` cancels synthesis/playback and clears buffered text;
- `language`, `voice`, and `instructions` optionally override profile defaults.

The scheduler emits at strong punctuation, uses commas only after a useful prefix,
and caps unpunctuated buffers. Known style tags such as `[sad]` are converted into
Qwen instructions instead of spoken literally. A local end-to-end smoke request can
be published with:

```bash
.venv/bin/g1-speech speak --config config.gpu.json \
  --language English --voice Ryan "Hello! This is a streaming robot voice."
```

Playback state is published through the existing playback topic, so other robot
components see the same speaking state and request ID.

## Verification and acceptance

At startup, verify log lines report the intended physical ALSA input/output devices,
`audio_processing_mode=webrtc`, and nonzero render/capture frame counters. Production
acceptance should cover:

1. far-end-only speech at normal and maximum speaker volume;
2. near-end-only speech from several angles and distances;
3. double-talk, including interruption at the first word;
4. room reverberation, mechanical noise, motion noise, and USB replug/reboot;
5. ASR false accepts, missed barge-ins, clipped speech, and stop-to-silence latency.

A synthetic 30-second echo test on this Orin NX build achieved 17.54 dB ERLE over
the last five seconds with an 80 ms simulated delay. That proves native AEC3 is
active and converges on a simple path; it is not a substitute for the enclosure and
open-world tests above.
