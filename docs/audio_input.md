# Low-latency audio input on Jetson

The speech service treats ALSA hardware capture as the preferred robot path and
PulseAudio as an explicit fallback. This concerns microphone input only; TTS audio
output and interruption are outside this repository.

## Why the configuration does not contain `hw:1,0`

ALSA card numbers change when USB devices are re-enumerated. The service therefore
stores the stable card ID from `/proc/asound/card*/id`, then resolves that ID to the
current PortAudio hardware device at startup.

List IDs and hardware capabilities:

```bash
for path in /proc/asound/card*/id; do
  printf '%s: %s\n' "$path" "$(<"$path")"
done

arecord -l
cat /proc/asound/card1/stream0  # use the current number only for inspection
```

Example host policy:

```bash
# deploy.env
AUDIO_INPUT_BACKEND="alsa"
AUDIO_INPUT_FALLBACK="pulse"
ALSA_INPUT_CARD="Microphone"
ALSA_INPUT_DEVICE=0

# Native USB microphone format, not the ASR format.
ALSA_INPUT_SAMPLE_RATE=48000
ALSA_INPUT_CHANNELS=2
ALSA_INPUT_DTYPE="int16"

PULSE_INPUT_DEVICE="pulse"
# Optional: pin the Pulse source instead of following its default.
PULSE_SOURCE="alsa_input.usb-...analog-stereo"
AUDIO_INPUT_BLOCK_MS=20
AUDIO_INPUT_LATENCY="low"
```

The ASR/VAD pipeline always receives 16 kHz mono float32. When the hardware exposes
48 kHz stereo int16, capture callbacks enqueue the native block immediately. A
consumer thread performs stereo downmix and FIR low-pass decimation to 16 kHz, so
resampling work is not executed in PortAudio's real-time callback.

`AUDIO_INPUT_BLOCK_MS=20` gives 960 native frames at 48 kHz and 320 pipeline samples
at 16 kHz. Increase it only when logs show PortAudio overflow/underflow status.

## Fallback behavior

At every initial open or reconnect, the service tries the configured ALSA hardware
first. It falls back to PulseAudio only when all of the following are true:

- ALSA resolution or stream opening fails;
- `AUDIO_INPUT_FALLBACK="pulse"`;
- `PULSE_INPUT_DEVICE` is available.

When `PULSE_INPUT_DEVICE="pulse"`, set `PULSE_SOURCE` to a name from
`pactl list short sources` if the fallback must not follow PulseAudio's changing
desktop default.

The fallback is never silent. Startup logs identify:

```text
Preferred audio input alsa failed; falling back to pulse: ...
Microphone started: backend=pulse ...
```

To require hardware capture and fail closed instead, use:

```bash
AUDIO_INPUT_FALLBACK=""
```

To select PulseAudio deliberately:

```bash
AUDIO_INPUT_BACKEND="pulse"
AUDIO_INPUT_FALLBACK=""
PULSE_INPUT_DEVICE="pulse"
```

## PulseAudio ownership

ALSA `hw` access is normally exclusive. A dedicated microphone still managed by
PulseAudio can be unavailable to PortAudio even when its stable ALSA ID exists.
Diagnose ownership without changing the system:

```bash
pactl list short sources
fuser -v /dev/snd/*
g1-speech doctor --config config.json
```

For guaranteed direct capture, configure PulseAudio not to claim the dedicated
robot microphone. Keep a separate desktop/default input available if Pulse fallback
is required. The project does not kill PulseAudio, unload modules, or rewrite user
audio policy automatically because those actions affect unrelated desktop audio.

## Verification

After changing `deploy.env`:

```bash
sudo g1-speech-service restart
g1-speech-service logs
```

The definitive line is `Microphone started`. It reports the backend actually
opened, resolved hardware name, native capture format, 16 kHz pipeline format,
20 ms block size, and PortAudio's reported latency. Periodic metrics also include:

```text
audio_input_backend
audio_input_device
audio_input_latency_ms
```

Do not infer that ALSA is active merely because `AUDIO_INPUT_BACKEND=alsa` appears
in the deployment file. Check the runtime log or metrics.

## Current Orin NX observation

On the measured host, the Hollyland card ID is `Microphone`; it currently maps to
`hw:1,0` and exposes only 48 kHz, two-channel S16/S24 capture. PulseAudio currently
owns that PCM, so the service correctly reports ALSA unavailable and selects the
configured Pulse fallback until the dedicated card is released from PulseAudio.

An independent smoke test against an available USB camera microphone verified the
same ALSA path end to end: hardware capture opened directly, PortAudio reported
20.0 ms input latency, and every 48 kHz block produced 320 samples at 16 kHz.
