# Low-latency audio input on Jetson

Robot production deployments use direct ALSA hardware capture. PulseAudio remains
available for the desktop and for explicit diagnostics, but the speech service does
not silently switch microphones when its selected hardware input is unavailable.
Microphone ownership is independent of the optional full-duplex AEC/TTS path. See
[`full_duplex_audio.md`](full_duplex_audio.md) for output, render reference, and
barge-in behavior.

## Production policy

Use this baseline in `deploy.env`:

```bash
AUDIO_INPUT_BACKEND="alsa"
AUDIO_INPUT_FALLBACK=""

# Set this to an ID printed by /proc/asound/card*/id on the target robot.
ALSA_INPUT_CARD="your-card-id"
ALSA_INPUT_DEVICE=0

# The microphone's native format, not the ASR format.
ALSA_INPUT_SAMPLE_RATE=48000
ALSA_INPUT_CHANNELS=2
ALSA_INPUT_DTYPE="int16"

AUDIO_INPUT_BLOCK_MS=20
AUDIO_INPUT_LATENCY="low"
```

Failing closed is intentional. A disconnected or busy robot microphone should
produce an actionable error instead of silently changing to a webcam, monitor, or
other PulseAudio default source.

## Selecting a microphone

The project never hard-codes a microphone manufacturer and does not infer quality
from USB vendor names. ALSA and USB metadata provide capabilities, not a reliable
measure of acoustic quality. A production robot should therefore select its tested
microphone explicitly.

List the stable ALSA card IDs and the corresponding capture hardware:

```bash
for path in /proc/asound/card*/id; do
  printf '%s: %s\n' "$path" "$(<"$path")"
done

arecord -l
```

Copy the selected ID into `ALSA_INPUT_CARD`. Do not store `hw:1,0`: numeric ALSA
card indices can change whenever USB devices are re-enumerated. At startup and after
a reconnect, the service resolves the configured stable ID to the current PortAudio
`hw:N,M` device.

During interactive installation, `scripts/setup-cpu.sh` lists the available card
IDs and asks the user to select one. `ALSA_INPUT_CARD=""` enables conservative
auto-detection, but it succeeds only when exactly one ALSA hardware input matches
the configured PCM device. Multiple candidates are treated as an error rather than
guessed. This makes empty-card auto-detection convenient on simple hosts without
making a robot's microphone choice unpredictable.

Inspect the selected device's native formats before setting sample rate, channels,
and dtype:

```bash
arecord -l
cat /proc/asound/cardN/stream0  # N is used only for inspection
```

The ASR/VAD pipeline always receives 16 kHz mono float32. When the hardware exposes
48 kHz stereo int16, capture callbacks enqueue the native block immediately. A
consumer thread performs stereo downmix and FIR low-pass decimation to 16 kHz, so
resampling work is not executed in PortAudio's real-time callback.

`AUDIO_INPUT_BLOCK_MS=20` gives 960 native frames at 48 kHz and 320 pipeline samples
at 16 kHz. Increase it only when logs show PortAudio overflow or underflow status.

## Reserving the selected microphone for ALSA

ALSA `hw` access is normally exclusive. PulseAudio can discover a USB microphone
and later reopen its PCM even if its source currently says `SUSPENDED`. For a
dedicated robot microphone, configure PulseAudio to ignore the selected card while
continuing to manage all other desktop audio devices.

First resolve the selected ALSA ID to its current sysfs card and inspect its stable
USB attributes:

```bash
ALSA_CARD_ID="your-card-id"

for path in /proc/asound/card*/id; do
  if [ "$(<"$path")" = "$ALSA_CARD_ID" ]; then
    card_name="$(basename "$(dirname "$path")")"
    udevadm info --attribute-walk --path="/sys/class/sound/$card_name"
  fi
done
```

From the USB device parent, record `idVendor`, `idProduct`, and `serial`. Then create
a host-local rule using those values:

```udev
# /etc/udev/rules.d/91-g1-speech-pulseaudio-ignore.rules
# Replace the placeholders with attributes from the selected microphone.
ACTION=="add|change", SUBSYSTEM=="sound", KERNEL=="card*", SUBSYSTEMS=="usb", ATTRS{idVendor}=="vvvv", ATTRS{idProduct}=="pppp", ATTRS{serial}=="device-serial", ENV{PULSE_IGNORE}="1"
```

This is a user-selected device policy, not a vendor-specific project default. If a
device exposes no serial number, omit the serial match only after accepting that all
devices with the same vendor/product IDs will be reserved. Do not edit the packaged
`/usr/lib/udev/rules.d/90-pulseaudio.rules`; package updates overwrite it.

Load the host-local rule, stop speech capture, and unplug/replug the selected USB
receiver so PulseAudio processes a fresh device event:

```bash
sudo udevadm control --reload-rules
```

If a stale PulseAudio card remains after replugging, restart only the current user's
PulseAudio instance:

```bash
systemctl --user restart pulseaudio.service pulseaudio.socket
```

Verify that the selected microphone is absent from PulseAudio but still present in
ALSA:

```bash
pactl list short cards
pactl list short sources
arecord -l
sudo fuser -v /dev/snd/*
```

Before the speech service starts, the selected capture PCM should have no owner.
After startup, the speech process should own it. Other PulseAudio cards and sources
should remain available.

Permanent `PULSE_IGNORE` and Pulse fallback through the same microphone are mutually
exclusive. The production configuration therefore uses
`AUDIO_INPUT_FALLBACK=""`. A deployment that deliberately needs fallback must use a
different PulseAudio microphone and pin that source explicitly.

## Understanding `SUSPENDED`

`SUSPENDED` in `pactl list short sources` means PulseAudio still has a logical
source for the card but has suspended it. With `module-suspend-on-idle`, the ALSA PCM
is normally released while suspended. It does **not** by itself prove `Device busy`
and it does not prove which backend the speech service opened.

Use ownership and runtime evidence instead:

```bash
sudo fuser -v /dev/snd/*
g1-speech-service logs
```

The definitive startup line is:

```text
Microphone started: backend=alsa device='... (hw:N,M)' ...
```

Periodic metrics must likewise report:

```text
"audio_input_backend": "alsa"
```

If ALSA opening fails and Pulse fallback was explicitly enabled, the transition is
never silent:

```text
Preferred audio input alsa failed; falling back to pulse: ...
Microphone started: backend=pulse ...
```

In the measured ALSA Direct runs, the selected Pulse source was `SUSPENDED`, the PCM
had no PulseAudio owner before startup, and both the startup log and metrics reported
`backend=alsa`. Those runs therefore used ALSA Direct; they did not take the
PulseAudio fallback path.

## Explicit PulseAudio mode

PulseAudio remains available for desktop testing or deployments that intentionally
prefer it:

```bash
AUDIO_INPUT_BACKEND="pulse"
AUDIO_INPUT_FALLBACK=""
PULSE_INPUT_DEVICE="pulse"

# Optional: pin a source rather than following the desktop default.
PULSE_SOURCE="alsa_input.usb-...analog-stereo"
```

Do not enable this mode for a card hidden by `PULSE_IGNORE`.

## Final verification

After changing `deploy.env` or the host audio policy:

```bash
g1-speech doctor --config config.json
sudo g1-speech-service restart
g1-speech-service logs
```

Confirm the resolved device, native capture format, 16 kHz mono pipeline format,
block size, and PortAudio latency in the `Microphone started` line. Do not infer
that ALSA is active merely because `AUDIO_INPUT_BACKEND="alsa"` appears in
`deploy.env`; always verify the runtime log or metrics.
