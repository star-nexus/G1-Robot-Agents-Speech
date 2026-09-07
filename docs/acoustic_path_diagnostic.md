# JP6.2 robot-only acoustic path diagnostic

This diagnostic was run after enabling the correct sherpa VAD speech edge. It
does not change VAD threshold, output volume, TTS text/model, playback strategy,
or Control Plane semantics. The opt-in harness captures three 16 kHz streams on
one host monotonic clock:

- `render_reference.wav`: mono PCM presented to WebRTC reverse processing at the
  final hardware-write boundary;
- `raw_microphone.wav`: microphone PCM after hardware-format conversion but before
  WebRTC APM;
- `post_aec.wav`: the same capture block after WebRTC APM;
- `synchronized_streams.npz`: samples, validity masks, block timestamps, and VAD
  edge timestamps;
- `analysis.json`: windowed cross-correlation, continuity, level, and edge results.

The capture timestamp is the PortAudio callback endpoint. Raw and post-AEC blocks
share that exact timestamp. Render timestamps are taken immediately before the
matching PortAudio write and AEC render-reference submission.

## Why the first acceptance run used 100 ms capture

`config.tts-demo.local.json` contains `audio.block_ms=100`. The direct Python live
harness loaded that JSON and did not source the host `deploy.env`, which contains
`AUDIO_INPUT_BLOCK_MS=20`. The previous production human logs confirm 20 ms blocks
and approximately 20 ms input latency; the first negative-control run instead
reported 100 ms blocks and approximately 100 ms latency.

The live acceptance harness now defaults explicitly to `--audio-block-ms 20`.
Both timings were measured without changing VAD parameters:

| Capture timing | Measured echo delay | Delay span | Robot-only VAD edges |
|---|---:|---:|---:|
| 20 ms block / 20 ms latency | about 97--102 ms | 2.1--6.3 ms in runs with enough correlated windows | 1 |
| 100 ms block / 100 ms latency | 97.375 ms | 0.687 ms | 1 |

The 100 ms capture setting changes scheduling granularity but did not create the
physical render-to-echo delay or eliminate the false edge.

## Fixed-delay A/B

Production 20 ms capture timing was held constant while sweeping a bounded range
around the measured echo delay. A delay window is accepted only when absolute
normalized render/raw correlation is at least 0.5. Fewer than three accepted
windows are reported as insufficient for an individual drift conclusion.

| Configured `stream_delay_ms` | Correlated delay median | Accepted span | Accepted windows | Post-AEC RMS at VAD edge | VAD edges |
|---:|---:|---:|---:|---:|---:|
| 60 | 99.344 ms | 4.000 ms | 4 | -47.481 dBFS | 1 |
| 80 | 101.906 ms | 6.312 ms | 2 | -36.798 dBFS | 1 |
| 100 | 97.219 ms | 5.187 ms | 2 | -44.979 dBFS | 1 |
| 120 | 96.906 ms | 2.124 ms | 6 | -68.171 dBFS | 1 |
| 140 | 106.250 ms | not individually assessable | 1 | -32.311 dBFS | 1 |

The raw robot echo in accepted windows was typically around -38 to -43 dBFS.
WebRTC AEC often attenuated the middle of the utterance by tens of decibels, but
the post-AEC residual rose again near every false VAD edge. No delay candidate
achieved the required `robot-only speech -> zero VAD edges`, so none was persisted
and no candidate advanced to the human near-end acceptance step.

## Delay stability versus reference continuity

The physical render-to-microphone delay did not materially drift. Runs with at
least three trustworthy windows stayed within 2.1--4.0 ms; the 100 ms timing
control stayed within 0.7 ms. The failure is instead strongly associated with a
discontinuous WebRTC reverse-stream timeline:

| Delay run | Render blocks | Largest render interval | Render gap excess | Capture median/max interval |
|---:|---:|---:|---:|---:|
| 60 | 459 | 1241.729 ms | 1253.100 ms | 19.999 / 22.056 ms |
| 80 | 257 | 1245.281 ms | 1294.437 ms | 19.998 / 22.299 ms |
| 100 | 152 | 1268.682 ms | 1266.421 ms | 19.998 / 20.444 ms |
| 120 | 288 | 1231.754 ms | 1237.128 ms | 19.997 / 20.633 ms |
| 140 | 219 | 1245.101 ms | 1246.472 ms | 20.000 / 20.477 ms |

Each TTS run submitted roughly 30 ms of initial PCM, then had a 1.23--1.27 second
provider gap before render writes resumed. Capture/APM processing continued at its
normal 10 ms frame cadence, but `process_reverse_stream()` is currently called
only for actual PCM hardware writes. It therefore received no silence frames for
about 123--127 capture/APM frames. This compresses the far-end reference timebase
inside AEC even though the external acoustic delay remains stable.

This evidence explains why a fixed `stream_delay_ms` sweep cannot meet the gate:
the missing reverse frames are approximately 1.25 seconds, larger than both the
measured 98 ms propagation delay and the configured delay range. The data supports
reverse-stream discontinuity as the next hypothesis to test, rather than VAD
threshold or a drifting room/device delay. No continuous-silence/reference design
change is included in this diagnostic task.

## Continuous render-clock experiment

The persistent speaker writer was then made the sole WebRTC render-clock
authority. In WebRTC/persistent mode it submits one 10 ms hardware block on every
speaker-clock tick: queued PCM when available, otherwise an explicit all-zero
block. The exact mono signal derived from every submitted speaker block is passed
to WebRTC reverse processing. Clock-only silence is unscoped Control Plane data;
it is excluded from queued-PCM, stale-block, and committed-residual counters.

An unchanged JP6.2 robot-only run used 20 ms microphone capture, 10 ms speaker
blocks, `stream_delay_ms=80`, and the same VAD, volume, TTS, and persistent
interruption settings. The synchronized frozen artifact is retained at
`artifacts/acoustic/continuous-render-delay80-20260822-frozen` (the directory is
repository-ignored).

| Result | Continuous render-clock run |
|---|---:|
| Render interval median / maximum | 9.998 / 11.601 ms |
| Render gaps over 15 ms | 0 |
| Former provider-gap interval | eliminated |
| Physical echo-delay median / span | 101.000 / 0.062 ms |
| Accepted correlation windows | 3 |
| Median raw / post-AEC level | -50.783 / -68.510 dBFS |
| Median AEC attenuation | 18.867 dB |
| VAD edges / Control invalidations | 1 / 1 |

The remaining false edge occurred at 3206.250 ms from audible render start,
approximately the utterance tail. Its local post-AEC level was -60.857 dBFS with
a 0.004242 peak. Accepted-window attenuation weakened from 45.508 dB to
18.867 dB and then 9.792 dB toward that tail. WebRTC processed 850 render frames
and 372 capture frames with zero processing errors; PortAudio reported zero
abort calls, write errors, underflows, stale pre-write drops, or forbidden stale
write attempts.

This falsifies the hypothesis that the 1.25-second missing reverse-stream interval
was the only cause of false robot-only VAD. The continuous clock fixed the timebase
but did **not** meet the `robot-only speech -> zero VAD edges` gate. Per the test
boundary, no VAD threshold or further fixed-delay tuning was performed, and human
double-talk acceptance was not run. Hardware release acceptance remains unsigned.

## Reproduction

Keep the near-end microphone area silent:

```bash
TTS_ENABLED=1 AUDIO_INPUT_BACKEND=alsa ALSA_INPUT_CARD=Microphone \
  ALSA_OUTPUT_CARD=Audio AUDIO_OUTPUT_INTERRUPT_STRATEGY=persistent \
  SPEECH_TRANSPORT=inprocess PYTHONPATH=src \
  .venv-gpu/bin/python tests/acceptance/acoustic_path_diagnostic.py \
  --config config.tts-demo.local.json \
  --artifact-dir /tmp/acoustic-trace-block20-delay80 \
  --block-ms 20 --stream-delay-ms 80 --force-cpu-asr
```

The retained sweep artifacts for this run are under
`/tmp/acoustic-sweep-delay{60,80,100,120,140}`. Hardware acceptance remains
unsigned.
