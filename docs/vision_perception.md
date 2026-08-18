# Vision perception

STAR Runtime treats vision as an optional observation input, not as part of the
camera driver, Agent identity, or transport layer:

```text
V4L2 MJPEG camera -> LatestFrameCamera -> VisionRouter -> current Agent turn
                                                |
                              rules or constrained llama.cpp classifier
```

`LatestFrameCamera` asks FFmpeg to copy the camera's MJPEG stream without decoding or
re-encoding it and retains only the newest complete JPEG. A frame is converted to an
OpenAI-compatible data URL only when the current turn needs vision. The completed turn
stored in `WindowMemory` contains the user's original text and the assistant response,
never the image. This bounds host memory and model context growth on a 16 GB Orin NX.

## Start a visual Runtime

The camera must expose MJPEG through V4L2, and `ffmpeg` must be available. Start a
llama.cpp server with a Qwen3-VL GGUF and its matching multimodal projector:

```bash
scripts/run-qwen3-vl-llama.sh \
  models/Qwen3-VL-4B-Instruct-Q4_K_M.gguf \
  models/mmproj-Qwen3-VL-4B-Instruct-f16.gguf
```

Then start the integrated speech Runtime. The alias below matches the visual server
launcher's default:

```bash
.venv/bin/g1-speech runtime \
  --config config.tts-demo.local.json \
  --role-package roles/tifa-lockhart \
  --model qwen3-vl-4b-q4 \
  --vision auto \
  --camera-device /dev/video0
```

Modes are deliberately simple:

- `--vision off` is the default and creates no camera or classifier objects.
- `--vision auto` uses explicit visual-language rules first, then a constrained V/T/U
  classifier. Uncertainty or classifier failure safely selects a fresh image.
- `--vision always` attaches a fresh image on every user turn and makes no separate
  classifier request.

Camera defaults are 640×480 at 5 FPS, with a two-second freshness limit. Override
them with `--camera-width`, `--camera-height`, `--camera-fps`, and
`--camera-frame-max-age`. `--vision-route-log PATH` enables an optional mode-0600
JSONL decision log; it is disabled by default.

## Split across machines

The same command works when the VLM runs on another Orin NX or a server: set `--url`
to that machine's OpenAI-compatible llama.cpp endpoint. ASR, Agent, and TTS can still
be split with DDS or ROS 2 by using `star-runtime serve` and `star-runtime agent`.
Visual routing and the current image remain internal to the Agent process; only the
selected model request is sent to the configured model endpoint.
