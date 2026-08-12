# ROS 2 Integration

ROS 2 is optional. The default DDS deployment and its dependencies remain
unchanged.

## One-time setup

Install ROS 2 for your platform, then run from the project root:

```bash
bash scripts/setup-ros2.sh
```

The script discovers the installed ROS distribution, creates and builds
`ros2_ws`, verifies the Python message bindings, and installs the ROS 2 systemd
units. It preserves the current shell's `ROS_DOMAIN_ID` for the background
services. Set `ROS2_DOMAIN_ID`, `ROS2_SETUP`, or `ROS2_WORKSPACE_SETUP` in
`deploy.env` only when the automatic values are not suitable.

## Background service

Choose the inference backend and ROS 2 transport with one command:

```bash
sudo g1-speech-service cpu ros2
sudo g1-speech-service gpu ros2

g1-speech-service status
g1-speech-service logs
```

Switch back to DDS without reinstalling:

```bash
sudo g1-speech-service cpu dds
sudo g1-speech-service gpu dds
```

The four CPU/GPU and DDS/ROS 2 services are mutually exclusive. The selector
stops the previous mode, starts the requested mode, and enables it at boot.

## Topics

| Topic | Type | Direction |
|---|---|---|
| `/hri/speech/final` | `g1_speech_msgs/msg/SpeechEvent` | Service to Agent |
| `/hri/playback/state` | `g1_speech_msgs/msg/PlaybackState` | Agent/TTS to service |
| `/hri/tts/request` | `g1_speech_msgs/msg/TtsTextChunk` | Agent to streaming TTS |

Topic names are relative internally, so ROS namespaces and remapping work
without changing service configuration. In the root namespace they resolve to
the paths above.

After sourcing the ROS installation and workspace in the subscriber terminal:

```bash
ros2 topic echo /hri/speech/final g1_speech_msgs/msg/SpeechEvent
```

## Lifecycle node

For applications that need explicit lifecycle transitions instead of the
standalone background service, launch and activate it in one command:

```bash
ros2 launch g1_speech_ros2 speech_lifecycle.launch.py \
  config_file:="$PWD/config.json" autostart:=true
```

Set `autostart:=false` when an external lifecycle manager should control the
node, then transition it explicitly:

```bash
ros2 launch g1_speech_ros2 speech_lifecycle.launch.py \
  config_file:="$PWD/config.json" autostart:=false
ros2 lifecycle set /g1_speech configure
ros2 lifecycle set /g1_speech activate
```

The lifecycle mapping is:

- `configure`: construct the pipeline and load SenseVoice without opening the microphone.
- `activate`: start transport, microphone capture, VAD, and recognition workers.
- `deactivate`: stop capture and workers while retaining the loaded model.
- `cleanup`: release transport and pipeline resources.

`config_file` is a startup/configuration parameter. It can be changed while the
node is Unconfigured, but changes are rejected while Inactive or Active. Run
`cleanup`, change the path, and configure again. Resource parameters are not
hot-switched; CPU/GPU selection remains an explicit deployment decision.

## Acceptance test

```bash
source /opt/ros/$ROS_DISTRO/setup.bash
source ros2_ws/install/setup.bash
PYTHONPATH="$PWD/src:$PYTHONPATH" python acceptance/ros2_roundtrip.py
```

The complete lifecycle test uses the current Python environment to load either
the CPU or GPU runtime. Stop the background speech service first so the test can
own the microphone:

```bash
sudo g1-speech-service stop

PYTHONPATH="$PWD/src:$PYTHONPATH" \
  .venv/bin/python acceptance/ros2_lifecycle.py --config config.json

# GPU variant
PYTHONPATH="$PWD/src:$PYTHONPATH" \
  .venv-gpu/bin/python acceptance/ros2_lifecycle.py --config config.gpu.json
```

This test performs configure, activate, deactivate, reactivate, and cleanup on
the real ROS graph. It also verifies that SenseVoice is loaded once, the
microphone restarts, and unsafe `config_file` changes are rejected.
