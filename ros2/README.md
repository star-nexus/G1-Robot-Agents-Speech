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

Topic names are relative internally, so ROS namespaces and remapping work
without changing service configuration. In the root namespace they resolve to
the paths above.

After sourcing the ROS installation and workspace in the subscriber terminal:

```bash
ros2 topic echo /hri/speech/final g1_speech_msgs/msg/SpeechEvent
```

## Lifecycle node

For applications that need explicit lifecycle transitions instead of the
standalone background service:

```bash
source /opt/ros/$ROS_DISTRO/setup.bash
source ros2_ws/install/setup.bash

ros2 launch g1_speech_ros2 speech_lifecycle.launch.py \
  config_file:="$PWD/config.json"
ros2 lifecycle set /g1_speech configure
ros2 lifecycle set /g1_speech activate
```

The lifecycle mapping is:

- `configure`: construct the pipeline and load SenseVoice without opening the microphone.
- `activate`: start transport, microphone capture, VAD, and recognition workers.
- `deactivate`: stop capture and workers while retaining the loaded model.
- `cleanup`: release transport and pipeline resources.

## Acceptance test

```bash
source /opt/ros/$ROS_DISTRO/setup.bash
source ros2_ws/install/setup.bash
PYTHONPATH="$PWD/src:$PYTHONPATH" python acceptance/ros2_roundtrip.py
```
