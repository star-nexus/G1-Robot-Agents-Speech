# ROS 2 Integration

The core service keeps ROS 2 optional. ROS dependencies are installed through
the active ROS distribution and `rosdep`, not from PyPI.

```bash
source /opt/ros/$ROS_DISTRO/setup.bash
python3 -m venv --system-site-packages .venv-ros2
source .venv-ros2/bin/activate
python -m pip install -e '.[asr]'

mkdir -p ros2_ws/src
ln -s "$PWD/ros2/g1_speech_msgs" ros2_ws/src/g1_speech_msgs
ln -s "$PWD/ros2/g1_speech_ros2" ros2_ws/src/g1_speech_ros2
cd ros2_ws
rosdep install --from-paths src --ignore-src -r -y
python -m colcon build --symlink-install
source install/setup.bash
```

Standalone ROS 2 transport:

```bash
g1-speech serve --config config.json --transport ros2
```

Managed lifecycle node:

```bash
ros2 launch g1_speech_ros2 speech_lifecycle.launch.py \
  config_file:="$PWD/config.json"
ros2 lifecycle set /g1_speech configure
ros2 lifecycle set /g1_speech activate
```

Topics:

| Topic | Type | Direction |
|---|---|---|
| `hri/speech/final` | `g1_speech_msgs/msg/SpeechEvent` | Service to Agent |
| `hri/playback/state` | `g1_speech_msgs/msg/PlaybackState` | Agent/TTS to service |

Topic names are relative so ROS namespaces and remapping work without changing
the service configuration. In the root namespace they resolve to `/hri/...`.

Run the transport acceptance test after sourcing the workspace:

```bash
PYTHONPATH="$PWD/src:$PYTHONPATH" python acceptance/ros2_roundtrip.py
```

The lifecycle mapping is:

- `configure`: construct the pipeline and load SenseVoice without opening the microphone.
- `activate`: start transport, microphone capture, VAD, and recognition workers.
- `deactivate`: stop capture and workers while retaining the loaded model.
- `cleanup`: release transport and pipeline resources.
