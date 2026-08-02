from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import LifecycleNode


def generate_launch_description():
    config_file = LaunchConfiguration("config_file")
    namespace = LaunchConfiguration("namespace")
    return LaunchDescription(
        [
            DeclareLaunchArgument("config_file", default_value="config.json"),
            DeclareLaunchArgument("namespace", default_value=""),
            LifecycleNode(
                package="g1_speech_ros2",
                executable="speech_lifecycle_node",
                name="g1_speech",
                namespace=namespace,
                output="screen",
                parameters=[{"config_file": config_file}],
            ),
        ]
    )
