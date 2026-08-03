from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, EmitEvent, RegisterEventHandler
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessStart
from launch.events import matches_action
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import LifecycleNode
from launch_ros.event_handlers import OnStateTransition
from launch_ros.events.lifecycle import ChangeState
from lifecycle_msgs.msg import Transition


def generate_launch_description():
    config_file = LaunchConfiguration("config_file")
    namespace = LaunchConfiguration("namespace")
    autostart = LaunchConfiguration("autostart")
    speech_node = LifecycleNode(
        package="g1_speech_ros2",
        executable="speech_lifecycle_node",
        name="g1_speech",
        namespace=namespace,
        output="screen",
        parameters=[{"config_file": config_file}],
    )

    configure_on_start = RegisterEventHandler(
        OnProcessStart(
            target_action=speech_node,
            on_start=[
                EmitEvent(
                    event=ChangeState(
                        lifecycle_node_matcher=matches_action(speech_node),
                        transition_id=Transition.TRANSITION_CONFIGURE,
                    )
                )
            ],
        ),
        condition=IfCondition(autostart),
    )
    activate_when_inactive = RegisterEventHandler(
        OnStateTransition(
            target_lifecycle_node=speech_node,
            goal_state="inactive",
            entities=[
                EmitEvent(
                    event=ChangeState(
                        lifecycle_node_matcher=matches_action(speech_node),
                        transition_id=Transition.TRANSITION_ACTIVATE,
                    )
                )
            ],
        ),
        condition=IfCondition(autostart),
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("config_file", default_value="config.json"),
            DeclareLaunchArgument("namespace", default_value=""),
            DeclareLaunchArgument(
                "autostart",
                default_value="false",
                description="Configure and activate the speech node automatically",
            ),
            configure_on_start,
            activate_when_inactive,
            speech_node,
        ]
    )
