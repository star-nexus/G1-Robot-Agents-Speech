"""Managed ROS 2 lifecycle node for the speech service."""

from __future__ import annotations

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.lifecycle import LifecycleNode, TransitionCallbackReturn

from g1_speech.app import SpeechService
from g1_speech.config import load_config
from g1_speech.lifecycle import SpeechLifecycleController


class SpeechLifecycleNode(LifecycleNode):
    def __init__(self) -> None:
        super().__init__("g1_speech")
        self.declare_parameter("config_file", "config.json")
        self._controller = SpeechLifecycleController(self._create_service)

    def _create_service(self, config):
        return SpeechService(
            config,
            transport_backend="ros2",
            ros_node=self,
            ros_lifecycle=True,
        )

    def on_configure(self, state):
        try:
            config_file = self.get_parameter("config_file").value
            self._controller.configure(load_config(config_file))
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f"Failed to configure speech service: {exc}")
            return TransitionCallbackReturn.FAILURE
        self.get_logger().info("Speech service configured and model loaded")
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state):
        result = super().on_activate(state)
        if result != TransitionCallbackReturn.SUCCESS:
            return result
        try:
            self._controller.activate()
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f"Failed to activate speech service: {exc}")
            super().on_deactivate(state)
            return TransitionCallbackReturn.FAILURE
        self.get_logger().info("Speech service active")
        return TransitionCallbackReturn.SUCCESS

    def on_deactivate(self, state):
        try:
            self._controller.deactivate()
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f"Failed to deactivate speech service: {exc}")
            return TransitionCallbackReturn.FAILURE
        result = super().on_deactivate(state)
        if result == TransitionCallbackReturn.SUCCESS:
            self.get_logger().info("Speech service inactive")
        return result

    def on_cleanup(self, state):
        try:
            self._controller.cleanup()
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f"Failed to clean up speech service: {exc}")
            return TransitionCallbackReturn.FAILURE
        self.get_logger().info("Speech service resources released")
        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, state):
        self._controller.cleanup()
        return TransitionCallbackReturn.SUCCESS


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SpeechLifecycleNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node._controller.cleanup()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
