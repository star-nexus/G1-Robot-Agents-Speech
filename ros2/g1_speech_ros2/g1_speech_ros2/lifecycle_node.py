"""Managed ROS 2 lifecycle node for the speech service."""

from __future__ import annotations

import logging
import signal
import threading

import rclpy
from rcl_interfaces.msg import SetParametersResult
from rclpy.executors import ExternalShutdownException, SingleThreadedExecutor
from rclpy.lifecycle import LifecycleNode, TransitionCallbackReturn

from g1_speech.app import SpeechService
from g1_speech.config import load_config
from g1_speech.lifecycle import SpeechLifecycleController

logger = logging.getLogger(__name__)


class SpeechLifecycleNode(LifecycleNode):
    def __init__(self) -> None:
        super().__init__("g1_speech")
        self.declare_parameter("config_file", "config.json")
        self._controller = SpeechLifecycleController(self._create_service)
        self.add_on_set_parameters_callback(self._validate_parameter_update)

    def _validate_parameter_update(self, parameters):
        for parameter in parameters:
            if parameter.name != "config_file":
                continue
            if self._controller.configured:
                return SetParametersResult(
                    successful=False,
                    reason=(
                        "config_file can only be changed while the lifecycle node "
                        "is Unconfigured; run cleanup first"
                    ),
                )
            if not isinstance(parameter.value, str) or not parameter.value.strip():
                return SetParametersResult(
                    successful=False,
                    reason="config_file must be a non-empty path",
                )
        return SetParametersResult(successful=True)

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
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    rclpy.init(args=args)
    node = SpeechLifecycleNode()
    stopped = threading.Event()
    executor = SingleThreadedExecutor()
    executor.add_node(node)

    def stop(*_args) -> None:
        logger.info("Lifecycle node shutdown requested")
        stopped.set()
        executor.wake()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    try:
        while not stopped.is_set():
            executor.spin_once(timeout_sec=0.2)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        logger.info("Lifecycle node cleanup started")
        node._controller.cleanup()
        executor.remove_node(node)
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        logger.info("Lifecycle node shutdown complete")


if __name__ == "__main__":
    main()
