#!/usr/bin/env python3
"""Exercise the real ROS 2 lifecycle node through a complete state cycle."""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import rclpy
from lifecycle_msgs.msg import State, Transition
from lifecycle_msgs.srv import ChangeState, GetState
from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
from rcl_interfaces.srv import SetParameters


def _service_name(node_name: str, service: str) -> str:
    return f"{node_name.rstrip('/')}/{service}"


class LifecycleClient:
    def __init__(self, node, target: str, timeout: float) -> None:
        self._node = node
        self._timeout = timeout
        self._change = node.create_client(
            ChangeState, _service_name(target, "change_state")
        )
        self._state = node.create_client(GetState, _service_name(target, "get_state"))
        self._parameters = node.create_client(
            SetParameters, _service_name(target, "set_parameters")
        )

    def wait_until_ready(self) -> None:
        deadline = time.monotonic() + self._timeout
        for client in (self._change, self._state, self._parameters):
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not client.wait_for_service(timeout_sec=remaining):
                raise TimeoutError(f"ROS 2 service was not ready: {client.srv_name}")

    def state(self) -> int:
        response = self._call(self._state, GetState.Request())
        return response.current_state.id

    def wait_for_state(self, expected: int) -> None:
        deadline = time.monotonic() + self._timeout
        while time.monotonic() < deadline:
            if self.state() == expected:
                return
            time.sleep(0.05)
        raise TimeoutError(f"lifecycle state did not become {expected}")

    def transition(self, transition_id: int, expected_state: int) -> None:
        request = ChangeState.Request()
        request.transition.id = transition_id
        response = self._call(self._change, request)
        if not response.success:
            raise RuntimeError(f"lifecycle transition {transition_id} was rejected")
        self.wait_for_state(expected_state)

    def set_config_file(self, path: Path) -> tuple[bool, str]:
        request = SetParameters.Request()
        request.parameters = [
            Parameter(
                name="config_file",
                value=ParameterValue(
                    type=ParameterType.PARAMETER_STRING,
                    string_value=str(path),
                ),
            )
        ]
        response = self._call(self._parameters, request)
        result = response.results[0]
        return result.successful, result.reason

    def _call(self, client, request):
        future = client.call_async(request)
        rclpy.spin_until_future_complete(self._node, future, timeout_sec=self._timeout)
        if not future.done():
            raise TimeoutError(f"ROS 2 service call timed out: {client.srv_name}")
        exception = future.exception()
        if exception is not None:
            raise RuntimeError(f"ROS 2 service call failed: {client.srv_name}") from exception
        return future.result()


def _terminate(process: subprocess.Popen[str], timeout: float) -> str:
    if process.poll() is None:
        process.send_signal(signal.SIGINT)
    try:
        output, _ = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        process.terminate()
        try:
            output, _ = process.communicate(timeout=3.0)
        except subprocess.TimeoutExpired:
            process.kill()
            output, _ = process.communicate()
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--node", default="/g1_speech")
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()
    if args.timeout <= 0:
        raise ValueError("timeout must be greater than zero")
    config = args.config.expanduser().resolve()
    if not config.is_file():
        raise FileNotFoundError(config)

    environment = os.environ.copy()
    command = [
        sys.executable,
        "-m",
        "g1_speech_ros2.lifecycle_node",
        "--ros-args",
        "-p",
        f"config_file:={config}",
    ]
    process = subprocess.Popen(
        command,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    output = ""
    rclpy.init()
    client_node = rclpy.create_node("g1_speech_lifecycle_acceptance")
    client = LifecycleClient(client_node, args.node, args.timeout)
    try:
        client.wait_until_ready()
        client.wait_for_state(State.PRIMARY_STATE_UNCONFIGURED)

        client.transition(
            Transition.TRANSITION_CONFIGURE,
            State.PRIMARY_STATE_INACTIVE,
        )
        accepted, reason = client.set_config_file(config)
        if accepted or "Unconfigured" not in reason:
            raise AssertionError(
                "config_file modification was not rejected while configured: "
                f"accepted={accepted} reason={reason!r}"
            )

        client.transition(
            Transition.TRANSITION_ACTIVATE,
            State.PRIMARY_STATE_ACTIVE,
        )
        client.transition(
            Transition.TRANSITION_DEACTIVATE,
            State.PRIMARY_STATE_INACTIVE,
        )
        client.transition(
            Transition.TRANSITION_ACTIVATE,
            State.PRIMARY_STATE_ACTIVE,
        )
        client.transition(
            Transition.TRANSITION_DEACTIVATE,
            State.PRIMARY_STATE_INACTIVE,
        )
        client.transition(
            Transition.TRANSITION_CLEANUP,
            State.PRIMARY_STATE_UNCONFIGURED,
        )
        accepted, reason = client.set_config_file(config)
        if not accepted:
            raise AssertionError(
                f"config_file modification failed while Unconfigured: {reason}"
            )
    finally:
        client_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        output = _terminate(process, args.timeout)

    if process.returncode not in (0, -signal.SIGINT):
        raise RuntimeError(
            f"lifecycle node exited with {process.returncode}\n{output}"
        )
    if output.count("Loading SenseVoice:") != 1:
        raise AssertionError(
            "SenseVoice was not loaded exactly once across reactivation\n" + output
        )
    if output.count("Microphone started:") != 2:
        raise AssertionError(
            "microphone did not restart exactly once across reactivation\n" + output
        )
    print("[PASS] ROS 2 lifecycle configure/activate/reactivate/cleanup")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
