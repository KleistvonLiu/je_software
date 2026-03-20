#!/usr/bin/env python3
"""Terminal-driven PCB arrival signal publisher."""

from __future__ import annotations

import select
import sys
import termios
import tty
from typing import TextIO

import rclpy
from action_msgs.msg import GoalStatus
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy
from rclpy.qos import QoSProfile
from rclpy.qos import ReliabilityPolicy

from je_software.action import RecoverToInitial
from je_software.msg import PcbPresence

ANSI_GREEN = '\033[32m'
ANSI_RESET = '\033[0m'


def green_text(text: str) -> str:
    return f'{ANSI_GREEN}{text}{ANSI_RESET}'


class TerminalKeySignalNode(Node):
    """Publish fixed PCB presence states from terminal key presses."""

    def __init__(self) -> None:
        super().__init__('terminal_key_signal_node')
        self.declare_parameter('presence_topic', '/vision/line/pcb_presence')
        self.declare_parameter('publish_hz', 5.0)
        self.declare_parameter('poll_hz', 20.0)
        self.declare_parameter('key_trigger', 'p')
        self.declare_parameter('key_clear', 'c')
        self.declare_parameter('key_recover', 'r')
        self.declare_parameter('key_quit', 'q')
        self.declare_parameter(
            'recover_action_name',
            '/pcb_process/recover_to_initial',
        )
        self.declare_parameter('source', 'terminal')

        self.presence_topic = str(self.get_parameter('presence_topic').value)
        publish_hz = float(self.get_parameter('publish_hz').value)
        poll_hz = float(self.get_parameter('poll_hz').value)
        self.key_trigger = str(self.get_parameter('key_trigger').value)
        self.key_clear = str(self.get_parameter('key_clear').value)
        self.key_recover = str(self.get_parameter('key_recover').value)
        self.key_quit = str(self.get_parameter('key_quit').value)
        self.recover_action_name = str(
            self.get_parameter('recover_action_name').value
        )
        self.source = str(self.get_parameter('source').value)

        qos = QoSProfile(depth=1)
        qos.reliability = ReliabilityPolicy.RELIABLE
        qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self.publisher = self.create_publisher(PcbPresence, self.presence_topic, qos)
        self.recover_action_client = ActionClient(
            self,
            RecoverToInitial,
            self.recover_action_name,
        )

        self._present = False
        self._stable = False
        self._ready = False
        self._input_stream: TextIO | None = None
        self._input_fd: int | None = None
        self._original_termios = None
        self._recover_goal_future = None
        self._recover_result_future = None
        self._recover_goal_handle = None

        self._prepare_input_stream()
        self._publish_timer = self.create_timer(
            1.0 / max(publish_hz, 1.0),
            self._publish_current_state,
        )
        self._poll_timer = self.create_timer(
            1.0 / max(poll_hz, 1.0),
            self._poll_keypress,
        )

        self.get_logger().info(
            'Keyboard ready: '
            f'{self.key_trigger}=pcb ready, '
            f'{self.key_clear}=clear, '
            f'{self.key_recover}=recover to initial, '
            f'{self.key_quit}=quit'
        )
        self._publish_current_state()

    def _prepare_input_stream(self) -> None:
        stream: TextIO | None = None
        if sys.stdin is not None and sys.stdin.isatty():
            stream = sys.stdin
        else:
            try:
                stream = open('/dev/tty', 'r', encoding='utf-8', buffering=1)
            except OSError as exc:
                self.get_logger().warn(f'No foreground TTY available: {exc}')
                return

        fd = stream.fileno()
        try:
            self._original_termios = termios.tcgetattr(fd)
            tty.setcbreak(fd)
        except termios.error as exc:
            self.get_logger().warn(f'Failed to enable cbreak mode: {exc}')
            if stream is not sys.stdin:
                stream.close()
            return

        self._input_stream = stream
        self._input_fd = fd

    def _set_state(self, present: bool) -> None:
        self._present = bool(present)
        self._stable = bool(present)
        self._ready = bool(present)
        self._publish_current_state()
        label = 'READY' if present else 'CLEARED'
        self.get_logger().info(green_text(f'PCB signal state -> {label}'))

    def _poll_keypress(self) -> None:
        if self._input_stream is None or self._input_fd is None:
            return

        readable, _, _ = select.select([self._input_fd], [], [], 0.0)
        if not readable:
            return

        try:
            key = self._input_stream.read(1)
        except OSError as exc:
            self.get_logger().warn(f'Keyboard read failed: {exc}')
            return

        if not key:
            return
        key = key.lower()
        if key == self.key_trigger:
            self._set_state(True)
        elif key == self.key_clear:
            self._set_state(False)
        elif key == self.key_recover:
            self._send_recover_goal()
        elif key == self.key_quit:
            self.get_logger().info('Quit requested from keyboard.')
            rclpy.shutdown()

    def _send_recover_goal(self) -> None:
        if self._recover_goal_future is not None or self._recover_result_future is not None:
            self.get_logger().warn('Recover-to-initial is already running.')
            return

        if not self.recover_action_client.wait_for_server(timeout_sec=0.0):
            self.get_logger().warn(
                f'Recover action server not ready: {self.recover_action_name}'
            )
            return

        self.get_logger().info('Requesting recover-to-initial action.')
        future = self.recover_action_client.send_goal_async(
            RecoverToInitial.Goal(),
            feedback_callback=self._on_recover_feedback,
        )
        self._recover_goal_future = future
        future.add_done_callback(self._on_recover_goal_response)

    def _on_recover_feedback(self, feedback_msg) -> None:
        feedback = feedback_msg.feedback
        self.get_logger().info(
            'Recover feedback: '
            f'stage={feedback.stage} '
            f'step={feedback.step_index} '
            f'name={feedback.active_step_name}'
        )

    def _on_recover_goal_response(self, future) -> None:
        self._recover_goal_future = None
        try:
            goal_handle = future.result()
        except Exception as exc:  # pragma: no cover - defensive logging
            self.get_logger().error(f'Failed to send recover goal: {exc}')
            return

        if not goal_handle.accepted:
            self.get_logger().warn('Recover-to-initial goal was rejected.')
            return

        self._recover_goal_handle = goal_handle
        self.get_logger().info(green_text('Recover-to-initial goal accepted.'))
        result_future = goal_handle.get_result_async()
        self._recover_result_future = result_future
        result_future.add_done_callback(self._on_recover_result)

    def _on_recover_result(self, future) -> None:
        self._recover_result_future = None
        self._recover_goal_handle = None
        try:
            goal_result = future.result()
        except Exception as exc:  # pragma: no cover - defensive logging
            self.get_logger().error(f'Failed to get recover result: {exc}')
            return

        result = goal_result.result
        if (
            goal_result.status == GoalStatus.STATUS_SUCCEEDED
            and result.success
        ):
            self.get_logger().info(
                green_text('Recover-to-initial completed successfully.')
            )
            return

        self.get_logger().error(
            'Recover-to-initial failed: '
            f'status={goal_result.status} '
            f'error_code={result.error_code}'
        )

    def _publish_current_state(self) -> None:
        msg = PcbPresence()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.present = self._present
        msg.stable = self._stable
        msg.ready_for_pick = self._ready
        msg.source = self.source
        self.publisher.publish(msg)

    def destroy_node(self) -> bool:
        if self._input_fd is not None and self._original_termios is not None:
            try:
                termios.tcsetattr(
                    self._input_fd,
                    termios.TCSADRAIN,
                    self._original_termios,
                )
            except termios.error:
                pass
        if self._input_stream is not None and self._input_stream is not sys.stdin:
            self._input_stream.close()
        return super().destroy_node()


def main() -> None:
    rclpy.init()
    node = TerminalKeySignalNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
