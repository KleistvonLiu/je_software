#!/usr/bin/env python3
"""Replay JEARM joint states from a JSONL log for RViz."""

from __future__ import annotations

import json
import select
import sys
import termios
import time
import tty
from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import QoSProfile
from rclpy.qos import ReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import JointState

DEFAULT_JOINT_NAMES = [
    'joint1',
    'joint2',
    'joint3',
    'joint4',
    'joint5',
    'joint6',
    'joint7',
]
DEFAULT_ROBOT_KEY = 'Robot0'
ANSI_GREEN = '\033[32m'
ANSI_RESET = '\033[0m'


def green_text(text: str) -> str:
    return f'{ANSI_GREEN}{text}{ANSI_RESET}'


@dataclass(frozen=True)
class JointFrame:
    """Single replay frame extracted from the JSONL log."""

    stamp_ns: int
    positions: list[float]


def _logger_warn(logger, message: str) -> None:
    if logger is not None:
        logger.warning(message)


def _coerce_stamp_ns(record: dict) -> int:
    """Use the recorded ROS time when it exists, otherwise fall back to seconds."""
    stamp_ns = record.get('__ros_stamp_ns')
    if stamp_ns is not None:
        try:
            return max(int(stamp_ns), 0)
        except (TypeError, ValueError):
            return 0

    stamp_sec = record.get('__ros_stamp_sec')
    if stamp_sec is None:
        return 0
    try:
        return max(int(float(stamp_sec) * 1_000_000_000), 0)
    except (TypeError, ValueError):
        return 0


def load_joint_frames_from_jsonl(
    jsonl_path: str,
    *,
    robot_key: str = DEFAULT_ROBOT_KEY,
    joint_count: int = 7,
    logger=None,
) -> list[JointFrame]:
    """Load valid joint frames from the JSONL file."""
    path = Path(jsonl_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f'JSONL file does not exist: {path}')
    if path.is_dir():
        raise IsADirectoryError(f'JSONL path is a directory: {path}')

    frames: list[JointFrame] = []
    with path.open('r', encoding='utf-8') as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                _logger_warn(
                    logger,
                    f'Skipping invalid JSON at line {line_no}: {exc}',
                )
                continue

            robot = record.get(robot_key)
            if not isinstance(robot, dict):
                _logger_warn(
                    logger,
                    f'Skipping line {line_no}: missing {robot_key} object',
                )
                continue

            joint_values = robot.get('Joint')
            if not isinstance(joint_values, list) or len(joint_values) != joint_count:
                _logger_warn(
                    logger,
                    f'Skipping line {line_no}: {robot_key}.Joint length is not {joint_count}',
                )
                continue

            try:
                positions = [float(value) for value in joint_values]
            except (TypeError, ValueError) as exc:
                _logger_warn(
                    logger,
                    f'Skipping line {line_no}: invalid joint value: {exc}',
                )
                continue

            frames.append(
                JointFrame(
                    stamp_ns=_coerce_stamp_ns(record),
                    positions=positions,
                )
            )

    if not frames:
        raise ValueError(f'No valid joint frames found in {path}')
    return frames


def make_joint_state_message(
    frame: JointFrame,
    joint_names: list[str],
) -> JointState:
    """Convert a replay frame into a standard JointState message."""
    msg = JointState()
    msg.name = list(joint_names)
    msg.position = list(frame.positions)
    if frame.stamp_ns > 0:
        msg.header.stamp = Time(nanoseconds=frame.stamp_ns).to_msg()
    return msg


class JearmJsonlReplayerNode(Node):
    """Replay a single-arm JSONL log as /joint_states for RViz."""

    def __init__(self) -> None:
        super().__init__('jearm_jsonl_replayer_node')
        self.declare_parameter('jsonl_path', '')
        self.declare_parameter('joint_state_topic', '/joint_states')
        self.declare_parameter('joint_names', DEFAULT_JOINT_NAMES)
        self.declare_parameter('rate_hz', 30.0)
        self.declare_parameter('paused', False)
        self.declare_parameter('use_recorded_timestamps', False)
        self.declare_parameter('follow_recorded_timing', True)

        self.jsonl_path = str(self.get_parameter('jsonl_path').value).strip()
        if not self.jsonl_path:
            raise ValueError('Parameter "jsonl_path" must not be empty.')

        self.joint_state_topic = str(self.get_parameter('joint_state_topic').value)
        self.joint_names = [
            str(name)
            for name in self.get_parameter('joint_names').value
        ]
        if len(self.joint_names) != 7:
            raise ValueError('Parameter "joint_names" must contain exactly 7 names.')

        # 这里的 rate_hz 是“输出到 /joint_states 的刷新频率”，不是“每秒推进多少条原始日志帧”。
        # 当 follow_recorded_timing=true 时，节点会按日志时间轴做插值，然后以该频率连续发布。
        self.rate_hz = max(float(self.get_parameter('rate_hz').value), 1e-6)
        self._paused = bool(self.get_parameter('paused').value)
        # RViz/TF 默认按当前系统时间消费变换。
        # 如果直接回放日志里的旧时间戳，robot_state_publisher 会继续沿用这些历史时间，
        # 最终在 RViz 中看起来像“模型能显示，但不跟着动”。
        # 因此默认使用当前发布时间；只有明确需要复现原始录制时间轴时才打开该开关。
        self._use_recorded_timestamps = bool(
            self.get_parameter('use_recorded_timestamps').value
        )
        # 是否沿用日志原始时间轴来控制播放速度。
        # 打开后，会根据录制时的帧间隔做线性插值，视觉上会比“固定频率逐帧跳着播”平滑很多。
        self._follow_recorded_timing = bool(
            self.get_parameter('follow_recorded_timing').value
        )
        self._current_index = 0
        self._input_stream: TextIO | None = None
        self._input_fd: int | None = None
        self._original_termios = None

        self._frames = load_joint_frames_from_jsonl(
            self.jsonl_path,
            robot_key=DEFAULT_ROBOT_KEY,
            joint_count=len(self.joint_names),
            logger=self.get_logger(),
        )
        self._frame_offsets_ns = self._build_frame_offsets_ns()
        self._playback_elapsed_ns = 0
        self._last_tick_ns = time.monotonic_ns()

        qos = QoSProfile(depth=10)
        qos.reliability = ReliabilityPolicy.RELIABLE
        self.publisher = self.create_publisher(
            JointState,
            self.joint_state_topic,
            qos,
        )

        self._prepare_input_stream()
        self._replay_timer = self.create_timer(1.0 / self.rate_hz, self._replay_once)
        self._poll_timer = self.create_timer(1.0 / 20.0, self._poll_keypress)

        # Publish the first frame immediately so RViz has a valid pose at startup.
        self._publish_current_frame()

        self.get_logger().info(
            'JEARM JSONL replayer ready: '
            f'file={self.jsonl_path}, '
            f'frames={len(self._frames)}, '
            f'rate_hz={self.rate_hz:.3f}, '
            f'paused={self._paused}, '
            f'use_recorded_timestamps={self._use_recorded_timestamps}, '
            f'follow_recorded_timing={self._follow_recorded_timing}'
        )
        self.get_logger().info('Keyboard ready: space=pause/resume, q=quit')

    def _build_frame_offsets_ns(self) -> list[int]:
        """把日志里的绝对时间戳转换成从第 0 帧开始的相对时间。

        如果日志没有可靠时间戳，或者用户关闭了 follow_recorded_timing，
        后续就退化回“固定频率逐帧推进”的老逻辑。
        """
        if not self._follow_recorded_timing or len(self._frames) < 2:
            return []

        first_stamp_ns = self._frames[0].stamp_ns
        if first_stamp_ns <= 0:
            return []

        offsets = [0]
        last_offset = 0
        for frame in self._frames[1:]:
            offset_ns = frame.stamp_ns - first_stamp_ns
            if offset_ns <= last_offset:
                return []
            offsets.append(offset_ns)
            last_offset = offset_ns
        return offsets

    def _sample_positions(self) -> list[float]:
        """根据当前播放时间，返回应当显示的关节位置。

        有原始时间轴时做线性插值；没有时退回固定频率逐帧播放。
        """
        if not self._frame_offsets_ns:
            return list(self._frames[self._current_index].positions)

        if self._playback_elapsed_ns <= 0:
            self._current_index = 0
            return list(self._frames[0].positions)

        if self._playback_elapsed_ns >= self._frame_offsets_ns[-1]:
            self._current_index = len(self._frames) - 1
            return list(self._frames[-1].positions)

        upper_index = bisect_right(self._frame_offsets_ns, self._playback_elapsed_ns)
        lower_index = max(upper_index - 1, 0)
        self._current_index = lower_index

        start_ns = self._frame_offsets_ns[lower_index]
        end_ns = self._frame_offsets_ns[upper_index]
        span_ns = max(end_ns - start_ns, 1)
        alpha = (self._playback_elapsed_ns - start_ns) / span_ns

        start_positions = self._frames[lower_index].positions
        end_positions = self._frames[upper_index].positions
        return [
            start + (end - start) * alpha
            for start, end in zip(start_positions, end_positions)
        ]

    def _prepare_input_stream(self) -> None:
        stream: TextIO | None = None
        if sys.stdin is not None and sys.stdin.isatty():
            stream = sys.stdin
        else:
            try:
                stream = open('/dev/tty', 'r', encoding='utf-8', buffering=1)
            except OSError as exc:
                self.get_logger().warning(f'No foreground TTY available: {exc}')
                return

        fd = stream.fileno()
        try:
            self._original_termios = termios.tcgetattr(fd)
            tty.setcbreak(fd)
        except termios.error as exc:
            self.get_logger().warning(f'Failed to enable cbreak mode: {exc}')
            if stream is not sys.stdin:
                stream.close()
            return

        self._input_stream = stream
        self._input_fd = fd

    def _publish_current_frame(self) -> None:
        positions = self._sample_positions()
        formatted_positions = "[" + ", ".join(f"{x:.4f}" for x in positions) + "]"
        print(
            f'[jearm_jsonl_replayer_node] publish frame={self._current_index} '
            f'joint_positions={formatted_positions}',
            flush=True,
        )
        msg = make_joint_state_message(
            JointFrame(
                stamp_ns=self._frames[self._current_index].stamp_ns,
                positions=positions,
            ),
            self.joint_names,
        )
        if not self._use_recorded_timestamps:
            msg.header.stamp = self.get_clock().now().to_msg()
        self.publisher.publish(msg)

    def _replay_once(self) -> None:
        now_ns = time.monotonic_ns()
        elapsed_since_last_tick_ns = max(now_ns - self._last_tick_ns, 0)
        self._last_tick_ns = now_ns

        if self._paused:
            return

        if self._frame_offsets_ns:
            if self._playback_elapsed_ns < self._frame_offsets_ns[-1]:
                self._playback_elapsed_ns = min(
                    self._playback_elapsed_ns + elapsed_since_last_tick_ns,
                    self._frame_offsets_ns[-1],
                )
        else:
            if self._current_index >= len(self._frames) - 1:
                return
            self._current_index += 1

        self._publish_current_frame()

    def _handle_key(self, key: str) -> None:
        if key == ' ':
            self._paused = not self._paused
            # 恢复播放时重置 tick 基准，避免暂停期间累计的墙钟时间一次性灌进回放时间轴。
            self._last_tick_ns = time.monotonic_ns()
            state = 'PAUSED' if self._paused else 'RUNNING'
            self.get_logger().info(green_text(f'Replay state -> {state}'))
            return

        if key.lower() == 'q':
            self.get_logger().info('Quit requested from keyboard.')
            rclpy.shutdown()

    def _poll_keypress(self) -> None:
        if self._input_stream is None or self._input_fd is None:
            return

        readable, _, _ = select.select([self._input_fd], [], [], 0.0)
        if not readable:
            return

        try:
            key = self._input_stream.read(1)
        except OSError as exc:
            self.get_logger().warning(f'Keyboard read failed: {exc}')
            return

        if key:
            self._handle_key(key)

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
    try:
        node = JearmJsonlReplayerNode()
    except Exception as exc:
        print(f'Failed to create replayer node: {exc}')
        if rclpy.ok():
            rclpy.shutdown()
        return

    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        try:
            node.destroy_node()
        except KeyboardInterrupt:
            pass
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
