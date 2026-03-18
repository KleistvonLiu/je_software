#!/usr/bin/env python3
"""统一 ZMQ 机器人网关节点。

本节点统一承接两类职责：
1. 向 `jeserver.cpp` 发送 `MoveA/MoveL` 指令
2. 从 `jeserver.cpp` 读取 `State`，并以 ROS topic / service 暴露最新机器人状态

这让上层任务节点不需要直接碰 ZMQ 协议，只需要消费稳定的 ROS 接口。
"""

from __future__ import annotations

import copy
import json
import threading
import time
from typing import Any

import rclpy
from rclpy.action import ActionServer
from rclpy.action import CancelResponse
from rclpy.action import GoalResponse
from rclpy.node import Node
from rclpy.parameter import Parameter

from je_software.action import ExecuteMotionSequence
from je_software.msg import MotionStep
from je_software.msg import RobotState
from je_software.pcb_process_common import GRIPPER
from je_software.pcb_process_common import MOVEJ
from je_software.pcb_process_common import MOVEL
from je_software.pcb_process_common import WAIT
from je_software.pcb_process_common import make_pose_stamped
from je_software.pcb_process_common import pose_to_rpy_list
from je_software.srv import GetRobotState

try:
    import zmq
except ModuleNotFoundError:  # pragma: no cover - validated at runtime
    zmq = None


STATE_PREFIX = 'State '
STATE_SOURCE = 'zmq_state'
STATE_FRAME_ID = 'base_link'


def _robot_key(robot_id: int) -> str:
    """返回 jeserver 使用的机器人 key。"""
    return f'Robot{int(robot_id)}'


def _float_list(values: list[Any]) -> list[float]:
    """把 JSON 数组统一转成 float 列表。"""
    return [float(value) for value in values]


def parse_state_message(message: str) -> dict[str, Any] | None:
    """解析一条 `State <json>` 文本消息。"""
    if not message.startswith(STATE_PREFIX):
        return None

    try:
        payload = json.loads(message[len(STATE_PREFIX) :])
    except json.JSONDecodeError:
        return None

    if not isinstance(payload, dict):
        return None
    return payload


def build_robot_state_from_json(
    state_json: dict[str, Any],
    robot_id: int,
    *,
    frame_id: str = STATE_FRAME_ID,
    source: str = STATE_SOURCE,
    stamp=None,
) -> RobotState:
    """从 jeserver 的 State JSON 构造 RobotState 消息。"""
    state = RobotState()
    state.robot_id = int(robot_id)
    state.source = str(source)
    state.header.frame_id = str(frame_id)
    state.tcp_pose.header.frame_id = str(frame_id)
    if stamp is not None:
        state.header.stamp = stamp
        state.tcp_pose.header.stamp = stamp

    robot = state_json.get(_robot_key(robot_id))
    if not isinstance(robot, dict):
        return state

    joints = robot.get('Joint')
    if isinstance(joints, list) and len(joints) >= 7:
        state.joint_valid = True
        state.joint_position = _float_list(joints)

        joint_velocity = robot.get('JointVelocity')
        if isinstance(joint_velocity, list) and len(joint_velocity) >= 7:
            state.joint_velocity = _float_list(joint_velocity)

        joint_effort = robot.get('JointSensorTorque')
        if isinstance(joint_effort, list) and len(joint_effort) >= 7:
            state.joint_effort = _float_list(joint_effort)

    cartesian = robot.get('Cartesian')
    if isinstance(cartesian, list) and len(cartesian) >= 6:
        state.cartesian_valid = True
        state.tcp_pose = make_pose_stamped(cartesian[:6], frame_id)
        if stamp is not None:
            state.tcp_pose.header.stamp = stamp

    state.valid = bool(state.joint_valid or state.cartesian_valid)
    return state


def clone_robot_state(state: RobotState) -> RobotState:
    """返回一份 RobotState 深拷贝，避免共享缓存对象被意外修改。"""
    return copy.deepcopy(state)


def _build_movel_payload(step: MotionStep, robot_id: int) -> str:
    """把笛卡尔 MotionStep 转成 jeserver 兼容的 MoveL。"""
    pose = pose_to_rpy_list(step.target_pose)
    tra_speed = float(step.velocity)
    rot_speed = float(step.acceleration)
    if rot_speed <= 0.0:
        rot_speed = tra_speed

    payload = {
        _robot_key(robot_id): {
            'cartesian': pose,
            'rot_speed': rot_speed,
            'tra_speed': tra_speed,
        }
    }
    return f'MoveL {json.dumps(payload, separators=(",", ":"))}'


def _build_movea_payload(
    step: MotionStep,
    robot_id: int,
    joint_target: list[float],
) -> str:
    """把关节 MotionStep 转成 jeserver 兼容的 MoveA。"""
    payload = {
        _robot_key(robot_id): {
            'joint': [float(value) for value in joint_target],
            'speed': float(step.velocity),
        }
    }
    return f'MoveA {json.dumps(payload, separators=(",", ":"))}'


def motion_step_to_payload(
    step: MotionStep,
    robot_id: int,
    *,
    movea_joint_targets: dict[str, list[float]] | None = None,
) -> str | None:
    """把 MotionStep 转成发送给 jeserver 的 ZMQ 文本。"""
    if step.command_type == WAIT:
        return None
    if step.command_type == MOVEL:
        return _build_movel_payload(step, robot_id)
    if step.command_type == MOVEJ:
        joint_target = [float(value) for value in step.joint_target]
        if joint_target:
            return _build_movea_payload(step, robot_id, joint_target)

        joint_targets = movea_joint_targets or {}
        joint_target = joint_targets.get(step.name)
        if not joint_target:
            raise ValueError(
                'MOVEJ step '
                f'"{step.name}" is missing step.joint_target and '
                f'movea_joint_targets.{step.name}'
            )
        return _build_movea_payload(step, robot_id, joint_target)
    if step.command_type == GRIPPER:
        payload = {
            'robot_id': int(robot_id),
            'command': step.gripper_command,
        }
        return f'Gripper {json.dumps(payload, separators=(",", ":"))}'
    raise ValueError(f'Unsupported command_type: {step.command_type}')


class ZmqMotionBackendNode(Node):
    """统一负责运动下发和机器人状态读取。"""

    def __init__(self) -> None:
        super().__init__('zmq_motion_backend_node')
        if zmq is None:
            raise RuntimeError(
                'pyzmq is required for zmq_motion_backend_node.'
            )

        self.declare_parameter(
            'action_name',
            '/motion_backend/execute_motion_sequence',
        )
        self.declare_parameter('command_endpoint', 'tcp://*:8001')
        self.declare_parameter('socket_mode', 'bind')
        self.declare_parameter('robot_id', 0)
        self.declare_parameter('send_timeout_ms', 1000)
        self.declare_parameter('linger_ms', 0)
        self.declare_parameter('startup_delay_sec', 0.2)
        self.declare_parameter('state_endpoint', 'tcp://192.168.0.99:8000')
        self.declare_parameter('state_socket_mode', 'connect')
        self.declare_parameter('state_recv_timeout_ms', 100)
        self.declare_parameter('state_poll_hz', 20.0)
        self.declare_parameter('state_topic', '/motion_backend/robot_state')
        self.declare_parameter(
            'state_service_name',
            '/motion_backend/get_robot_state',
        )
        # 空数组在 rclpy 里会被推断成 BYTE_ARRAY。
        # 这里这些参数实际都应该是 double[] 关节目标。
        self.declare_parameter(
            'movea_joint_targets.pre_grasp',
            Parameter.Type.DOUBLE_ARRAY,
        )
        self.declare_parameter(
            'movea_joint_targets.inspection_pre',
            Parameter.Type.DOUBLE_ARRAY,
        )
        self.declare_parameter(
            'movea_joint_targets.pre_place',
            Parameter.Type.DOUBLE_ARRAY,
        )
        self.declare_parameter(
            'movea_joint_targets.home',
            Parameter.Type.DOUBLE_ARRAY,
        )

        self.robot_id = int(self.get_parameter('robot_id').value)
        self._movea_joint_targets = self._load_movea_joint_targets()
        self._publish_period_sec = 0.0
        state_poll_hz = float(self.get_parameter('state_poll_hz').value)
        if state_poll_hz > 0.0:
            self._publish_period_sec = 1.0 / state_poll_hz

        self._context = zmq.Context()
        self._command_socket = self._create_command_socket()
        self._state_socket = self._create_state_socket()

        self._state_lock = threading.Lock()
        self._latest_robot_state: RobotState | None = None
        self._state_thread_running = True
        self._last_state_publish_time = 0.0

        self.state_publisher = self.create_publisher(
            RobotState,
            str(self.get_parameter('state_topic').value),
            10,
        )
        self.state_service = self.create_service(
            GetRobotState,
            str(self.get_parameter('state_service_name').value),
            self._handle_get_robot_state,
        )

        startup_delay_sec = float(
            self.get_parameter('startup_delay_sec').value
        )
        if startup_delay_sec > 0.0:
            time.sleep(startup_delay_sec)

        self._state_thread = threading.Thread(
            target=self._state_reader_loop,
            name='zmq_state_reader',
            daemon=True,
        )
        self._state_thread.start()

        self._action_server = ActionServer(
            self,
            ExecuteMotionSequence,
            str(self.get_parameter('action_name').value),
            execute_callback=self._execute_callback,
            goal_callback=self._goal_callback,
            cancel_callback=self._cancel_callback,
        )
        self.get_logger().info(
            'ZMQ gateway ready: '
            f'command={self.get_parameter("socket_mode").value}:'
            f'{self.get_parameter("command_endpoint").value}, '
            f'state={self.get_parameter("state_socket_mode").value}:'
            f'{self.get_parameter("state_endpoint").value}'
        )

    def _create_command_socket(self):
        socket = self._context.socket(zmq.PUB)
        socket.setsockopt(zmq.SNDHWM, 0)
        socket.setsockopt(zmq.IMMEDIATE, 1)
        socket.setsockopt(
            zmq.SNDTIMEO,
            int(self.get_parameter('send_timeout_ms').value),
        )
        socket.setsockopt(
            zmq.LINGER,
            int(self.get_parameter('linger_ms').value),
        )

        endpoint = str(self.get_parameter('command_endpoint').value)
        mode = str(self.get_parameter('socket_mode').value).strip().lower()
        if mode == 'connect':
            socket.connect(endpoint)
        else:
            socket.bind(endpoint)
        return socket

    def _create_state_socket(self):
        socket = self._context.socket(zmq.SUB)
        socket.setsockopt(zmq.RCVHWM, 1)
        socket.setsockopt(zmq.CONFLATE, 1)
        socket.setsockopt(
            zmq.RCVTIMEO,
            int(self.get_parameter('state_recv_timeout_ms').value),
        )
        socket.setsockopt(
            zmq.LINGER,
            int(self.get_parameter('linger_ms').value),
        )
        socket.setsockopt_string(zmq.SUBSCRIBE, STATE_PREFIX)

        endpoint = str(self.get_parameter('state_endpoint').value)
        mode = str(self.get_parameter('state_socket_mode').value).strip().lower()
        if mode == 'bind':
            socket.bind(endpoint)
        else:
            socket.connect(endpoint)
        return socket

    def _load_movea_joint_targets(self) -> dict[str, list[float]]:
        """读取 YAML 里配置的固定 MoveA 关节目标。"""
        targets: dict[str, list[float]] = {}
        for name, parameter in self.get_parameters_by_prefix(
            'movea_joint_targets'
        ).items():
            values = parameter.value
            if values is None:
                continue
            normalized = [float(value) for value in values]
            if normalized:
                targets[name] = normalized
        return targets

    def _state_reader_loop(self) -> None:
        """后台线程持续接收 `State`，更新缓存并节流发布 topic。"""
        while self._state_thread_running:
            try:
                message = self._state_socket.recv_string()
            except zmq.Again:
                continue
            except zmq.ZMQError:
                if not self._state_thread_running:
                    break
                continue
            except Exception as exc:  # pragma: no cover - defensive
                self.get_logger().error(f'State recv failed: {exc}')
                continue

            state_json = parse_state_message(message)
            if state_json is None:
                continue

            state = build_robot_state_from_json(
                state_json,
                self.robot_id,
                frame_id=STATE_FRAME_ID,
                source=STATE_SOURCE,
                stamp=self.get_clock().now().to_msg(),
            )
            if not state.valid:
                continue

            with self._state_lock:
                self._latest_robot_state = clone_robot_state(state)

            now = time.monotonic()
            if (
                self._publish_period_sec <= 0.0
                or now - self._last_state_publish_time >= self._publish_period_sec
            ):
                self.state_publisher.publish(state)
                self._last_state_publish_time = now

    def _handle_get_robot_state(self, _request, response):
        """返回当前缓存的最新机器人状态快照。"""
        with self._state_lock:
            state = (
                clone_robot_state(self._latest_robot_state)
                if self._latest_robot_state is not None
                else None
            )

        if state is None:
            response.success = False
            response.reason = 'robot_state_unavailable'
            return response

        response.success = True
        response.reason = ''
        response.state = state
        return response

    def _goal_callback(
        self,
        goal_request: ExecuteMotionSequence.Goal,
    ) -> GoalResponse:
        if not goal_request.steps:
            self.get_logger().warn('Rejecting empty motion sequence goal.')
            return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    def _cancel_callback(self, _goal_handle) -> CancelResponse:
        return CancelResponse.ACCEPT

    def _execute_callback(self, goal_handle):
        result = ExecuteMotionSequence.Result()
        feedback = ExecuteMotionSequence.Feedback()
        try:
            for index, step in enumerate(goal_handle.request.steps):
                if goal_handle.is_cancel_requested:
                    goal_handle.canceled()
                    result.success = False
                    result.error_code = 'goal_canceled'
                    return result

                feedback.stage = goal_handle.request.sequence_name
                feedback.step_index = index
                feedback.active_step_name = step.name
                goal_handle.publish_feedback(feedback)

                payload = motion_step_to_payload(
                    step,
                    self.robot_id,
                    movea_joint_targets=self._movea_joint_targets,
                )
                if payload is not None:
                    self._command_socket.send_string(payload)
                    self.get_logger().info(
                        f'[{goal_handle.request.sequence_name}] '
                        f'sent {step.command_type}:{step.name}'
                    )

                dwell_sec = max(float(step.dwell_sec), 0.0)
                if dwell_sec > 0.0:
                    time.sleep(dwell_sec)

            goal_handle.succeed()
            result.success = True
            result.error_code = ''
            return result
        except Exception as exc:  # pragma: no cover - exercised in integration
            goal_handle.abort()
            result.success = False
            result.error_code = str(exc)
            self.get_logger().error(f'ExecuteMotionSequence failed: {exc}')
            return result

    def destroy_node(self) -> bool:
        self._state_thread_running = False
        if hasattr(self, '_state_thread'):
            self._state_thread.join(timeout=1.0)
        self._action_server.destroy()
        self._command_socket.close(0)
        self._state_socket.close(0)
        self._context.term()
        return super().destroy_node()


def main() -> None:
    rclpy.init()
    node = ZmqMotionBackendNode()
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
