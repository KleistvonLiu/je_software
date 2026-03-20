#!/usr/bin/env python3
"""使用 MoveIt2 把语义动作解析为可执行关节轨迹。

当前 PCB demo 的上层状态机仍然只关心：
- pre_grasp / grasp / inspection / place / home 这些“语义动作”

真正下发给 jeserver.cpp 之前，需要先把这些语义动作落到关节空间：
- MOVEJ + target_pose: 通过 `/compute_ik` 求一个关节解
- MOVEL + target_pose: 通过 `/compute_cartesian_path` 求一段笛卡尔直线轨迹，
  再展开成多条 MoveA 可执行的关节路点

这样上层状态机不需要理解 MoveIt2 消息细节，执行层也不需要直接支持
MoveIt 的 RobotTrajectory。
"""

from __future__ import annotations

import copy
import importlib
import math
import threading
import time
from typing import Any

import rclpy
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException
from rclpy.executors import SingleThreadedExecutor

from je_software.msg import MotionStep
from .pcb_process_common import GRIPPER
from .pcb_process_common import MOVEJ
from .pcb_process_common import MOVEL
from .pcb_process_common import WAIT
from .pcb_process_common import pose_to_rpy_list
from .pcb_process_common import require_joint_list
from je_software.srv import GetRobotState

DEFAULT_JOINT_NAMES = [
    'joint1',
    'joint2',
    'joint3',
    'joint4',
    'joint5',
    'joint6',
    'joint7',
]
MOVEIT_SUCCESS = 1
JOINT_EQUAL_TOLERANCE = 1e-9


def _format_values(values) -> str:
    return '[' + ', '.join(f'{float(value):.6f}' for value in values) + ']'


def _duration_to_sec(duration_msg) -> float:
    return float(duration_msg.sec) + float(duration_msg.nanosec) / 1e9


def _moveit_error_ok(error_code) -> bool:
    if error_code is None:
        return True
    success_value = getattr(error_code, 'SUCCESS', MOVEIT_SUCCESS)
    return int(getattr(error_code, 'val', success_value)) == int(success_value)


def _joint_lists_equal(left: list[float], right: list[float]) -> bool:
    if len(left) != len(right):
        return False
    return all(
        abs(float(left_value) - float(right_value)) <= JOINT_EQUAL_TOLERANCE
        for left_value, right_value in zip(left, right)
    )


def _reorder_joint_positions(
    joint_names: list[str],
    positions: list[float],
    *,
    expected_names: list[str],
) -> list[float]:
    """把 MoveIt 返回的关节结果统一重排成 joint1..joint7 顺序。"""
    normalized_positions = [float(value) for value in positions]
    if joint_names and len(joint_names) == len(normalized_positions):
        index_by_name = {
            str(name): index
            for index, name in enumerate(joint_names)
        }
        missing = [
            name
            for name in expected_names
            if name not in index_by_name
        ]
        if not missing:
            return [
                normalized_positions[index_by_name[name]]
                for name in expected_names
            ]

    if len(normalized_positions) >= len(expected_names):
        return normalized_positions[: len(expected_names)]

    raise ValueError(
        'joint_position_length_mismatch: '
        f'expected>={len(expected_names)}, actual={len(normalized_positions)}'
    )


def _normalize_joint_positions_near_reference(
    joint_positions: list[float],
    reference_positions: list[float],
) -> list[float]:
    """把关节角归一到最接近参考点的那一圈。

    当前 JEARM 的关节上限非常大，MoveIt 常会返回等价但多转若干圈的解。
    例如 -7.36 rad 与 -1.08 rad 对连续关节是同一姿态，但直接发给机器人会
    走很多圈。这里把每个关节都拉回到“最接近 seed / 上一个路点”的等价角。
    """
    if len(joint_positions) != len(reference_positions):
        return [float(value) for value in joint_positions]

    normalized: list[float] = []
    for value, reference in zip(joint_positions, reference_positions):
        value = float(value)
        reference = float(reference)
        wrap_count = round((reference - value) / math.tau)
        normalized.append(value + wrap_count * math.tau)
    return normalized


class MoveItMotionResolver:
    """把语义 MotionStep 解析成可执行关节 MotionStep。"""

    def __init__(
        self,
        *,
        node,
        robot_state_service_name: str,
        moveit_ik_service: str,
        moveit_cartesian_service: str,
        moveit_group_name: str,
        moveit_tip_link: str,
        moveit_ik_timeout_sec: float,
        moveit_ik_attempts: int,
        moveit_cartesian_max_step: float,
        moveit_cartesian_jump_threshold: float,
        moveit_avoid_collisions: bool,
        robot_state_timeout_sec: float,
        joint_names: list[str] | None = None,
    ) -> None:
        self._node = node
        self._logger = node.get_logger()
        self._moveit_group_name = str(moveit_group_name)
        self._moveit_tip_link = str(moveit_tip_link)
        self._moveit_ik_timeout_sec = float(moveit_ik_timeout_sec)
        self._moveit_ik_attempts = max(int(moveit_ik_attempts), 1)
        self._moveit_cartesian_max_step = float(moveit_cartesian_max_step)
        self._moveit_cartesian_jump_threshold = float(
            moveit_cartesian_jump_threshold
        )
        self._moveit_avoid_collisions = bool(moveit_avoid_collisions)
        self._robot_state_timeout_sec = float(robot_state_timeout_sec)
        self._joint_names = list(joint_names or DEFAULT_JOINT_NAMES)
        self._shutdown = False

        try:
            self._moveit_srv = importlib.import_module('moveit_msgs.srv')
            self._moveit_msg = importlib.import_module('moveit_msgs.msg')
            self._sensor_msgs = importlib.import_module('sensor_msgs.msg')
        except ModuleNotFoundError as exc:  # pragma: no cover - runtime guarded
            raise RuntimeError(
                'MoveIt2 Python interfaces are unavailable. '
                'Please install ROS Humble MoveIt2 packages before running the PCB demo.'
            ) from exc

        # 这里单独拉一个 helper node，并在后台线程里持续 spin。
        # 原因是任务管理器会在订阅/状态机回调里同步等待 service 结果；
        # 如果直接复用 task_manager 自己的 executor，就会出现“在回调里等自己”的死锁，
        # 最终表现为 robot_state_request_timeout / moveit_*_timeout。
        helper_name = f'{node.get_name()}_moveit_helper'
        self._helper_node = rclpy.create_node(helper_name)
        self._helper_executor = SingleThreadedExecutor()
        self._helper_executor.add_node(self._helper_node)
        self._helper_spin_thread = threading.Thread(
            target=self._spin_helper_executor,
            name=f'{helper_name}_executor',
            daemon=True,
        )
        self._helper_spin_thread.start()

        self._robot_state_client = self._helper_node.create_client(
            GetRobotState,
            str(robot_state_service_name),
        )
        self._ik_client = self._helper_node.create_client(
            self._moveit_srv.GetPositionIK,
            str(moveit_ik_service),
        )
        self._cartesian_client = self._helper_node.create_client(
            self._moveit_srv.GetCartesianPath,
            str(moveit_cartesian_service),
        )

    def _spin_helper_executor(self) -> None:
        """后台 spin helper executor，关闭时吞掉预期的 ExternalShutdownException。"""
        try:
            self._helper_executor.spin()
        except ExternalShutdownException:
            pass

    def shutdown(self) -> None:
        """释放 helper node 和后台 executor 线程。"""
        if self._shutdown:
            return
        self._shutdown = True

        if getattr(self, '_helper_executor', None) is not None:
            self._helper_executor.shutdown()
        if getattr(self, '_helper_spin_thread', None) is not None:
            self._helper_spin_thread.join(timeout=1.0)
        if getattr(self, '_helper_node', None) is not None:
            self._helper_node.destroy_node()

    def _wait_for_future_result(
        self,
        future,
        *,
        timeout_sec: float,
        timeout_error: str,
    ):
        """等待 helper node 上的异步 future 完成。

        helper node 已经在独立 executor 线程里 spin，所以这里不再对 task_manager
        自己的 node 做 spin_until_future_complete，避免回调重入死锁。
        """
        deadline = time.monotonic() + max(float(timeout_sec), 0.0)
        while time.monotonic() <= deadline:
            if future.done():
                return future.result()
            time.sleep(0.01)

        if future.done():
            return future.result()
        raise RuntimeError(timeout_error)

    def resolve_steps(self, steps) -> list[MotionStep]:
        """把一组语义步骤解析为最终可执行步骤。"""
        resolved_steps: list[MotionStep] = []
        if not self._requires_moveit_resolution(steps):
            return [copy.deepcopy(step) for step in steps]

        current_seed = self._get_current_joint_seed()
        for step in steps:
            if step.command_type in (GRIPPER, WAIT):
                resolved_steps.append(copy.deepcopy(step))
                continue

            if step.command_type == MOVEJ:
                explicit_joint_target = [float(value) for value in step.joint_target]
                if explicit_joint_target:
                    current_seed = require_joint_list(
                        explicit_joint_target,
                        f'{step.name}.joint_target',
                    )
                    resolved_steps.append(copy.deepcopy(step))
                    continue

                resolved_joint_target = self._compute_ik_for_step(
                    step,
                    current_seed,
                )
                resolved_step = copy.deepcopy(step)
                resolved_step.joint_target = resolved_joint_target
                resolved_steps.append(resolved_step)
                current_seed = resolved_joint_target
                continue

            if step.command_type == MOVEL:
                cartesian_steps = self._compute_cartesian_steps(
                    step,
                    current_seed,
                )
                if not cartesian_steps:
                    raise RuntimeError(
                        f'cartesian_path_empty:{step.name}'
                    )
                resolved_steps.extend(cartesian_steps)
                current_seed = [float(value) for value in cartesian_steps[-1].joint_target]
                continue

            raise RuntimeError(
                f'unsupported_motion_step_for_moveit:{step.command_type}'
            )

        return resolved_steps

    def _requires_moveit_resolution(self, steps) -> bool:
        for step in steps:
            if step.command_type == MOVEL:
                return True
            if step.command_type == MOVEJ and not list(step.joint_target):
                return True
        return False

    def _get_current_joint_seed(self) -> list[float]:
        """从统一 ZMQ 网关读取当前关节状态，作为 MoveIt seed。"""
        if not self._robot_state_client.wait_for_service(
            timeout_sec=self._robot_state_timeout_sec
        ):
            raise RuntimeError('robot_state_service_unavailable')

        future = self._robot_state_client.call_async(GetRobotState.Request())
        response = self._wait_for_future_result(
            future,
            timeout_sec=self._robot_state_timeout_sec,
            timeout_error='robot_state_request_timeout',
        )
        if not response.success:
            raise RuntimeError(f'robot_state_request_failed:{response.reason}')
        if not response.state.valid or not response.state.joint_valid:
            raise RuntimeError('robot_state_invalid')

        joint_positions = list(response.state.joint_position)
        seed = _reorder_joint_positions(
            [],
            joint_positions,
            expected_names=self._joint_names,
        )
        self._logger.info(
            'Using current robot state as MoveIt seed: '
            f'{_format_values(seed)}'
        )
        return seed

    def _make_moveit_robot_state(self, joint_positions: list[float]):
        robot_state = self._moveit_msg.RobotState()
        joint_state = self._sensor_msgs.JointState()
        joint_state.header.stamp = self._node.get_clock().now().to_msg()
        joint_state.header.frame_id = 'base_link'
        joint_state.name = list(self._joint_names)
        joint_state.position = [float(value) for value in joint_positions]
        robot_state.joint_state = joint_state
        return robot_state

    def _compute_ik_for_step(
        self,
        step: MotionStep,
        seed_joint_positions: list[float],
    ) -> list[float]:
        if not self._ik_client.wait_for_service(
            timeout_sec=self._moveit_ik_timeout_sec
        ):
            raise RuntimeError('moveit_ik_service_unavailable')

        request = self._moveit_srv.GetPositionIK.Request()
        request.ik_request.group_name = self._moveit_group_name
        request.ik_request.robot_state = self._make_moveit_robot_state(
            seed_joint_positions
        )
        request.ik_request.pose_stamped = step.target_pose
        request.ik_request.ik_link_name = self._moveit_tip_link
        if hasattr(request.ik_request, 'attempts'):
            request.ik_request.attempts = self._moveit_ik_attempts
        if hasattr(request.ik_request, 'timeout'):
            request.ik_request.timeout = Duration(
                seconds=max(self._moveit_ik_timeout_sec, 0.0)
            ).to_msg()
        if hasattr(request.ik_request, 'avoid_collisions'):
            request.ik_request.avoid_collisions = self._moveit_avoid_collisions

        future = self._ik_client.call_async(request)
        response = self._wait_for_future_result(
            future,
            timeout_sec=self._moveit_ik_timeout_sec,
            timeout_error=f'moveit_ik_timeout:{step.name}',
        )
        if not _moveit_error_ok(getattr(response, 'error_code', None)):
            error_code = getattr(getattr(response, 'error_code', None), 'val', None)
            raise RuntimeError(f'moveit_ik_failed:{step.name}:{error_code}')

        solution_joint_state = response.solution.joint_state
        resolved_joint_target = _reorder_joint_positions(
            list(solution_joint_state.name),
            list(solution_joint_state.position),
            expected_names=self._joint_names,
        )
        resolved_joint_target = _normalize_joint_positions_near_reference(
            resolved_joint_target,
            seed_joint_positions,
        )
        self._logger.info(
            f'MoveIt IK resolved {step.name}: '
            f'target_pose={_format_values(pose_to_rpy_list(step.target_pose))}, '
            f'joint_target={_format_values(resolved_joint_target)}'
        )
        return resolved_joint_target

    def _compute_cartesian_steps(
        self,
        step: MotionStep,
        seed_joint_positions: list[float],
    ) -> list[MotionStep]:
        if not self._cartesian_client.wait_for_service(
            timeout_sec=self._moveit_ik_timeout_sec
        ):
            raise RuntimeError('moveit_cartesian_service_unavailable')

        request = self._moveit_srv.GetCartesianPath.Request()
        request.header.stamp = self._node.get_clock().now().to_msg()
        request.header.frame_id = (
            step.target_pose.header.frame_id or 'base_link'
        )
        request.start_state = self._make_moveit_robot_state(seed_joint_positions)
        request.group_name = self._moveit_group_name
        request.link_name = self._moveit_tip_link
        request.waypoints = [copy.deepcopy(step.target_pose.pose)]
        request.max_step = float(self._moveit_cartesian_max_step)
        request.jump_threshold = float(self._moveit_cartesian_jump_threshold)
        request.avoid_collisions = self._moveit_avoid_collisions

        # 不同 MoveIt2 版本的 GetCartesianPath 请求字段略有差异。
        # 这里按“有这个字段就设置”的方式做兼容，避免强绑某一个小版本。
        if hasattr(request, 'max_cartesian_speed'):
            request.max_cartesian_speed = max(float(step.velocity), 0.0)
        if hasattr(request, 'cartesian_speed_limited_link'):
            request.cartesian_speed_limited_link = self._moveit_tip_link
        if hasattr(request, 'prismatic_jump_threshold'):
            request.prismatic_jump_threshold = 0.0
        if hasattr(request, 'revolute_jump_threshold'):
            request.revolute_jump_threshold = 0.0

        future = self._cartesian_client.call_async(request)
        response = self._wait_for_future_result(
            future,
            timeout_sec=self._moveit_ik_timeout_sec,
            timeout_error=f'moveit_cartesian_timeout:{step.name}',
        )
        if not _moveit_error_ok(getattr(response, 'error_code', None)):
            error_code = getattr(getattr(response, 'error_code', None), 'val', None)
            raise RuntimeError(
                f'moveit_cartesian_failed:{step.name}:{error_code}'
            )

        fraction = float(getattr(response, 'fraction', 0.0))
        if fraction < 0.999:
            raise RuntimeError(
                f'moveit_cartesian_incomplete:{step.name}:fraction={fraction:.6f}'
            )

        trajectory = response.solution.joint_trajectory
        point_count = len(trajectory.points)
        if point_count == 0:
            raise RuntimeError(f'moveit_cartesian_empty:{step.name}')

        reordered_points: list[list[float]] = []
        reference_positions = list(seed_joint_positions)
        for point in trajectory.points:
            reordered_joint_positions = _reorder_joint_positions(
                list(trajectory.joint_names),
                list(point.positions),
                expected_names=self._joint_names,
            )
            reordered_joint_positions = _normalize_joint_positions_near_reference(
                reordered_joint_positions,
                reference_positions,
            )
            reordered_points.append(reordered_joint_positions)
            reference_positions = reordered_joint_positions

        kept_points: list[tuple[list[float], float]] = []
        previous_positions = list(seed_joint_positions)
        previous_time_sec = 0.0
        for point, joint_positions in zip(trajectory.points, reordered_points):
            if _joint_lists_equal(joint_positions, previous_positions):
                previous_time_sec = _duration_to_sec(point.time_from_start)
                continue

            current_time_sec = _duration_to_sec(point.time_from_start)
            dwell_sec = max(current_time_sec - previous_time_sec, 0.0)
            kept_points.append((joint_positions, dwell_sec))
            previous_positions = joint_positions
            previous_time_sec = current_time_sec

        if not kept_points:
            raise RuntimeError(f'moveit_cartesian_empty_after_dedup:{step.name}')

        if all(dwell_sec <= 0.0 for _, dwell_sec in kept_points):
            fallback_dwell_sec = max(float(step.dwell_sec), 0.0)
            each_dwell_sec = (
                fallback_dwell_sec / len(kept_points)
                if kept_points
                else 0.0
            )
            kept_points = [
                (joint_positions, each_dwell_sec)
                for joint_positions, _ in kept_points
            ]

        resolved_steps: list[MotionStep] = []
        for index, (joint_positions, dwell_sec) in enumerate(
            kept_points,
            start=1,
        ):
            resolved_step = MotionStep()
            resolved_step.name = f'{step.name}_{index:04d}'
            resolved_step.command_type = MOVEJ
            resolved_step.joint_target = [float(value) for value in joint_positions]
            resolved_step.velocity = float(step.velocity)
            resolved_step.acceleration = float(step.acceleration)
            resolved_step.blend_radius = float(step.blend_radius)
            resolved_step.dwell_sec = float(dwell_sec)
            resolved_steps.append(resolved_step)

        self._logger.info(
            f'MoveIt Cartesian resolved {step.name}: '
            f'points={len(resolved_steps)}, '
            f'final_joint_target={_format_values(resolved_steps[-1].joint_target)}'
        )
        return resolved_steps
