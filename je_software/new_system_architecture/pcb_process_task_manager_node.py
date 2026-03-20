#!/usr/bin/env python3
"""固定点 PCB 流程任务管理器。

这个节点是整套 demo 的“大脑”，只负责流程编排，不直接做感知和运动：
1. 监听 PCB 到位信号
2. 在合适时机请求固定抓取位姿
3. 调用动作 action 执行抓取/送检/放置/回 home
4. 触发检测并等待异步结果
5. 在异常时尝试回 home 做最小恢复

这版是典型的“状态机调度 + service 查询 + action 执行”结构。
"""

from __future__ import annotations

import json
import threading
from enum import Enum
from functools import partial

import rclpy
from action_msgs.msg import GoalStatus
from rclpy.action import ActionServer
from rclpy.action import ActionClient
from rclpy.action import CancelResponse
from rclpy.action import GoalResponse
from rclpy.node import Node
from rclpy.parameter import Parameter

from je_software.action import ExecuteMotionSequence
from je_software.action import RecoverToInitial
from je_software.msg import InspectionResult
from je_software.msg import PcbPresence
from .moveit_motion_resolver import MoveItMotionResolver
from .pcb_process_common import build_joint_trajectory_sequence
from .pcb_process_common import build_home_sequence
from .pcb_process_common import build_inspection_sequence
from .pcb_process_common import build_pick_sequence
from .pcb_process_common import build_place_sequence
from .pcb_process_common import build_recover_to_initial_sequence
from .pcb_process_common import pose_to_rpy_list
from .pcb_process_common import require_joint_list
from .pcb_process_common import require_pose_list
from je_software.srv import GetAvailableSlot
from je_software.srv import GetPcbPickPose
from je_software.srv import GetRobotState
from je_software.srv import TriggerInspection

ANSI_GREEN = '\033[32m'
ANSI_RESET = '\033[0m'


def green_text(text: str) -> str:
    return f'{ANSI_GREEN}{text}{ANSI_RESET}'


class ProcessState(str, Enum):
    """PCB 闭环 demo 的有限状态机状态定义。"""

    IDLE = 'IDLE'
    WAIT_INIT_JOINT_STATE = 'WAIT_INIT_JOINT_STATE'
    EXECUTE_INITIALIZATION = 'EXECUTE_INITIALIZATION'
    WAIT_PCB = 'WAIT_PCB'
    MOVE_HOME_BEFORE_PICK = 'MOVE_HOME_BEFORE_PICK'
    REQUEST_PICK_POSE = 'REQUEST_PICK_POSE'
    EXECUTE_PICK = 'EXECUTE_PICK'
    MOVE_TO_INSPECTION = 'MOVE_TO_INSPECTION'
    TRIGGER_INSPECTION = 'TRIGGER_INSPECTION'
    WAIT_INSPECTION_RESULT = 'WAIT_INSPECTION_RESULT'
    REQUEST_GOOD_SLOT = 'REQUEST_GOOD_SLOT'
    EXECUTE_PLACE = 'EXECUTE_PLACE'
    GO_HOME = 'GO_HOME'
    EXECUTE_MANUAL_RECOVERY = 'EXECUTE_MANUAL_RECOVERY'
    ERROR = 'ERROR'


class PcbProcessTaskManagerNode(Node):
    """串联固定 PCB 流程，从到位信号一路跑到回 home。"""

    def __init__(self) -> None:
        super().__init__('pcb_process_task_manager_node')
        self._declare_parameters()

        # 先把所有流程相关参数一次性读出来。
        # 这批参数来自统一 YAML，方便后续现场调试和换点位。
        self.frame_id = str(self.get_parameter('frame_id').value)
        self.enable_initialization = bool(
            self.get_parameter('enable_initialization').value
        )
        self.robot_state_service_name = str(
            self.get_parameter('robot_state_service').value
        )
        self.initial_state_poll_sec = float(
            self.get_parameter('initial_state_poll_sec').value
        )
        self.initial_joint_position = require_joint_list(
            self.get_parameter('initial_joint_position').value,
            'initial_joint_position',
        )
        self.initial_joint_tolerance = float(
            self.get_parameter('initial_joint_tolerance').value
        )
        self.initial_joint_state_timeout_sec = float(
            self.get_parameter('initial_joint_state_timeout_sec').value
        )
        self.init_trajectory_points = self._parse_joint_waypoint_strings(
            self.get_parameter_or(
                'init_trajectory_points',
                Parameter(
                    'init_trajectory_points',
                    type_=Parameter.Type.STRING_ARRAY,
                    value=[],
                ),
            ).value
        )
        self.home_pose = require_pose_list(
            self.get_parameter('home_pose').value,
            'home_pose',
        )
        self.inspection_pre_pose = require_pose_list(
            self.get_parameter('inspection_pre_pose').value,
            'inspection_pre_pose',
        )
        self.inspection_pose = require_pose_list(
            self.get_parameter('inspection_pose').value,
            'inspection_pose',
        )
        self.pick_pre_offset = require_pose_list(
            self.get_parameter('pick_pre_offset').value,
            'pick_pre_offset',
        )
        self.pick_retreat_offset = require_pose_list(
            self.get_parameter('pick_retreat_offset').value,
            'pick_retreat_offset',
        )
        self.place_offset = require_pose_list(
            self.get_parameter('place_offset').value,
            'place_offset',
        )
        self.place_pre_offset = require_pose_list(
            self.get_parameter('place_pre_offset').value,
            'place_pre_offset',
        )
        self.place_retreat_offset = require_pose_list(
            self.get_parameter('place_retreat_offset').value,
            'place_retreat_offset',
        )
        self.target_box_type = str(self.get_parameter('target_box_type').value)
        self.startup_delay_sec = float(
            self.get_parameter('startup_delay_sec').value
        )
        self.pick_service_timeout_sec = float(
            self.get_parameter('pick_service_timeout_sec').value
        )
        self.slot_service_timeout_sec = float(
            self.get_parameter('slot_service_timeout_sec').value
        )
        self.inspection_service_timeout_sec = float(
            self.get_parameter('inspection_service_timeout_sec').value
        )
        self.motion_server_timeout_sec = float(
            self.get_parameter('motion_server_timeout_sec').value
        )
        self.moveit_ik_service = str(
            self.get_parameter('moveit_ik_service').value
        )
        self.moveit_cartesian_service = str(
            self.get_parameter('moveit_cartesian_service').value
        )
        self.moveit_group_name = str(
            self.get_parameter('moveit_group_name').value
        )
        self.moveit_tip_link = str(
            self.get_parameter('moveit_tip_link').value
        )
        self.moveit_ik_timeout_sec = float(
            self.get_parameter('moveit_ik_timeout_sec').value
        )
        self.moveit_ik_attempts = int(
            self.get_parameter('moveit_ik_attempts').value
        )
        self.moveit_cartesian_max_step = float(
            self.get_parameter('moveit_cartesian_max_step').value
        )
        self.moveit_movej_max_joint_step = float(
            self.get_parameter('moveit_movej_max_joint_step').value
        )
        self.moveit_cartesian_jump_threshold = float(
            self.get_parameter('moveit_cartesian_jump_threshold').value
        )
        self.moveit_avoid_collisions = bool(
            self.get_parameter('moveit_avoid_collisions').value
        )

        self.movej_config = {
            'velocity': float(self.get_parameter('movej_velocity').value),
            'acceleration': float(
                self.get_parameter('movej_acceleration').value
            ),
            'blend_radius': float(
                self.get_parameter('movej_blend_radius').value
            ),
            'dwell_sec': float(self.get_parameter('movej_dwell_sec').value),
        }
        self.init_movej_config = {
            'velocity': float(
                self.get_parameter('init_movej_velocity').value
            ),
            'acceleration': float(
                self.get_parameter('init_movej_acceleration').value
            ),
            'blend_radius': float(
                self.get_parameter('init_movej_blend_radius').value
            ),
            'dwell_sec': float(
                self.get_parameter('init_movej_dwell_sec').value
            ),
        }
        self.movel_config = {
            'velocity': float(self.get_parameter('movel_velocity').value),
            'acceleration': float(
                self.get_parameter('movel_acceleration').value
            ),
            'blend_radius': float(
                self.get_parameter('movel_blend_radius').value
            ),
            'dwell_sec': float(self.get_parameter('movel_dwell_sec').value),
        }
        self.gripper_dwell_sec = float(
            self.get_parameter('gripper_dwell_sec').value
        )
        self.inspection_wait_sec = float(
            self.get_parameter('inspection_wait_sec').value
        )

        # 运行时状态：
        # - _state: 当前 FSM 状态
        # - _current_pcb_id: 当前这块板的流程 ID
        # - _cycle_index: 自增计数，用于生成 pcb_id
        # - _last_ready_for_pick: 用来做 ready 信号上升沿检测
        # - _recovering: 避免错误恢复阶段反复进入 recover_home
        self._state = ProcessState.IDLE
        self._current_pcb_id: str | None = None
        self._cycle_index = 0
        self._last_ready_for_pick = False
        self._recovering = False
        self._init_wait_started_at = None
        self._init_check_timer = None
        self._robot_state_request_in_flight = False
        self._manual_recovery_active = False
        self._manual_recovery_goal_handle = None
        self._manual_recovery_done = threading.Event()
        self._manual_recovery_result: tuple[bool, str] | None = None

        # 订阅两个高层状态 topic：
        # 1. PCB 到位状态
        # 2. 检测结果
        self.create_subscription(
            PcbPresence,
            str(self.get_parameter('presence_topic').value),
            self._presence_callback,
            10,
        )
        self.create_subscription(
            InspectionResult,
            str(self.get_parameter('inspection_result_topic').value),
            self._inspection_result_callback,
            10,
        )

        # 三个 service client 分别对应：
        # 1. 请求抓取位姿
        # 2. 请求可用槽位
        # 3. 触发检测
        # 4. 请求机器人当前状态
        self.robot_state_client = self.create_client(
            GetRobotState,
            self.robot_state_service_name,
        )
        self.pick_pose_client = self.create_client(
            GetPcbPickPose,
            str(self.get_parameter('pick_pose_service').value),
        )
        self.slot_client = self.create_client(
            GetAvailableSlot,
            str(self.get_parameter('slot_service').value),
        )
        self.inspection_client = self.create_client(
            TriggerInspection,
            str(self.get_parameter('inspection_trigger_service').value),
        )
        self._motion_resolver = self._create_motion_resolver()
        # 运动通过 action 执行，因为它是耗时操作，需要反馈和结果。
        self.motion_client = ActionClient(
            self,
            ExecuteMotionSequence,
            str(self.get_parameter('motion_action_name').value),
        )
        self.recover_action_server = ActionServer(
            self,
            RecoverToInitial,
            '/pcb_process/recover_to_initial',
            execute_callback=self._execute_recover_to_initial,
            goal_callback=self._recover_to_initial_goal_callback,
            cancel_callback=self._recover_to_initial_cancel_callback,
            handle_accepted_callback=self._recover_to_initial_handle_accepted,
        )

        # 启动时给一点时间等待其他节点起来，然后再进入 WAIT_PCB。
        self._startup_timer = self.create_timer(
            max(self.startup_delay_sec, 0.05),
            self._on_startup_timer,
        )
        self.get_logger().info(
            'PCB process task manager ready. '
            f'loaded initial_joint_position={self._format_values(self.initial_joint_position)}, '
            f'home_pose={self._format_values(self.home_pose)}, '
            f'inspection_pre_pose={self._format_values(self.inspection_pre_pose)}, '
            f'inspection_pose={self._format_values(self.inspection_pose)}, '
            f'pick_pre_offset={self._format_values(self.pick_pre_offset)}, '
            f'pick_retreat_offset={self._format_values(self.pick_retreat_offset)}, '
            f'place_offset={self._format_values(self.place_offset)}, '
            f'place_pre_offset={self._format_values(self.place_pre_offset)}, '
            f'place_retreat_offset={self._format_values(self.place_retreat_offset)}, '
            f'init_trajectory_points_count={len(self.init_trajectory_points)}, '
            f'moveit_group_name={self.moveit_group_name}, '
            f'moveit_tip_link={self.moveit_tip_link}, '
            f'moveit_ik_service={self.moveit_ik_service}, '
            f'moveit_cartesian_service={self.moveit_cartesian_service}, '
            f'moveit_movej_max_joint_step={self.moveit_movej_max_joint_step:.6f}'
        )

    def _format_values(self, values) -> str:
        return '[' + ', '.join(f'{float(value):.6f}' for value in values) + ']'

    def _declare_parameters(self) -> None:
        self.declare_parameter('presence_topic', '/vision/line/pcb_presence')
        self.declare_parameter(
            'inspection_result_topic',
            '/inspection/result',
        )
        self.declare_parameter(
            'pick_pose_service',
            '/vision/line/get_pcb_pick_pose',
        )
        self.declare_parameter(
            'slot_service',
            '/vision/slot/get_available_slot',
        )
        self.declare_parameter(
            'inspection_trigger_service',
            '/inspection/trigger',
        )
        self.declare_parameter(
            'motion_action_name',
            '/motion_backend/execute_motion_sequence',
        )
        self.declare_parameter('frame_id', 'base_link')
        self.declare_parameter('enable_initialization', False)
        self.declare_parameter(
            'robot_state_service',
            '/motion_backend/get_robot_state',
        )
        self.declare_parameter('initial_state_poll_sec', 0.1)
        self.declare_parameter(
            'initial_joint_position',
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        )
        self.declare_parameter('initial_joint_tolerance', 0.02)
        self.declare_parameter('initial_joint_state_timeout_sec', 5.0)
        # 空列表在 rclpy 里会被推断成 BYTE_ARRAY。
        # 这里我们实际需要的是 string[]，因为 YAML 中每个初始化路点都写成一条字符串。
        self.declare_parameter(
            'init_trajectory_points',
            Parameter.Type.STRING_ARRAY,
        )
        self.declare_parameter('target_box_type', 'good')
        self.declare_parameter('startup_delay_sec', 0.2)
        self.declare_parameter('pick_service_timeout_sec', 1.0)
        self.declare_parameter('slot_service_timeout_sec', 1.0)
        self.declare_parameter('inspection_service_timeout_sec', 1.0)
        self.declare_parameter('motion_server_timeout_sec', 1.0)
        self.declare_parameter('moveit_ik_service', '/compute_ik')
        self.declare_parameter(
            'moveit_cartesian_service',
            '/compute_cartesian_path',
        )
        self.declare_parameter('moveit_group_name', 'jearm_arm')
        self.declare_parameter('moveit_tip_link', 'Link7')
        self.declare_parameter('moveit_ik_timeout_sec', 2.0)
        self.declare_parameter('moveit_ik_attempts', 1)
        self.declare_parameter('moveit_cartesian_max_step', 0.01)
        self.declare_parameter('moveit_movej_max_joint_step', 0.1)
        self.declare_parameter('moveit_cartesian_jump_threshold', 0.0)
        self.declare_parameter('moveit_avoid_collisions', False)

        self.declare_parameter('home_pose', [0.32, 0.00, 0.28, 3.14, 0.0, 0.0])
        self.declare_parameter(
            'inspection_pre_pose',
            [0.18, 0.18, 0.22, 3.14, 0.0, 0.0],
        )
        self.declare_parameter(
            'inspection_pose',
            [0.18, 0.18, 0.14, 3.14, 0.0, 0.0],
        )
        self.declare_parameter(
            'pick_pre_offset',
            [0.0, 0.0, 0.08, 0.0, 0.0, 0.0],
        )
        self.declare_parameter(
            'pick_retreat_offset',
            [0.0, 0.0, 0.08, 0.0, 0.0, 0.0],
        )
        self.declare_parameter(
            'place_offset',
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        )
        self.declare_parameter(
            'place_pre_offset',
            [0.0, 0.0, 0.08, 0.0, 0.0, 0.0],
        )
        self.declare_parameter(
            'place_retreat_offset',
            [0.0, 0.0, 0.08, 0.0, 0.0, 0.0],
        )
        self.declare_parameter('movej_velocity', 0.3)
        self.declare_parameter('movej_acceleration', 0.3)
        self.declare_parameter('movej_blend_radius', 0.0)
        self.declare_parameter('movej_dwell_sec', 0.3)
        self.declare_parameter('init_movej_velocity', 0.2)
        self.declare_parameter('init_movej_acceleration', 0.2)
        self.declare_parameter('init_movej_blend_radius', 0.0)
        self.declare_parameter('init_movej_dwell_sec', 0.3)
        self.declare_parameter('movel_velocity', 0.1)
        self.declare_parameter('movel_acceleration', 0.1)
        self.declare_parameter('movel_blend_radius', 0.0)
        self.declare_parameter('movel_dwell_sec', 0.3)
        self.declare_parameter('gripper_dwell_sec', 0.4)
        self.declare_parameter('inspection_wait_sec', 0.5)

    def _parse_joint_waypoint_strings(
        self,
        values,
    ) -> list[list[float]]:
        """把 YAML 里的字符串数组解析成关节路点列表。

        ROS2 参数不支持“二维 double 数组”，所以这里约定每个轨迹点写成一条字符串：
        - `"0.1, 0.2, ..."`
        - `"[0.1, 0.2, ...]"`

        解析后统一转成 7 维 float 列表，供初始化轨迹直接使用。
        """
        points: list[list[float]] = []
        for index, raw_value in enumerate(values):
            text = str(raw_value).strip()
            if not text:
                continue

            if text.startswith('['):
                parsed = json.loads(text)
            else:
                parsed = [
                    float(item.strip())
                    for item in text.split(',')
                    if item.strip()
                ]

            points.append(
                require_joint_list(
                    parsed,
                    f'init_trajectory_points[{index}]',
                )
            )
        return points

    def _create_motion_resolver(self):
        return MoveItMotionResolver(
            node=self,
            robot_state_service_name=self.robot_state_service_name,
            moveit_ik_service=self.moveit_ik_service,
            moveit_cartesian_service=self.moveit_cartesian_service,
            moveit_group_name=self.moveit_group_name,
            moveit_tip_link=self.moveit_tip_link,
            moveit_ik_timeout_sec=self.moveit_ik_timeout_sec,
            moveit_ik_attempts=self.moveit_ik_attempts,
            moveit_cartesian_max_step=self.moveit_cartesian_max_step,
            moveit_movej_max_joint_step=self.moveit_movej_max_joint_step,
            moveit_cartesian_jump_threshold=self.moveit_cartesian_jump_threshold,
            moveit_avoid_collisions=self.moveit_avoid_collisions,
            robot_state_timeout_sec=max(
                self.motion_server_timeout_sec,
                self.moveit_ik_timeout_sec,
            ),
        )

    def _resolve_motion_steps(self, sequence_name: str, steps):
        input_steps = list(steps)
        resolved_steps = self._motion_resolver.resolve_steps(input_steps)
        self.get_logger().info(
            f'MoveIt resolved {sequence_name}: '
            f'input_steps={len(input_steps)}, '
            f'output_steps={len(resolved_steps)}'
        )
        return resolved_steps

    def _on_startup_timer(self) -> None:
        self._startup_timer.cancel()
        if not self.enable_initialization:
            self._transition(ProcessState.WAIT_PCB, 'startup_complete')
            return

        # 初始化模式下，启动后周期性向统一 ZMQ 网关请求当前机器人状态。
        self._init_wait_started_at = self.get_clock().now()
        self._transition(
            ProcessState.WAIT_INIT_JOINT_STATE,
            'startup_waiting_robot_state',
        )
        self._init_check_timer = self.create_timer(
            max(self.initial_state_poll_sec, 0.05),
            self._check_initialization_ready,
        )

    def _stop_init_check_timer(self) -> None:
        if self._init_check_timer is None:
            return
        self._init_check_timer.cancel()
        self.destroy_timer(self._init_check_timer)
        self._init_check_timer = None

    def _request_robot_state_snapshot(self) -> None:
        """向统一网关请求一次最新机器人状态。"""
        if self._robot_state_request_in_flight:
            return
        if not self.robot_state_client.wait_for_service(timeout_sec=0.0):
            return

        request = GetRobotState.Request()
        future = self.robot_state_client.call_async(request)
        self._robot_state_request_in_flight = True
        future.add_done_callback(self._on_robot_state_response)

    def _try_start_initialization(self, joint_positions) -> None:
        """用一次有效关节快照判断是否可以开始初始化轨迹。"""
        diffs = [
            abs(current - expected)
            for current, expected in zip(
                joint_positions,
                self.initial_joint_position,
            )
        ]
        mismatch_index = next(
            (
                index
                for index, diff in enumerate(diffs)
                if diff > self.initial_joint_tolerance
            ),
            None,
        )
        if mismatch_index is not None:
            self._stop_init_check_timer()
            current = float(joint_positions[mismatch_index])
            expected = self.initial_joint_position[mismatch_index]
            diff = diffs[mismatch_index]
            self._enter_error(
                'initial_joint_mismatch:'
                f'joint{mismatch_index + 1}:'
                f'current={current:.6f},'
                f'expected={expected:.6f},'
                f'diff={diff:.6f}',
                recover=False,
            )
            return

        if not self.init_trajectory_points:
            self._stop_init_check_timer()
            self._enter_error('init_trajectory_points_empty', recover=False)
            return

        self._stop_init_check_timer()
        self.get_logger().info(
            'Using preloaded init_trajectory_points for initialization: '
            f'count={len(self.init_trajectory_points)}'
        )
        for index, point in enumerate(self.init_trajectory_points, start=1):
            self.get_logger().info(
                f'init_trajectory_points[{index}]='
                f'{self._format_values(point)}'
            )
        steps = build_joint_trajectory_sequence(
            self.init_trajectory_points,
            self.init_movej_config,
            step_name_prefix='init_waypoint',
        )
        self._transition(
            ProcessState.EXECUTE_INITIALIZATION,
            'initial_joint_check_passed',
        )
        self._send_motion_goal(
            'initialization',
            'initialization_sequence',
            steps,
        )

    def _on_robot_state_response(self, future) -> None:
        """消费机器人状态 service 响应，只在初始化阶段使用。"""
        self._robot_state_request_in_flight = False
        if self._state != ProcessState.WAIT_INIT_JOINT_STATE:
            return

        try:
            response = future.result()
        except Exception:
            return

        if not response.success or not response.state.valid:
            return
        if not response.state.joint_valid:
            return
        if len(response.state.joint_position) < 7:
            return

        self._try_start_initialization(response.state.joint_position[:7])

    def _check_initialization_ready(self) -> None:
        """等待机器人状态并完成“初始位校验 -> 初始化轨迹执行”的启动门禁。"""
        if self._state != ProcessState.WAIT_INIT_JOINT_STATE:
            return

        if self._init_wait_started_at is None:
            self._init_wait_started_at = self.get_clock().now()

        elapsed_sec = (
            self.get_clock().now() - self._init_wait_started_at
        ).nanoseconds / 1e9

        if elapsed_sec >= self.initial_joint_state_timeout_sec:
            self._stop_init_check_timer()
            self._enter_error(
                'initial_joint_state_timeout',
                recover=False,
            )
            return

        if self._robot_state_request_in_flight:
            return

        self._request_robot_state_snapshot()

    def _recover_to_initial_goal_callback(
        self,
        _goal_request: RecoverToInitial.Goal,
    ) -> GoalResponse:
        allowed_states = {
            ProcessState.IDLE,
            ProcessState.WAIT_PCB,
            ProcessState.ERROR,
        }
        if (
            self._state not in allowed_states
            or self._manual_recovery_active
            or self._recovering
        ):
            return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    def _recover_to_initial_cancel_callback(self, _goal_handle) -> CancelResponse:
        return CancelResponse.REJECT

    def _recover_to_initial_handle_accepted(self, goal_handle) -> None:
        thread = threading.Thread(
            target=goal_handle.execute,
            name='recover_to_initial_execute',
            daemon=True,
        )
        thread.start()

    def _finalize_manual_recovery(
        self,
        success: bool,
        error_code: str,
    ) -> None:
        if self._manual_recovery_result is None:
            self._manual_recovery_result = (bool(success), str(error_code))
            self._manual_recovery_done.set()

    def _build_manual_recovery_steps(self):
        return build_recover_to_initial_sequence(
            self.frame_id,
            self.home_pose,
            self.initial_joint_position,
            self.init_trajectory_points,
            self.movej_config,
        )

    def _execute_recover_to_initial(self, goal_handle):
        result = RecoverToInitial.Result()
        self._manual_recovery_active = True
        self._manual_recovery_goal_handle = goal_handle
        self._manual_recovery_result = None
        self._manual_recovery_done.clear()

        try:
            self.get_logger().info(
                'Using preloaded values for manual recovery: '
                f'home_pose={self._format_values(self.home_pose)}, '
                f'initial_joint_position={self._format_values(self.initial_joint_position)}, '
                f'init_trajectory_points_count={len(self.init_trajectory_points)}'
            )
            steps = self._build_manual_recovery_steps()
        except Exception as exc:
            error_code = f'manual_recovery_prepare_failed:{exc}'
            self._enter_error(error_code, recover=False)
            self._finalize_manual_recovery(False, error_code)
        else:
            self._current_pcb_id = None
            self._transition(
                ProcessState.EXECUTE_MANUAL_RECOVERY,
                'manual_recovery_requested',
            )
            self._send_motion_goal(
                'manual_recover',
                'manual_recover_sequence',
                steps,
            )

        while rclpy.ok():
            if self._manual_recovery_done.wait(timeout=0.1):
                break

        success, error_code = self._manual_recovery_result or (
            False,
            'manual_recovery_interrupted',
        )
        self._manual_recovery_active = False
        self._manual_recovery_goal_handle = None
        self._manual_recovery_result = None
        self._manual_recovery_done.clear()

        result.success = bool(success)
        result.error_code = str(error_code)
        if success:
            goal_handle.succeed()
        else:
            goal_handle.abort()
        return result

    def _transition(self, state: ProcessState, reason: str) -> None:
        # 统一从这里打印状态迁移日志，便于排查流程卡在哪一步。
        if self._state == state:
            return
        self.get_logger().info(
            green_text(
                f'STATE {self._state.value} -> {state.value}: {reason}'
            )
        )
        self._state = state

    def _reset_for_next_cycle(self) -> None:
        # 一轮流程完成后，只保留状态机和计数器，把当前板上下文清空。
        self._current_pcb_id = None
        self._recovering = False
        self._robot_state_request_in_flight = False
        self._transition(ProcessState.WAIT_PCB, 'cycle_reset')

    def _presence_callback(self, msg: PcbPresence) -> None:
        ready = bool(msg.ready_for_pick)
        # 只在 WAIT_PCB 状态下响应“false -> true”的上升沿。
        # 这样键盘节点持续重复发布 ready=true 时，不会重复启动多个流程。
        if (
            self._state == ProcessState.WAIT_PCB
            and ready
            and not self._last_ready_for_pick
        ):
            self._cycle_index += 1
            self._current_pcb_id = f'pcb_{self._cycle_index:06d}'
            self._transition(ProcessState.MOVE_HOME_BEFORE_PICK, 'pcb_ready_edge')
            self.get_logger().info(
                'Using preloaded home_pose for home_before_pick: '
                f'{self._format_values(self.home_pose)}'
            )
            steps = build_home_sequence(
                self.frame_id,
                self.home_pose,
                self.movej_config,
            )
            self._send_motion_goal(
                'home_before_pick',
                'home_before_pick_sequence',
                steps,
            )
        self._last_ready_for_pick = ready

    def _request_pick_pose(self) -> None:
        # 先等 service 可用，再发异步请求。
        # 这里不阻塞 spin，响应结果通过 done_callback 回来。
        if not self.pick_pose_client.wait_for_service(
            timeout_sec=self.pick_service_timeout_sec
        ):
            self._enter_error('pick_pose_service_unavailable')
            return

        request = GetPcbPickPose.Request()
        request.require_stable = True
        future = self.pick_pose_client.call_async(request)
        future.add_done_callback(self._on_pick_pose_response)

    def _on_pick_pose_response(self, future) -> None:
        # 任何 service 失败，都统一进入错误恢复逻辑。
        try:
            response = future.result()
        except Exception as exc:
            self._enter_error(f'pick_pose_request_failed:{exc}')
            return

        if not response.success:
            self._enter_error(f'pick_pose_rejected:{response.reason}')
            return

        # 感知成功后，立刻把抓取动作序列拼出来并发给运动后端。
        self.get_logger().info(
            'Using returned pick_pose_base with preloaded pick offsets: '
            f'pick_pose={self._format_values(pose_to_rpy_list(response.pick_pose_base))}, '
            f'pick_pre_offset={self._format_values(self.pick_pre_offset)}, '
            f'pick_retreat_offset={self._format_values(self.pick_retreat_offset)}'
        )
        steps = build_pick_sequence(
            response.pick_pose_base,
            self.frame_id,
            self.pick_pre_offset,
            self.pick_retreat_offset,
            self.movej_config,
            self.movel_config,
            self.gripper_dwell_sec,
        )
        self._transition(ProcessState.EXECUTE_PICK, 'pick_pose_ready')
        self._send_motion_goal('pick', 'pick_sequence', steps)

    def _send_motion_goal(
        self,
        purpose: str,
        sequence_name: str,
        steps,
    ) -> None:
        # 每次发动作前都确认 action server 在线。
        if not self.motion_client.wait_for_server(
            timeout_sec=self.motion_server_timeout_sec
        ):
            self._enter_error('motion_action_unavailable')
            return

        try:
            resolved_steps = self._resolve_motion_steps(sequence_name, steps)
        except Exception as exc:
            self._enter_error(f'motion_prepare_failed:{purpose}:{exc}')
            return

        # sequence_name 用来标记当前这组动作属于哪个流程阶段。
        goal = ExecuteMotionSequence.Goal()
        goal.sequence_name = sequence_name
        goal.pcb_id = self._current_pcb_id or ''
        goal.steps = list(resolved_steps)
        future = self.motion_client.send_goal_async(
            goal,
            feedback_callback=partial(self._on_motion_feedback, purpose),
        )
        future.add_done_callback(
            partial(self._on_motion_goal_response, purpose)
        )

    def _on_motion_feedback(self, purpose: str, feedback_msg) -> None:
        # 反馈只做日志输出，帮助确认执行到哪一步。
        feedback = feedback_msg.feedback
        self.get_logger().info(
            f'[{purpose}] step={feedback.step_index} '
            f'name={feedback.active_step_name}'
        )
        if (
            purpose == 'manual_recover'
            and self._manual_recovery_goal_handle is not None
        ):
            recover_feedback = RecoverToInitial.Feedback()
            recover_feedback.stage = feedback.stage
            recover_feedback.step_index = feedback.step_index
            recover_feedback.active_step_name = feedback.active_step_name
            self._manual_recovery_goal_handle.publish_feedback(
                recover_feedback
            )

    def _on_motion_goal_response(self, purpose: str, future) -> None:
        # 这里只代表 goal 是否被 action server 接受，
        # 真正的执行成功与否还要看后续 result。
        try:
            goal_handle = future.result()
        except Exception as exc:
            self._enter_error(f'motion_goal_failed:{purpose}:{exc}')
            return

        if not goal_handle.accepted:
            self._enter_error(f'motion_goal_rejected:{purpose}')
            return

        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(partial(self._on_motion_result, purpose))

    def _on_motion_result(self, purpose: str, future) -> None:
        # 所有动作结果最终都在这里汇总，再决定下一个状态。
        try:
            wrapped = future.result()
        except Exception as exc:
            self._enter_error(f'motion_result_failed:{purpose}:{exc}')
            return

        result = wrapped.result
        status = wrapped.status
        if status != GoalStatus.STATUS_SUCCEEDED or not result.success:
            error_code = f'motion_failed:{purpose}:{result.error_code or status}'
            if purpose == 'manual_recover':
                self._enter_error(error_code, recover=False)
                self._finalize_manual_recovery(False, error_code)
                return
            self._enter_error(error_code)
            return

        if purpose == 'initialization':
            # 初始化固定关节轨迹走完后，流程才正式进入等待来板阶段。
            self._recovering = False
            self._transition(
                ProcessState.WAIT_PCB,
                'initialization_sequence_done',
            )
            return

        if purpose == 'home_before_pick':
            self._transition(
                ProcessState.REQUEST_PICK_POSE,
                'home_before_pick_done',
            )
            self._request_pick_pose()
            return

        if purpose == 'pick':
            # 抓取完成后，下一步不是立刻检测，而是先执行“移动到检测位”的动作序列。
            self._transition(
                ProcessState.MOVE_TO_INSPECTION,
                'pick_sequence_done',
            )
            self.get_logger().info(
                'Using preloaded inspection poses: '
                f'inspection_pre_pose={self._format_values(self.inspection_pre_pose)}, '
                f'inspection_pose={self._format_values(self.inspection_pose)}'
            )
            steps = build_inspection_sequence(
                self.frame_id,
                self.inspection_pre_pose,
                self.inspection_pose,
                self.movej_config,
                self.movel_config,
                self.inspection_wait_sec,
            )
            self._send_motion_goal(
                'inspection_move',
                'inspection_sequence',
                steps,
            )
            return

        if purpose == 'inspection_move':
            # 已经到检测位，才允许触发检测 service。
            self._transition(
                ProcessState.TRIGGER_INSPECTION,
                'inspection_pose_reached',
            )
            self._trigger_inspection()
            return

        if purpose == 'place':
            # 放置完成后统一回 home，流程才算结束。
            self._transition(ProcessState.GO_HOME, 'place_sequence_done')
            self.get_logger().info(
                'Using preloaded home_pose for home_sequence: '
                f'{self._format_values(self.home_pose)}'
            )
            steps = build_home_sequence(
                self.frame_id,
                self.home_pose,
                self.movej_config,
            )
            self._send_motion_goal('home', 'home_sequence', steps)
            return

        if purpose == 'manual_recover':
            self._current_pcb_id = None
            self._transition(
                ProcessState.WAIT_PCB,
                'manual_recovery_sequence_done',
            )
            self._finalize_manual_recovery(True, '')
            return

        if purpose in ('home', 'recover_home'):
            # 无论是正常回 home 还是异常恢复回 home，只要完成就进入下一轮等待。
            self._reset_for_next_cycle()

    def _trigger_inspection(self) -> None:
        # 触发检测仍然走 service；检测结果本身通过 topic 异步回来。
        if not self.inspection_client.wait_for_service(
            timeout_sec=self.inspection_service_timeout_sec
        ):
            self._enter_error('inspection_service_unavailable')
            return

        request = TriggerInspection.Request()
        request.pcb_id = self._current_pcb_id or ''
        future = self.inspection_client.call_async(request)
        future.add_done_callback(self._on_trigger_inspection_response)

    def _on_trigger_inspection_response(self, future) -> None:
        try:
            response = future.result()
        except Exception as exc:
            self._enter_error(f'inspection_trigger_failed:{exc}')
            return

        if not response.accepted:
            self._enter_error(f'inspection_rejected:{response.reason}')
            return
        self._transition(
            ProcessState.WAIT_INSPECTION_RESULT,
            'inspection_triggered',
        )

    def _inspection_result_callback(self, msg: InspectionResult) -> None:
        # 只有在等待检测结果的阶段，才消费检测 topic。
        if self._state != ProcessState.WAIT_INSPECTION_RESULT:
            return
        # pcb_id 不匹配则直接忽略，避免收到旧消息或别的流程消息。
        if not self._current_pcb_id or msg.pcb_id != self._current_pcb_id:
            return
        if not msg.valid:
            self._enter_error('inspection_result_invalid')
            return
        if msg.result != 'good':
            self._enter_error(f'unexpected_inspection_result:{msg.result}')
            return

        # 初版只接受 good，后续如果要接 bad/retry，可以从这里分叉。
        self._transition(ProcessState.REQUEST_GOOD_SLOT, 'inspection_good')
        self._request_good_slot()

    def _request_good_slot(self) -> None:
        # 根据当前检测结果向槽位服务请求一个目标槽位。
        if not self.slot_client.wait_for_service(
            timeout_sec=self.slot_service_timeout_sec
        ):
            self._enter_error('slot_service_unavailable')
            return

        request = GetAvailableSlot.Request()
        request.box_type = self.target_box_type
        request.require_empty = True
        future = self.slot_client.call_async(request)
        future.add_done_callback(self._on_slot_response)

    def _on_slot_response(self, future) -> None:
        try:
            response = future.result()
        except Exception as exc:
            self._enter_error(f'slot_request_failed:{exc}')
            return

        if not response.success:
            self._enter_error(f'slot_request_rejected:{response.reason}')
            return
        if not response.slot_empty:
            self._enter_error('slot_not_empty')
            return

        # 槽位可用后，生成固定放置动作序列。
        self.get_logger().info(
            'Using slot_pose_base with preloaded place offsets: '
            f'slot_pose={self._format_values(pose_to_rpy_list(response.slot_pose_base))}, '
            f'place_offset={self._format_values(self.place_offset)}, '
            f'place_pre_offset={self._format_values(self.place_pre_offset)}, '
            f'place_retreat_offset={self._format_values(self.place_retreat_offset)}'
        )
        steps = build_place_sequence(
            response.slot_pose_base,
            self.frame_id,
            self.place_offset,
            self.place_pre_offset,
            self.place_retreat_offset,
            self.movej_config,
            self.movel_config,
            self.gripper_dwell_sec,
        )
        self._transition(ProcessState.EXECUTE_PLACE, 'slot_ready')
        self._send_motion_goal('place', 'place_sequence', steps)

    def _enter_error(self, reason: str, *, recover: bool = True) -> None:
        # 所有错误统一走这里：
        # 1. 记录错误
        # 2. 切到 ERROR 状态
        # 3. 尝试执行 recover_home
        #
        # 如果 recover_home 本身也失败过一次，就不再递归恢复，直接 reset。
        self.get_logger().error(reason)
        self._stop_init_check_timer()
        self._robot_state_request_in_flight = False
        self._transition(ProcessState.ERROR, reason)
        if self._manual_recovery_active:
            self._finalize_manual_recovery(False, reason)
            recover = False
        if not recover:
            # 初始化起点不匹配这类错误，要求“只报错不乱动机器人”。
            return
        if self._recovering:
            self._reset_for_next_cycle()
            return
        self._recovering = True
        self.get_logger().info(
            'Using preloaded home_pose for recover_home: '
            f'{self._format_values(self.home_pose)}'
        )
        steps = build_home_sequence(
            self.frame_id,
            self.home_pose,
            self.movej_config,
        )
        self._send_motion_goal('recover_home', 'recover_home_sequence', steps)

    def destroy_node(self) -> bool:
        self._stop_init_check_timer()
        if getattr(self, '_motion_resolver', None) is not None:
            self._motion_resolver.shutdown()
        return super().destroy_node()


def main() -> None:
    rclpy.init()
    node = PcbProcessTaskManagerNode()
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
