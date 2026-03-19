#!/usr/bin/env python3
"""固定点 PCB 闭环示例共用工具。

这里放的是“流程无关、节点共用”的基础能力，主要包括：
1. 位姿格式转换：`[x, y, z, roll, pitch, yaw] <-> PoseStamped`
2. MotionStep 构造：统一封装 MoveJ / MoveL / Gripper / Wait
3. 固定流程动作模板：抓取、送检、放置、回 home

这样做的目的，是把“动作序列怎么拼”从任务状态机里拆出来，
后续如果替换成真实感知或真实规划，只需要复用这些基础拼装逻辑即可。
"""

from __future__ import annotations

import math
from typing import Iterable

from geometry_msgs.msg import PoseStamped

from je_software.msg import MotionStep

# MotionStep.command_type 的固定枚举值。
# 这里统一用字符串，便于 action goal / 单元测试 / ZMQ 后端保持一致。
MOVEJ = 'MOVEJ'
MOVEL = 'MOVEL'
GRIPPER = 'GRIPPER'
WAIT = 'WAIT'

# 夹爪命令的固定枚举值。
GRIPPER_OPEN = 'OPEN'
GRIPPER_CLOSE = 'CLOSE'
GRIPPER_NONE = 'NONE'


def require_pose_list(values: Iterable[float], name: str) -> list[float]:
    """把输入位姿强制规范成 6 维 float 列表。

    全流程约定位姿统一写成：
    `[x, y, z, roll, pitch, yaw]`

    这样 YAML、service 响应、动作模板里都能复用同一套格式。
    如果长度不对，直接抛异常，避免把错误参数带进运行时。
    """
    pose = [float(value) for value in values]
    if len(pose) != 6:
        raise ValueError(f'{name} must contain exactly 6 values.')
    return pose


def require_joint_list(
    values: Iterable[float],
    name: str,
    *,
    joint_count: int = 7,
) -> list[float]:
    """把输入关节序列规范成固定长度的 float 列表。

    当前初始化流程约定单臂一共 7 个关节，因此默认严格要求 7 维。
    如果现场后续换成别的自由度，可以通过 `joint_count` 扩展。
    """
    joints = [float(value) for value in values]
    if len(joints) != joint_count:
        raise ValueError(
            f'{name} must contain exactly {joint_count} joint values.'
        )
    return joints


def rpy_to_quaternion(roll: float, pitch: float, yaw: float) -> tuple[float, ...]:
    """把欧拉角转换成四元数。

    ROS 的 Pose 使用四元数，配置文件和人工录入更适合用 RPY，
    所以这里负责做两种表示之间的桥接。
    """
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    return qx, qy, qz, qw


def quaternion_to_rpy(
    qx: float,
    qy: float,
    qz: float,
    qw: float,
) -> tuple[float, float, float]:
    """把四元数转换回 roll / pitch / yaw。"""
    norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    # 先归一化，避免外部传入的四元数有微小数值漂移。
    if norm > 1e-12:
        qx /= norm
        qy /= norm
        qz /= norm
        qw /= norm
    else:
        qx = 0.0
        qy = 0.0
        qz = 0.0
        qw = 1.0

    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (qw * qy - qz * qx)
    if abs(sinp) >= 1.0:
        pitch = math.copysign(math.pi / 2.0, sinp)
    else:
        pitch = math.asin(sinp)

    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


def make_pose_stamped(values: Iterable[float], frame_id: str) -> PoseStamped:
    """从 `[x, y, z, roll, pitch, yaw]` 生成 PoseStamped。"""
    x_pos, y_pos, z_pos, roll, pitch, yaw = require_pose_list(values, 'pose')
    qx, qy, qz, qw = rpy_to_quaternion(roll, pitch, yaw)
    pose = PoseStamped()
    pose.header.frame_id = str(frame_id)
    pose.pose.position.x = x_pos
    pose.pose.position.y = y_pos
    pose.pose.position.z = z_pos
    pose.pose.orientation.x = qx
    pose.pose.orientation.y = qy
    pose.pose.orientation.z = qz
    pose.pose.orientation.w = qw
    return pose


def pose_to_rpy_list(pose: PoseStamped) -> list[float]:
    """把 PoseStamped 转回统一的 6 维位姿列表。"""
    roll, pitch, yaw = quaternion_to_rpy(
        pose.pose.orientation.x,
        pose.pose.orientation.y,
        pose.pose.orientation.z,
        pose.pose.orientation.w,
    )
    return [
        float(pose.pose.position.x),
        float(pose.pose.position.y),
        float(pose.pose.position.z),
        roll,
        pitch,
        yaw,
    ]


def offset_pose(
    pose: PoseStamped,
    offset: Iterable[float],
    frame_id: str | None = None,
) -> PoseStamped:
    """在原始位姿上叠加一个简单的 xyz-rpy 偏置。

    这版流程里，pre-grasp / retreat / pre-place 都是靠“基准位姿 + 固定偏置”
    生成出来的，没有引入更复杂的工具坐标系或路径规划逻辑。
    """
    base = pose_to_rpy_list(pose)
    delta = require_pose_list(offset, 'offset')
    # 初版直接做逐项相加，适合当前“固定点 + 固定姿态偏置”的闭环。
    combined = [base[index] + delta[index] for index in range(6)]
    target_frame = str(frame_id) if frame_id is not None else pose.header.frame_id
    return make_pose_stamped(combined, target_frame)


def make_motion_step(
    name: str,
    command_type: str,
    *,
    target_pose: PoseStamped | None = None,
    joint_target: Iterable[float] | None = None,
    velocity: float = 0.0,
    acceleration: float = 0.0,
    blend_radius: float = 0.0,
    gripper_command: str = GRIPPER_NONE,
    dwell_sec: float = 0.0,
) -> MotionStep:
    """创建 MotionStep，并统一补齐默认值。

    这样 task manager 在拼动作序列时，不需要每次都手写 message 细节。
    """
    step = MotionStep()
    step.name = str(name)
    step.command_type = str(command_type)
    if target_pose is not None:
        step.target_pose = target_pose
    if joint_target is not None:
        # joint_target 主要给 MoveA / 关节绝对轨迹使用；
        # 这里统一在 message 里携带，避免再靠 step.name 间接查表。
        step.joint_target = require_joint_list(
            joint_target,
            f'{name}.joint_target',
        )
    step.velocity = float(velocity)
    step.acceleration = float(acceleration)
    step.blend_radius = float(blend_radius)
    step.gripper_command = str(gripper_command)
    step.dwell_sec = float(dwell_sec)
    return step


def build_pick_sequence(
    pick_pose: PoseStamped,
    frame_id: str,
    pick_pre_offset: Iterable[float],
    pick_retreat_offset: Iterable[float],
    movej_config: dict[str, float],
    movel_config: dict[str, float],
    gripper_dwell_sec: float,
) -> list[MotionStep]:
    """生成固定抓取序列。

    序列顺序固定为：
    1. MoveJ 到抓取上方安全位
    2. MoveL 直线下降到抓取位
    3. 夹爪闭合
    4. MoveL 直线抬起退出
    """
    return [
        make_motion_step(
            'pre_grasp',
            MOVEJ,
            # pre-grasp 由抓取位叠加一个安全高度偏置得到。
            target_pose=offset_pose(pick_pose, pick_pre_offset, frame_id),
            **movej_config,
        ),
        make_motion_step(
            'grasp',
            MOVEL,
            # 真正抓取时直接走固定抓取位。
            target_pose=make_pose_stamped(pose_to_rpy_list(pick_pose), frame_id),
            **movel_config,
        ),
        make_motion_step(
            'close',
            GRIPPER,
            gripper_command=GRIPPER_CLOSE,
            dwell_sec=gripper_dwell_sec,
        ),
        make_motion_step(
            'retreat',
            MOVEL,
            # 抓完后沿固定方向退出，先不考虑碰撞检测。
            target_pose=offset_pose(pick_pose, pick_retreat_offset, frame_id),
            **movel_config,
        ),
    ]


def build_inspection_sequence(
    frame_id: str,
    inspection_pre_pose: Iterable[float],
    inspection_pose: Iterable[float],
    movej_config: dict[str, float],
    movel_config: dict[str, float],
    inspection_wait_sec: float,
) -> list[MotionStep]:
    """生成固定送检序列。

    这里把“移动到检测位”和“等待检测触发前的停顿”合成一个动作列表，
    任务管理器只关心什么时候调用，不关心具体拆成几步。
    """
    return [
        make_motion_step(
            'inspection_pre',
            MOVEJ,
            target_pose=make_pose_stamped(inspection_pre_pose, frame_id),
            **movej_config,
        ),
        make_motion_step(
            'inspection',
            MOVEL,
            target_pose=make_pose_stamped(inspection_pose, frame_id),
            **movel_config,
        ),
        make_motion_step(
            'inspection_wait',
            WAIT,
            # WAIT 不发 ZMQ，只在后端本地 sleep，便于把流程跑通。
            dwell_sec=float(inspection_wait_sec),
        ),
    ]


def build_place_sequence(
    slot_pose: PoseStamped,
    frame_id: str,
    place_offset: Iterable[float],
    place_pre_offset: Iterable[float],
    place_retreat_offset: Iterable[float],
    movej_config: dict[str, float],
    movel_config: dict[str, float],
    gripper_dwell_sec: float,
) -> list[MotionStep]:
    """生成固定放置序列。

    序列顺序固定为：
    1. MoveJ 到槽位上方
    2. MoveL 下降到放置位
    3. 夹爪张开
    4. MoveL 抬起退出
    """
    place_pose = offset_pose(slot_pose, place_offset, frame_id)
    return [
        make_motion_step(
            'pre_place',
            MOVEJ,
            # 先移动到目标槽位上方的安全高度。
            target_pose=offset_pose(place_pose, place_pre_offset, frame_id),
            **movej_config,
        ),
        make_motion_step(
            'place',
            MOVEL,
            target_pose=place_pose,
            **movel_config,
        ),
        make_motion_step(
            'open',
            GRIPPER,
            gripper_command=GRIPPER_OPEN,
            dwell_sec=gripper_dwell_sec,
        ),
        make_motion_step(
            'retreat',
            MOVEL,
            # 放完板后直接按固定偏置抬起。
            target_pose=offset_pose(place_pose, place_retreat_offset, frame_id),
            **movel_config,
        ),
    ]


def build_home_sequence(
    frame_id: str,
    home_pose: Iterable[float],
    movej_config: dict[str, float],
) -> list[MotionStep]:
    """生成固定回 home 序列。"""
    return [
        make_motion_step(
            'home',
            MOVEJ,
            target_pose=make_pose_stamped(home_pose, frame_id),
            **movej_config,
        ),
    ]


def build_joint_trajectory_sequence(
    joint_points: Iterable[Iterable[float]],
    movej_config: dict[str, float],
    *,
    step_name_prefix: str,
) -> list[MotionStep]:
    """把一组关节绝对位置拼成固定的 MoveJ 轨迹序列。

    每个点都被解释为“目标关节角”，轨迹顺序严格按照输入顺序执行。
    这正好适合初始化这种“机器人必须沿已知关节路点走一遍”的需求。
    """
    steps: list[MotionStep] = []
    for index, point in enumerate(joint_points, start=1):
        steps.append(
            make_motion_step(
                f'{step_name_prefix}_{index:03d}',
                MOVEJ,
                joint_target=require_joint_list(
                    point,
                    f'{step_name_prefix}[{index}]',
                ),
                **movej_config,
            )
        )
    return steps


def _joint_lists_equal(
    left: Iterable[float],
    right: Iterable[float],
    *,
    tolerance: float = 1e-9,
) -> bool:
    """判断两组关节值是否逐项一致。"""
    left_joints = require_joint_list(left, 'left')
    right_joints = require_joint_list(right, 'right')
    return all(
        abs(left_value - right_value) <= tolerance
        for left_value, right_value in zip(left_joints, right_joints)
    )


def build_recover_to_initial_sequence(
    frame_id: str,
    home_pose: Iterable[float],
    initial_joint_position: Iterable[float],
    init_trajectory_points: Iterable[Iterable[float]],
    movej_config: dict[str, float],
) -> list[MotionStep]:
    """生成“先回 home，再按初始化轨迹倒序回到初始检查位”的序列。"""
    points = [
        require_joint_list(point, f'init_trajectory_points[{index}]')
        for index, point in enumerate(init_trajectory_points)
    ]
    if not points:
        raise ValueError('init_trajectory_points_empty')

    initial_target = require_joint_list(
        initial_joint_position,
        'initial_joint_position',
    )
    reversed_points = list(reversed(points))

    steps = build_home_sequence(frame_id, home_pose, movej_config)
    steps.extend(
        build_joint_trajectory_sequence(
            reversed_points,
            movej_config,
            step_name_prefix='recover_init',
        )
    )
    if not _joint_lists_equal(reversed_points[-1], initial_target):
        steps.append(
            make_motion_step(
                'recover_initial_check_target',
                MOVEJ,
                joint_target=initial_target,
                **movej_config,
            )
        )
    return steps
