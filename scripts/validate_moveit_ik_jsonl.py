#!/usr/bin/env python3
"""用 JSONL 日志验证 MoveIt IK 与记录关节的一致性。"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import rclpy
from geometry_msgs.msg import PoseStamped
from moveit_msgs.msg import RobotState
from moveit_msgs.srv import GetPositionFK
from moveit_msgs.srv import GetPositionIK
from rclpy.duration import Duration
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


@dataclass(frozen=True)
class SampleFrame:
    index: int
    cartesian: list[float]
    joints: list[float]


def rpy_to_quaternion(roll: float, pitch: float, yaw: float) -> tuple[float, ...]:
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


def make_pose_stamped(values: list[float], frame_id: str) -> PoseStamped:
    x, y, z, roll, pitch, yaw = [float(value) for value in values]
    qx, qy, qz, qw = rpy_to_quaternion(roll, pitch, yaw)
    msg = PoseStamped()
    msg.header.frame_id = frame_id
    msg.pose.position.x = x
    msg.pose.position.y = y
    msg.pose.position.z = z
    msg.pose.orientation.x = qx
    msg.pose.orientation.y = qy
    msg.pose.orientation.z = qz
    msg.pose.orientation.w = qw
    return msg


def wrap_to_pi(value: float) -> float:
    while value > math.pi:
        value -= 2.0 * math.pi
    while value < -math.pi:
        value += 2.0 * math.pi
    return value


def angular_abs_diff(left: float, right: float) -> float:
    return abs(wrap_to_pi(float(left) - float(right)))


def evenly_sample_indices(total: int, sample_count: int) -> list[int]:
    if total <= 0:
        return []
    if sample_count >= total:
        return list(range(total))
    if sample_count <= 1:
        return [0]
    return sorted(
        {
            int(round(index * (total - 1) / (sample_count - 1)))
            for index in range(sample_count)
        }
    )


def load_sample_frames(
    path: Path,
    *,
    sample_count: int,
    robot_key: str,
) -> tuple[list[SampleFrame], int]:
    raw_frames: list[SampleFrame] = []
    with path.open('r', encoding='utf-8') as handle:
        for index, line in enumerate(handle):
            record = json.loads(line)
            robot = record.get(robot_key, {})
            cartesian = robot.get('Cartesian')
            joints = robot.get('Joint')
            if (
                isinstance(cartesian, list)
                and len(cartesian) >= 6
                and isinstance(joints, list)
                and len(joints) >= 7
            ):
                raw_frames.append(
                    SampleFrame(
                        index=index,
                        cartesian=[float(value) for value in cartesian[:6]],
                        joints=[float(value) for value in joints[:7]],
                    )
                )

    sample_indices = set(evenly_sample_indices(len(raw_frames), sample_count))
    sampled = [frame for idx, frame in enumerate(raw_frames) if idx in sample_indices]
    return sampled, len(raw_frames)


def make_robot_state(
    node,
    joint_names: list[str],
    joint_positions: list[float],
    frame_id: str,
) -> RobotState:
    joint_state = JointState()
    joint_state.header.stamp = node.get_clock().now().to_msg()
    joint_state.header.frame_id = frame_id
    joint_state.name = list(joint_names)
    joint_state.position = [float(value) for value in joint_positions]
    robot_state = RobotState()
    robot_state.joint_state = joint_state
    return robot_state


def format_values(values: list[float]) -> str:
    return '[' + ', '.join(f'{float(value):.6f}' for value in values) + ']'


def quaternion_to_rpy(x: float, y: float, z: float, w: float) -> list[float]:
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1.0:
        pitch = math.copysign(math.pi / 2.0, sinp)
    else:
        pitch = math.asin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return [roll, pitch, yaw]


def compare_pose_error(
    recorded_cartesian: list[float],
    fk_cartesian: list[float],
) -> tuple[float, float]:
    position_error = math.sqrt(
        sum(
            (float(fk_value) - float(recorded_value)) ** 2
            for fk_value, recorded_value in zip(
                fk_cartesian[:3],
                recorded_cartesian[:3],
            )
        )
    )
    orientation_error = max(
        abs(wrap_to_pi(float(fk_value) - float(recorded_value)))
        for fk_value, recorded_value in zip(
            fk_cartesian[3:],
            recorded_cartesian[3:],
        )
    )
    return position_error, orientation_error


def main() -> int:
    parser = argparse.ArgumentParser(
        description='Validate MoveIt IK against recorded JEARM JSONL data.'
    )
    parser.add_argument(
        '--jsonl-path',
        required=True,
        help='Path to robot_state_*.jsonl',
    )
    parser.add_argument(
        '--sample-count',
        type=int,
        default=200,
        help='Evenly sampled frame count across the whole file.',
    )
    parser.add_argument(
        '--robot-key',
        default='Robot0',
        help='Robot key inside the JSONL file.',
    )
    parser.add_argument(
        '--service-name',
        default='/pcb_moveit/compute_ik',
        help='MoveIt service name. For --mode ik it should be /compute_ik, for --mode fk it should be /compute_fk.',
    )
    parser.add_argument(
        '--group-name',
        default='jearm_arm',
        help='MoveIt planning group name.',
    )
    parser.add_argument(
        '--ik-link-name',
        default='Link7',
        help='Tip link used for IK.',
    )
    parser.add_argument(
        '--frame-id',
        default='base_link',
        help='Reference frame of the Cartesian data.',
    )
    parser.add_argument(
        '--timeout-sec',
        type=float,
        default=2.0,
        help='MoveIt IK request timeout in seconds.',
    )
    parser.add_argument(
        '--print-worst',
        type=int,
        default=10,
        help='How many worst samples to print.',
    )
    parser.add_argument(
        '--progress-every',
        type=int,
        default=10,
        help='Print progress every N sampled frames. Use 0 to disable.',
    )
    parser.add_argument(
        '--mode',
        choices=['ik', 'fk'],
        default='ik',
        help='Validation mode: ik compares IK output against recorded joints; fk compares FK output against recorded Cartesian pose.',
    )
    args = parser.parse_args()

    path = Path(args.jsonl_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f'JSONL file does not exist: {path}')

    sampled_frames, total_valid_frames = load_sample_frames(
        path,
        sample_count=max(args.sample_count, 1),
        robot_key=args.robot_key,
    )
    if not sampled_frames:
        raise RuntimeError(f'No valid frames found in {path}')

    rclpy.init()
    node = rclpy.create_node('validate_moveit_ik_jsonl')
    try:
        if args.mode == 'ik':
            client = node.create_client(GetPositionIK, args.service_name)
        else:
            client = node.create_client(GetPositionFK, args.service_name)
        if not client.wait_for_service(timeout_sec=10.0):
            raise RuntimeError(
                f'MoveIt service unavailable: {args.service_name}'
            )

        total_success = 0
        failure_count = 0
        per_frame_stats: list[tuple[float, float, int, list[float], list[float]]] = []

        for sample_index, frame in enumerate(sampled_frames, start=1):
            if args.progress_every > 0 and (
                sample_index == 1 or sample_index % args.progress_every == 0
            ):
                print(
                    f'progress: sample={sample_index}/{len(sampled_frames)} '
                    f'frame_index={frame.index}',
                    flush=True,
                )
            if args.mode == 'ik':
                request = GetPositionIK.Request()
                request.ik_request.group_name = args.group_name
                request.ik_request.robot_state = make_robot_state(
                    node,
                    DEFAULT_JOINT_NAMES,
                    frame.joints,
                    args.frame_id,
                )
                request.ik_request.pose_stamped = make_pose_stamped(
                    frame.cartesian,
                    args.frame_id,
                )
                request.ik_request.ik_link_name = args.ik_link_name
                if hasattr(request.ik_request, 'attempts'):
                    request.ik_request.attempts = 1
                if hasattr(request.ik_request, 'timeout'):
                    request.ik_request.timeout = Duration(
                        seconds=max(args.timeout_sec, 0.0)
                    ).to_msg()
                if hasattr(request.ik_request, 'avoid_collisions'):
                    request.ik_request.avoid_collisions = False
            else:
                request = GetPositionFK.Request()
                request.header.frame_id = args.frame_id
                request.fk_link_names = [args.ik_link_name]
                request.robot_state = make_robot_state(
                    node,
                    DEFAULT_JOINT_NAMES,
                    frame.joints,
                    args.frame_id,
                )

            future = client.call_async(request)
            rclpy.spin_until_future_complete(
                node,
                future,
                timeout_sec=max(args.timeout_sec + 1.0, 1.0),
            )
            if not future.done() or future.result() is None:
                failure_count += 1
                continue

            response = future.result()
            if response.error_code.val != response.error_code.SUCCESS:
                failure_count += 1
                continue

            if args.mode == 'ik':
                result_names = list(response.solution.joint_state.name)
                result_positions = list(response.solution.joint_state.position)
                if len(result_names) != len(result_positions):
                    failure_count += 1
                    continue

                joint_map = {
                    str(name): float(value)
                    for name, value in zip(result_names, result_positions)
                }
                if any(name not in joint_map for name in DEFAULT_JOINT_NAMES):
                    failure_count += 1
                    continue

                resolved = [joint_map[name] for name in DEFAULT_JOINT_NAMES]
                diffs = [
                    angular_abs_diff(ik_value, recorded_value)
                    for ik_value, recorded_value in zip(resolved, frame.joints)
                ]
                max_diff = max(diffs)
                mean_diff = sum(diffs) / len(diffs)
                per_frame_stats.append(
                    (
                        max_diff,
                        mean_diff,
                        frame.index,
                        frame.joints,
                        resolved,
                    )
                )
                total_success += 1
            else:
                if not response.pose_stamped:
                    failure_count += 1
                    continue
                pose = response.pose_stamped[0].pose
                resolved = [
                    float(pose.position.x),
                    float(pose.position.y),
                    float(pose.position.z),
                    *quaternion_to_rpy(
                        float(pose.orientation.x),
                        float(pose.orientation.y),
                        float(pose.orientation.z),
                        float(pose.orientation.w),
                    ),
                ]
                position_error, orientation_error = compare_pose_error(
                    frame.cartesian,
                    resolved,
                )
                per_frame_stats.append(
                    (
                        position_error,
                        orientation_error,
                        frame.index,
                        frame.cartesian,
                        resolved,
                    )
                )
                total_success += 1

        print(f'jsonl_path={path}')
        print(f'total_valid_frames={total_valid_frames}')
        print(f'sampled_frames={len(sampled_frames)}')
        print(f'mode={args.mode}')
        print(f'success={total_success}')
        print(f'failure={failure_count}')

        if per_frame_stats:
            first_metric = [item[0] for item in per_frame_stats]
            second_metric = [item[1] for item in per_frame_stats]

            if args.mode == 'ik':
                print(
                    'max_abs_joint_diff_rad: '
                    f'min={min(first_metric):.6f} '
                    f'mean={sum(first_metric)/len(first_metric):.6f} '
                    f'median={sorted(first_metric)[len(first_metric)//2]:.6f} '
                    f'max={max(first_metric):.6f}'
                )
                print(
                    'mean_abs_joint_diff_rad: '
                    f'min={min(second_metric):.6f} '
                    f'mean={sum(second_metric)/len(second_metric):.6f} '
                    f'median={sorted(second_metric)[len(second_metric)//2]:.6f} '
                    f'max={max(second_metric):.6f}'
                )
                print('worst_frames:')
                worst = sorted(per_frame_stats, reverse=True)[: max(args.print_worst, 0)]
                for max_diff, mean_diff, frame_index, recorded, resolved in worst:
                    diffs = [
                        angular_abs_diff(ik_value, recorded_value)
                        for ik_value, recorded_value in zip(resolved, recorded)
                    ]
                    print(
                        f'  frame={frame_index} '
                        f'max_diff={max_diff:.6f} '
                        f'mean_diff={mean_diff:.6f}'
                    )
                    print(f'    recorded={format_values(recorded)}')
                    print(f'    ik      ={format_values(resolved)}')
                    print(f'    diff    ={format_values(diffs)}')
            else:
                print(
                    'position_error_m: '
                    f'min={min(first_metric):.6f} '
                    f'mean={sum(first_metric)/len(first_metric):.6f} '
                    f'median={sorted(first_metric)[len(first_metric)//2]:.6f} '
                    f'max={max(first_metric):.6f}'
                )
                print(
                    'orientation_error_rad: '
                    f'min={min(second_metric):.6f} '
                    f'mean={sum(second_metric)/len(second_metric):.6f} '
                    f'median={sorted(second_metric)[len(second_metric)//2]:.6f} '
                    f'max={max(second_metric):.6f}'
                )
                print('worst_frames:')
                worst = sorted(per_frame_stats, reverse=True)[: max(args.print_worst, 0)]
                for pos_err, rot_err, frame_index, recorded, resolved in worst:
                    print(
                        f'  frame={frame_index} '
                        f'pos_err={pos_err:.6f} '
                        f'rot_err={rot_err:.6f}'
                    )
                    print(f'    recorded={format_values(recorded)}')
                    print(f'    fk      ={format_values(resolved)}')

    finally:
        node.destroy_node()
        rclpy.shutdown()

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
