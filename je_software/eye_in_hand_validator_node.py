#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import matplotlib
import numpy as np
import rclpy
from common.msg import OculusControllers
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image

from je_software.calibration_common import (
    REGION_BUCKETS,
    BoardPoint,
    CharucoBoardHelper,
    CharucoDetection,
    PoseSnapshot,
    average_transforms,
    draw_detection_overlay,
    draw_projected_board_point,
    image_msg_to_bgr,
    load_eye_in_hand_calibration_result,
    make_transform,
    opencv_supports_charuco,
    pose_to_transform,
    region_bucket_name,
    rotation_error_deg,
    rotation_matrix_to_quaternion,
    rotation_matrix_to_rpy_deg,
    stamp_to_seconds,
    transform_point,
    transform_to_json_dict,
)

matplotlib.use('Agg')
from matplotlib import pyplot as plt  # noqa: E402


SUPPORTED_MODES = ('consistency', 'point_check')
TOP_WORST_COUNT = 5


@dataclass
class LiveObservation:
    image_stamp_s: float
    image_width: int
    image_height: int
    bgr_image: np.ndarray
    detection: CharucoDetection
    rvec: np.ndarray | None
    tvec: np.ndarray | None
    reproj_mean_px: float | None
    pose_snapshot: PoseSnapshot | None
    pose_offset_s: float | None
    board_center_x_px: float | None
    board_center_y_px: float | None
    center_x_norm: float | None
    center_y_norm: float | None
    center_radius_norm: float | None


@dataclass
class ConsistencySample:
    sample_index: int
    image_stamp_s: float
    pose_stamp_s: float
    pose_offset_s: float
    pose_frame_id: str
    image_width: int
    image_height: int
    charuco_count: int
    reproj_mean_px: float
    board_center_x_px: float
    board_center_y_px: float
    center_x_norm: float
    center_y_norm: float
    center_radius_norm: float
    region_bucket: str
    gripper_rpy_deg: np.ndarray
    t_base_gripper: np.ndarray
    t_cam_target: np.ndarray
    t_base_target: np.ndarray
    raw_image_path: str
    overlay_image_path: str


@dataclass
class PointLock:
    lock_id: int
    point_name: str
    point_kind: str
    point_source_id: int | None
    point_t: np.ndarray
    image_stamp_s: float
    pose_stamp_s: float
    pose_offset_s: float
    pose_frame_id: str
    image_width: int
    image_height: int
    charuco_count: int
    reproj_mean_px: float
    board_center_x_px: float
    board_center_y_px: float
    center_x_norm: float
    center_y_norm: float
    center_radius_norm: float
    region_bucket: str
    t_base_gripper_lock: np.ndarray
    t_cam_target_lock: np.ndarray
    p_base_target: np.ndarray
    raw_image_path: str
    overlay_image_path: str


@dataclass
class PointMeasurement:
    measurement_index: int
    lock_id: int
    point_name: str
    point_kind: str
    point_source_id: int | None
    pose_stamp_s: float
    pose_frame_id: str
    p_base_measured: np.ndarray
    t_base_gripper_meas: np.ndarray
    error_m: np.ndarray


def _stats_mean_std_max(values: np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0:
        return {'mean': 0.0, 'std': 0.0, 'max': 0.0}
    return {
        'mean': float(np.mean(array)),
        'std': float(np.std(array)),
        'max': float(np.max(array)),
    }


def _stats_xyz_mm(vectors_m: np.ndarray) -> dict[str, dict[str, float]]:
    vectors = np.asarray(vectors_m, dtype=np.float64).reshape(-1, 3)
    axis_names = ('x', 'y', 'z')
    stats: dict[str, dict[str, float]] = {}
    for axis_index, axis_name in enumerate(axis_names):
        axis_values_mm = vectors[:, axis_index] * 1000.0
        stats[axis_name] = {
            'mean': float(np.mean(axis_values_mm)),
            'std': float(np.std(axis_values_mm)),
            'max_abs': float(np.max(np.abs(axis_values_mm))),
        }
    return stats


def _float_list(values: np.ndarray) -> list[float]:
    return [float(value) for value in np.asarray(values, dtype=np.float64).reshape(-1)]


class EyeInHandValidatorNode(Node):
    def __init__(self) -> None:
        super().__init__('eye_in_hand_validator')

        self._declare_parameters()

        if not opencv_supports_charuco():
            raise RuntimeError(
                'Current OpenCV build does not expose cv2.aruco Charuco APIs.'
            )

        self.mode = str(self.get_parameter('mode').value).strip().lower()
        self.calibration_result_path = str(
            self.get_parameter('calibration_result_path').value
        ).strip()
        self.image_topic = str(self.get_parameter('image_topic').value)
        self.camera_info_topic = str(self.get_parameter('camera_info_topic').value)
        self.oculus_topic = str(self.get_parameter('oculus_topic').value)
        self.arm = str(self.get_parameter('arm').value).strip().lower()
        self.output_root = Path(
            str(self.get_parameter('output_dir').value)
        ).expanduser()
        self.show_preview = bool(self.get_parameter('show_preview').value)
        self.preview_max_width = int(self.get_parameter('preview_max_width').value)
        self.process_interval_s = float(
            self.get_parameter('process_interval_s').value
        )
        self.pose_sync_tolerance_s = float(
            self.get_parameter('pose_sync_tolerance_s').value
        )
        self.max_reproj_error_px = float(
            self.get_parameter('max_reproj_error_px').value
        )
        self.min_charuco_corners = int(
            self.get_parameter('min_charuco_corners').value
        )
        self.image_is_rectified = bool(
            self.get_parameter('image_is_rectified').value
        )
        self.point_set = str(self.get_parameter('point_set').value).strip()
        self.charuco_ids_csv = str(
            self.get_parameter('charuco_ids_csv').value
        ).strip()

        self._validate_parameters()
        self.calibration = load_eye_in_hand_calibration_result(
            self.calibration_result_path
        )
        self.charuco_helper = CharucoBoardHelper.from_board_config(
            self.calibration.board_config
        )
        self._extra_charuco_ids = self._parse_charuco_ids_csv()
        self.board_points = self.charuco_helper.default_point_set(
            point_set=self.point_set,
            extra_charuco_ids=self._extra_charuco_ids,
        )

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        session_name = f'eye_in_hand_validation_{self.mode}_{timestamp}'
        self.session_dir = self.output_root / session_name
        self.images_dir = self.session_dir / 'images'
        self.images_dir.mkdir(parents=True, exist_ok=True)

        self.camera_matrix: np.ndarray | None = None
        self.dist_coeffs: np.ndarray | None = None
        self.camera_info_snapshot: dict[str, Any] = {}
        self.pose_buffer: deque[PoseSnapshot] = deque(maxlen=400)
        self.pose_frame_id_live = ''
        self.camera_frame_id_live = ''
        self.last_process_time_s = 0.0
        self.total_image_msgs = 0
        self.last_live_observation: LiveObservation | None = None
        self.last_status_log_s = 0.0
        self._printed_help = False
        self._fatal_error_message = ''

        self.consistency_samples: list[ConsistencySample] = []
        self.point_locks: list[PointLock] = []
        self.point_measurements: list[PointMeasurement] = []
        self.selected_point_index = 0
        self.active_lock_id: int | None = None

        self.info_sub = self.create_subscription(
            CameraInfo,
            self.camera_info_topic,
            self._camera_info_callback,
            qos_profile_sensor_data,
        )
        self.image_sub = self.create_subscription(
            Image,
            self.image_topic,
            self._image_callback,
            qos_profile_sensor_data,
        )
        self.oculus_sub = self.create_subscription(
            OculusControllers,
            self.oculus_topic,
            self._oculus_callback,
            qos_profile_sensor_data,
        )
        self.status_timer = self.create_timer(2.0, self._status_timer_callback)

        self.get_logger().info(
            'Eye-in-hand validator started: '
            f'mode={self.mode}, '
            f'image_topic={self.image_topic}, '
            f'camera_info_topic={self.camera_info_topic}, '
            f'oculus_topic={self.oculus_topic}, '
            f'arm={self.arm}, '
            f'calibration_result={self.calibration.calibration_path}, '
            f'output_dir={self.session_dir}'
        )
        self.get_logger().info(
            'Loaded calibration frames: '
            f'pose_frame={self.calibration.pose_frame_id}, '
            f'gripper_frame={self.calibration.gripper_frame}, '
            f'camera_frame={self.calibration.camera_frame_id}'
        )
        if self.mode == 'point_check':
            point_names = ', '.join(point.name for point in self.board_points)
            self.get_logger().info(f'Point set: {point_names}')
        self._log_help_once()

    def destroy_node(self) -> bool:
        if self.show_preview:
            cv2.destroyAllWindows()
        return super().destroy_node()

    def _declare_parameters(self) -> None:
        self.declare_parameter('mode', 'consistency')
        self.declare_parameter('calibration_result_path', '')
        self.declare_parameter('image_topic', '/camera/color/image_raw')
        self.declare_parameter(
            'camera_info_topic',
            '/camera/color/camera_info',
        )
        self.declare_parameter('oculus_topic', '/oculus_controllers')
        self.declare_parameter('arm', 'left')
        self.declare_parameter(
            'output_dir',
            str(Path.home() / 'eye_in_hand_validation'),
        )
        self.declare_parameter('show_preview', True)
        self.declare_parameter('preview_max_width', 1280)
        self.declare_parameter('image_is_rectified', True)
        self.declare_parameter('pose_sync_tolerance_s', 0.08)
        self.declare_parameter('process_interval_s', 0.15)
        self.declare_parameter('max_reproj_error_px', 3.0)
        self.declare_parameter('min_charuco_corners', 8)
        self.declare_parameter('point_set', 'center_corners')
        self.declare_parameter('charuco_ids_csv', '')

    def _validate_parameters(self) -> None:
        if self.mode not in SUPPORTED_MODES:
            supported = ', '.join(SUPPORTED_MODES)
            raise ValueError(f'Unsupported mode. Supported values: {supported}.')
        if not self.calibration_result_path:
            raise ValueError('calibration_result_path must not be empty.')
        if self.arm not in ('left', 'right'):
            raise ValueError("arm must be 'left' or 'right'.")
        if self.process_interval_s <= 0.0:
            raise ValueError('process_interval_s must be positive.')
        if self.pose_sync_tolerance_s <= 0.0:
            raise ValueError('pose_sync_tolerance_s must be positive.')
        if self.max_reproj_error_px <= 0.0:
            raise ValueError('max_reproj_error_px must be positive.')
        if self.min_charuco_corners < 4:
            raise ValueError('min_charuco_corners must be >= 4.')
        if self.point_set != 'center_corners':
            raise ValueError("Only point_set='center_corners' is supported.")

    def _parse_charuco_ids_csv(self) -> list[int]:
        if not self.charuco_ids_csv:
            return []
        values = []
        max_id = int(self.charuco_helper.charuco_corner_object_points.shape[0]) - 1
        for token in self.charuco_ids_csv.split(','):
            token = token.strip()
            if not token:
                continue
            charuco_id = int(token)
            if charuco_id < 0 or charuco_id > max_id:
                raise ValueError(
                    f'charuco id {charuco_id} out of range [0, {max_id}].'
                )
            values.append(charuco_id)
        return values

    def _camera_info_callback(self, msg: CameraInfo) -> None:
        if len(msg.k) != 9:
            self.get_logger().warn('Ignoring CameraInfo with invalid K size.')
            return

        raw_camera_matrix = np.asarray(msg.k, dtype=np.float64).reshape(3, 3)
        raw_dist_coeffs = np.asarray(msg.d, dtype=np.float64).reshape(-1, 1)
        projection = np.asarray(msg.p, dtype=np.float64).reshape(3, 4)

        if self.image_is_rectified and not np.allclose(projection[:, :3], 0.0):
            self.camera_matrix = projection[:, :3].copy()
            self.dist_coeffs = np.zeros((5, 1), dtype=np.float64)
        else:
            self.camera_matrix = raw_camera_matrix
            self.dist_coeffs = raw_dist_coeffs

        self.camera_frame_id_live = str(msg.header.frame_id)
        self.camera_info_snapshot = {
            'frame_id': self.camera_frame_id_live,
            'width': int(msg.width),
            'height': int(msg.height),
            'distortion_model': str(msg.distortion_model),
            'k': [float(value) for value in msg.k],
            'd': [float(value) for value in msg.d],
            'p': [float(value) for value in msg.p],
            'used_rectified_model': bool(self.image_is_rectified),
        }
        if self.camera_frame_id_live != self.calibration.camera_frame_id:
            self._shutdown_with_error(
                'camera_frame_id mismatch: '
                f'live={self.camera_frame_id_live}, '
                f'calibration={self.calibration.camera_frame_id}'
            )

    def _oculus_callback(self, msg: OculusControllers) -> None:
        use_left = self.arm == 'left'
        pose_valid = bool(msg.left_valid) if use_left else bool(msg.right_valid)
        if not pose_valid:
            return

        pose = msg.left_pose if use_left else msg.right_pose
        try:
            transform = pose_to_transform(pose)
        except ValueError as exc:
            self.get_logger().warn(f'Ignoring invalid {self.arm} pose: {exc}')
            return

        stamp_s = stamp_to_seconds(msg.header.stamp)
        frame_id = str(msg.header.frame_id)
        self.pose_frame_id_live = frame_id
        if frame_id != self.calibration.pose_frame_id:
            self._shutdown_with_error(
                'pose_frame_id mismatch: '
                f'live={frame_id}, calibration={self.calibration.pose_frame_id}'
            )
            return

        self.pose_buffer.append(
            PoseSnapshot(
                stamp_s=stamp_s,
                frame_id=frame_id,
                transform=transform,
            )
        )

    def _image_callback(self, msg: Image) -> None:
        if self._fatal_error_message:
            return

        self.total_image_msgs += 1
        now_s = time.monotonic()
        if now_s - self.last_process_time_s < self.process_interval_s:
            return
        self.last_process_time_s = now_s

        try:
            bgr_image = image_msg_to_bgr(msg)
        except ValueError as exc:
            self.get_logger().warn(f'Unsupported image encoding: {exc}')
            return

        preview_image = bgr_image.copy()
        observation = self._build_live_observation(msg, bgr_image)
        self.last_live_observation = observation

        if observation.detection.marker_ids is not None:
            draw_detection_overlay(preview_image, observation.detection)
        if observation.rvec is not None and observation.tvec is not None:
            assert self.camera_matrix is not None
            assert self.dist_coeffs is not None
            cv2.drawFrameAxes(
                preview_image,
                self.camera_matrix,
                self.dist_coeffs,
                observation.rvec,
                observation.tvec,
                self.charuco_helper.square_length_m * 2.0,
                2,
            )
            if self.mode == 'point_check':
                selected = self.board_points[self.selected_point_index]
                draw_projected_board_point(
                    preview_image,
                    board_helper=self.charuco_helper,
                    point_t=selected.point_t,
                    rvec=observation.rvec,
                    tvec=observation.tvec,
                    camera_matrix=self.camera_matrix,
                    dist_coeffs=self.dist_coeffs,
                    label=selected.name,
                )

        self._draw_status_overlay(preview_image, observation)
        if self.show_preview:
            display_image = self._resize_for_preview(preview_image)
            cv2.imshow('Eye-In-Hand Validator', display_image)
            self._handle_preview_key(observation, preview_image)

    def _build_live_observation(
        self,
        msg: Image,
        bgr_image: np.ndarray,
    ) -> LiveObservation:
        gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        detection = self.charuco_helper.detect(
            gray_image,
            camera_matrix=self.camera_matrix,
            dist_coeffs=self.dist_coeffs,
        )

        rvec: np.ndarray | None = None
        tvec: np.ndarray | None = None
        reproj_mean_px: float | None = None
        if (
            self.camera_matrix is not None
            and self.dist_coeffs is not None
            and detection.charuco_corners is not None
            and detection.charuco_ids is not None
            and int(detection.charuco_ids.shape[0]) >= self.min_charuco_corners
        ):
            solve_result = self.charuco_helper.solve_target_pose(
                charuco_corners=detection.charuco_corners,
                charuco_ids=detection.charuco_ids,
                camera_matrix=self.camera_matrix,
                dist_coeffs=self.dist_coeffs,
            )
            if solve_result is not None:
                rvec, tvec, reproj_mean_px = solve_result

        board_center_x_px: float | None = None
        board_center_y_px: float | None = None
        center_x_norm: float | None = None
        center_y_norm: float | None = None
        center_radius_norm: float | None = None
        if (
            detection.charuco_corners is not None
            and int(detection.charuco_corners.shape[0]) > 0
        ):
            mean_xy = np.mean(
                detection.charuco_corners.reshape(-1, 2),
                axis=0,
            )
            board_center_x_px = float(mean_xy[0])
            board_center_y_px = float(mean_xy[1])
            center_x_norm = board_center_x_px / float(max(1, msg.width))
            center_y_norm = board_center_y_px / float(max(1, msg.height))
            radius = math.sqrt(
                (center_x_norm - 0.5) ** 2 + (center_y_norm - 0.5) ** 2
            )
            center_radius_norm = float(radius / math.sqrt(0.5 ** 2 + 0.5 ** 2))

        image_stamp_s = stamp_to_seconds(msg.header.stamp)
        pose_snapshot, pose_offset_s = self._lookup_pose(image_stamp_s)
        return LiveObservation(
            image_stamp_s=image_stamp_s,
            image_width=int(msg.width),
            image_height=int(msg.height),
            bgr_image=bgr_image,
            detection=detection,
            rvec=rvec,
            tvec=tvec,
            reproj_mean_px=reproj_mean_px,
            pose_snapshot=pose_snapshot,
            pose_offset_s=pose_offset_s,
            board_center_x_px=board_center_x_px,
            board_center_y_px=board_center_y_px,
            center_x_norm=center_x_norm,
            center_y_norm=center_y_norm,
            center_radius_norm=center_radius_norm,
        )

    def _lookup_pose(
        self,
        image_stamp_s: float,
    ) -> tuple[PoseSnapshot | None, float | None]:
        if not self.pose_buffer:
            return None, None
        best_pose = min(
            self.pose_buffer,
            key=lambda candidate: abs(candidate.stamp_s - image_stamp_s),
        )
        offset_s = abs(best_pose.stamp_s - image_stamp_s)
        if offset_s > self.pose_sync_tolerance_s:
            return None, offset_s
        return best_pose, offset_s

    def _latest_pose(self) -> PoseSnapshot | None:
        if not self.pose_buffer:
            return None
        return self.pose_buffer[-1]

    def _status_timer_callback(self) -> None:
        if self._fatal_error_message:
            return
        details = [
            f'camera_info={"yes" if self.camera_matrix is not None else "no"}',
            f'pose_buffer={len(self.pose_buffer)}',
            f'image_msgs={self.total_image_msgs}',
        ]
        if self.mode == 'consistency':
            details.append(f'samples={len(self.consistency_samples)}')
        else:
            details.append(f'locks={len(self.point_locks)}')
            details.append(f'measurements={len(self.point_measurements)}')
            details.append(f'selected={self.board_points[self.selected_point_index].name}')
            details.append(
                f'active_lock={self.active_lock_id if self.active_lock_id is not None else "none"}'
            )
        self.get_logger().info('Status: ' + ', '.join(details))

    def _draw_status_overlay(
        self,
        image: np.ndarray,
        observation: LiveObservation,
    ) -> None:
        marker_count = 0
        if observation.detection.marker_ids is not None:
            marker_count = int(observation.detection.marker_ids.shape[0])

        lines = [
            f'mode: {self.mode}',
            f'camera_info: {"ready" if self.camera_matrix is not None else "waiting"}',
            f'pose_buffer: {len(self.pose_buffer)}  image_msgs: {self.total_image_msgs}',
            f'markers: {marker_count}  charuco: {observation.detection.charuco_count}',
            f'pose_frame: {self.pose_frame_id_live or "waiting"}',
            f'camera_frame: {self.camera_frame_id_live or "waiting"}',
            f'calib gripper_frame: {self.calibration.gripper_frame}',
        ]
        if observation.reproj_mean_px is not None:
            lines.append(f'reproj: {observation.reproj_mean_px:.3f}px')
        else:
            lines.append('reproj: n/a')
        if observation.pose_offset_s is not None:
            lines.append(f'pose_dt: {observation.pose_offset_s * 1000.0:.1f} ms')
        else:
            lines.append('pose_dt: n/a')

        if self.mode == 'consistency':
            lines.append(f'captured_samples: {len(self.consistency_samples)}')
            lines.append('keys: s/space=sample  c=report  r=reset  q=quit')
        else:
            selected = self.board_points[self.selected_point_index]
            lines.append(f'selected_point: {selected.name}')
            lines.append(f'locks: {len(self.point_locks)}  measures: {len(self.point_measurements)}')
            lines.append(
                'keys: n/p=point  l=lock  m=measure  c=report  r=reset  q=quit'
            )

        origin_x = 16
        origin_y = 28
        for index, line in enumerate(lines):
            y_value = origin_y + index * 24
            cv2.putText(
                image,
                line,
                (origin_x, y_value),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                image,
                line,
                (origin_x, y_value),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (30, 30, 30),
                1,
                cv2.LINE_AA,
            )

    def _resize_for_preview(self, image: np.ndarray) -> np.ndarray:
        if image.shape[1] <= self.preview_max_width:
            return image
        scale = self.preview_max_width / float(image.shape[1])
        return cv2.resize(
            image,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_AREA,
        )

    def _handle_preview_key(
        self,
        observation: LiveObservation,
        preview_image: np.ndarray,
    ) -> None:
        key = cv2.waitKey(1) & 0xFF
        if key == 255:
            return
        if key in (ord('q'), 27):
            self.get_logger().info('User requested exit.')
            if rclpy.ok():
                rclpy.shutdown()
            return
        if key == ord('h'):
            self._log_help_once(force=True)
            return
        if key == ord('r'):
            self._reset_session()
            return

        if self.mode == 'consistency':
            if key in (ord('s'), ord(' ')):
                self._capture_consistency_sample(observation, preview_image)
            elif key == ord('c'):
                self._generate_consistency_report()
        else:
            if key == ord('n'):
                self.selected_point_index = (
                    self.selected_point_index + 1
                ) % len(self.board_points)
                self.get_logger().info(
                    f'Selected point: {self.board_points[self.selected_point_index].name}'
                )
            elif key == ord('p'):
                self.selected_point_index = (
                    self.selected_point_index - 1
                ) % len(self.board_points)
                self.get_logger().info(
                    f'Selected point: {self.board_points[self.selected_point_index].name}'
                )
            elif key == ord('l'):
                self._lock_point(observation, preview_image)
            elif key == ord('m'):
                self._record_point_measurement()
            elif key == ord('c'):
                self._generate_point_check_report()

    def _reset_session(self) -> None:
        if self.mode == 'consistency':
            self.consistency_samples.clear()
            self._write_consistency_samples_snapshot()
            self.get_logger().info('Cleared consistency samples.')
            return
        self.point_locks.clear()
        self.point_measurements.clear()
        self.active_lock_id = None
        self._write_point_lock_snapshot()
        self._write_point_measurement_snapshot()
        self.get_logger().info('Cleared point-check locks and measurements.')

    def _capture_consistency_sample(
        self,
        observation: LiveObservation,
        preview_image: np.ndarray,
    ) -> None:
        rejection = self._validate_visual_sample(observation)
        if rejection is not None:
            self.get_logger().warn(f'Capture rejected: {rejection}')
            return

        assert observation.rvec is not None
        assert observation.tvec is not None
        assert observation.pose_snapshot is not None
        assert observation.reproj_mean_px is not None
        assert observation.board_center_x_px is not None
        assert observation.board_center_y_px is not None
        assert observation.center_x_norm is not None
        assert observation.center_y_norm is not None
        assert observation.center_radius_norm is not None

        rotation_ct, _ = cv2.Rodrigues(observation.rvec)
        t_cam_target = make_transform(rotation_ct, observation.tvec.reshape(3))
        t_base_target = (
            observation.pose_snapshot.transform
            @ self.calibration.t_gripper_cam
            @ t_cam_target
        )
        region_bucket = region_bucket_name(
            observation.center_x_norm,
            observation.center_y_norm,
        )
        gripper_rpy_deg = rotation_matrix_to_rpy_deg(
            observation.pose_snapshot.transform[:3, :3]
        )

        sample_index = len(self.consistency_samples)
        raw_image_path = self.images_dir / f'consistency_sample_{sample_index:03d}_raw.png'
        overlay_image_path = (
            self.images_dir / f'consistency_sample_{sample_index:03d}_overlay.png'
        )
        cv2.imwrite(str(raw_image_path), observation.bgr_image)
        cv2.imwrite(str(overlay_image_path), preview_image)

        sample = ConsistencySample(
            sample_index=sample_index,
            image_stamp_s=observation.image_stamp_s,
            pose_stamp_s=observation.pose_snapshot.stamp_s,
            pose_offset_s=float(observation.pose_offset_s or 0.0),
            pose_frame_id=observation.pose_snapshot.frame_id,
            image_width=observation.image_width,
            image_height=observation.image_height,
            charuco_count=int(observation.detection.charuco_count),
            reproj_mean_px=float(observation.reproj_mean_px),
            board_center_x_px=float(observation.board_center_x_px),
            board_center_y_px=float(observation.board_center_y_px),
            center_x_norm=float(observation.center_x_norm),
            center_y_norm=float(observation.center_y_norm),
            center_radius_norm=float(observation.center_radius_norm),
            region_bucket=region_bucket,
            gripper_rpy_deg=gripper_rpy_deg,
            t_base_gripper=observation.pose_snapshot.transform.copy(),
            t_cam_target=t_cam_target,
            t_base_target=t_base_target,
            raw_image_path=str(raw_image_path),
            overlay_image_path=str(overlay_image_path),
        )
        self.consistency_samples.append(sample)
        self._write_consistency_samples_snapshot()
        self.get_logger().info(
            'Captured consistency sample '
            f'#{sample_index}: '
            f'reproj={sample.reproj_mean_px:.3f}px, '
            f'pose_dt={sample.pose_offset_s * 1000.0:.1f}ms, '
            f'region={sample.region_bucket}'
        )

    def _lock_point(
        self,
        observation: LiveObservation,
        preview_image: np.ndarray,
    ) -> None:
        rejection = self._validate_visual_sample(observation)
        if rejection is not None:
            self.get_logger().warn(f'Lock rejected: {rejection}')
            return

        selected = self.board_points[self.selected_point_index]
        assert observation.rvec is not None
        assert observation.tvec is not None
        assert observation.pose_snapshot is not None
        assert observation.reproj_mean_px is not None
        assert observation.board_center_x_px is not None
        assert observation.board_center_y_px is not None
        assert observation.center_x_norm is not None
        assert observation.center_y_norm is not None
        assert observation.center_radius_norm is not None

        rotation_ct, _ = cv2.Rodrigues(observation.rvec)
        t_cam_target = make_transform(rotation_ct, observation.tvec.reshape(3))
        p_base_target = transform_point(
            observation.pose_snapshot.transform
            @ self.calibration.t_gripper_cam
            @ t_cam_target,
            selected.point_t,
        )
        region_bucket = region_bucket_name(
            observation.center_x_norm,
            observation.center_y_norm,
        )

        lock_id = len(self.point_locks)
        raw_image_path = self.images_dir / f'point_lock_{lock_id:03d}_raw.png'
        overlay_image_path = self.images_dir / f'point_lock_{lock_id:03d}_overlay.png'
        cv2.imwrite(str(raw_image_path), observation.bgr_image)
        cv2.imwrite(str(overlay_image_path), preview_image)

        lock = PointLock(
            lock_id=lock_id,
            point_name=selected.name,
            point_kind=selected.kind,
            point_source_id=selected.source_id,
            point_t=selected.point_t.copy(),
            image_stamp_s=observation.image_stamp_s,
            pose_stamp_s=observation.pose_snapshot.stamp_s,
            pose_offset_s=float(observation.pose_offset_s or 0.0),
            pose_frame_id=observation.pose_snapshot.frame_id,
            image_width=observation.image_width,
            image_height=observation.image_height,
            charuco_count=int(observation.detection.charuco_count),
            reproj_mean_px=float(observation.reproj_mean_px),
            board_center_x_px=float(observation.board_center_x_px),
            board_center_y_px=float(observation.board_center_y_px),
            center_x_norm=float(observation.center_x_norm),
            center_y_norm=float(observation.center_y_norm),
            center_radius_norm=float(observation.center_radius_norm),
            region_bucket=region_bucket,
            t_base_gripper_lock=observation.pose_snapshot.transform.copy(),
            t_cam_target_lock=t_cam_target,
            p_base_target=p_base_target,
            raw_image_path=str(raw_image_path),
            overlay_image_path=str(overlay_image_path),
        )
        self.point_locks.append(lock)
        self.active_lock_id = lock.lock_id
        self._write_point_lock_snapshot()
        self.get_logger().info(
            'Locked point '
            f'#{lock.lock_id}: {lock.point_name}, '
            f'target_base=[{lock.p_base_target[0]:.6f}, '
            f'{lock.p_base_target[1]:.6f}, {lock.p_base_target[2]:.6f}] m'
        )

    def _record_point_measurement(self) -> None:
        active_lock = self._active_lock()
        if active_lock is None:
            self.get_logger().warn('Measurement rejected: no active lock.')
            return

        pose_snapshot = self._latest_pose()
        if pose_snapshot is None:
            self.get_logger().warn('Measurement rejected: no live end-effector pose.')
            return
        if pose_snapshot.frame_id != self.calibration.pose_frame_id:
            self._shutdown_with_error(
                'pose_frame_id mismatch during measurement: '
                f'live={pose_snapshot.frame_id}, '
                f'calibration={self.calibration.pose_frame_id}'
            )
            return

        p_base_measured = pose_snapshot.transform[:3, 3].copy()
        error_m = p_base_measured - active_lock.p_base_target
        measurement = PointMeasurement(
            measurement_index=len(self.point_measurements),
            lock_id=active_lock.lock_id,
            point_name=active_lock.point_name,
            point_kind=active_lock.point_kind,
            point_source_id=active_lock.point_source_id,
            pose_stamp_s=pose_snapshot.stamp_s,
            pose_frame_id=pose_snapshot.frame_id,
            p_base_measured=p_base_measured,
            t_base_gripper_meas=pose_snapshot.transform.copy(),
            error_m=error_m,
        )
        self.point_measurements.append(measurement)
        self._write_point_measurement_snapshot()
        self.get_logger().info(
            'Recorded measurement '
            f'#{measurement.measurement_index} for lock #{measurement.lock_id}: '
            f'error_mm=[{error_m[0] * 1000.0:.2f}, '
            f'{error_m[1] * 1000.0:.2f}, {error_m[2] * 1000.0:.2f}], '
            f'norm={np.linalg.norm(error_m) * 1000.0:.2f}'
        )

    def _validate_visual_sample(
        self,
        observation: LiveObservation,
    ) -> str | None:
        if observation.rvec is None or observation.tvec is None:
            return 'no valid board pose in current frame'
        if observation.reproj_mean_px is None:
            return 'missing reprojection error'
        if observation.reproj_mean_px > self.max_reproj_error_px:
            return (
                f'reprojection error {observation.reproj_mean_px:.3f}px > '
                f'{self.max_reproj_error_px:.3f}px'
            )
        if observation.pose_snapshot is None:
            return 'no synchronized end-effector pose'
        if observation.pose_snapshot.frame_id != self.calibration.pose_frame_id:
            return (
                'pose_frame_id mismatch: '
                f'live={observation.pose_snapshot.frame_id}, '
                f'calibration={self.calibration.pose_frame_id}'
            )
        if observation.board_center_x_px is None or observation.board_center_y_px is None:
            return 'board center is unavailable'
        return None

    def _write_consistency_samples_snapshot(
        self,
        per_sample_metrics: dict[int, dict[str, Any]] | None = None,
    ) -> None:
        payload = {
            'created_at': datetime.now().isoformat(),
            'mode': 'consistency',
            'calibration_result_path': str(self.calibration.calibration_path),
            'image_topic': self.image_topic,
            'camera_info_topic': self.camera_info_topic,
            'oculus_topic': self.oculus_topic,
            'arm': self.arm,
            'pose_frame_id': self.calibration.pose_frame_id,
            'gripper_frame': self.calibration.gripper_frame,
            'camera_frame_id': self.calibration.camera_frame_id,
            'sample_count': len(self.consistency_samples),
            'samples': [
                self._consistency_sample_to_json(sample, per_sample_metrics)
                for sample in self.consistency_samples
            ],
        }
        output_path = self.session_dir / 'consistency_samples.json'
        with output_path.open('w', encoding='utf-8') as file_obj:
            json.dump(payload, file_obj, ensure_ascii=True, indent=2)

    def _consistency_sample_to_json(
        self,
        sample: ConsistencySample,
        per_sample_metrics: dict[int, dict[str, Any]] | None,
    ) -> dict[str, Any]:
        payload = {
            'sample_index': int(sample.sample_index),
            'image_stamp_s': float(sample.image_stamp_s),
            'pose_stamp_s': float(sample.pose_stamp_s),
            'pose_offset_s': float(sample.pose_offset_s),
            'pose_frame_id': sample.pose_frame_id,
            'image_width': int(sample.image_width),
            'image_height': int(sample.image_height),
            'charuco_count': int(sample.charuco_count),
            'reproj_mean_px': float(sample.reproj_mean_px),
            'board_center_x_px': float(sample.board_center_x_px),
            'board_center_y_px': float(sample.board_center_y_px),
            'center_x_norm': float(sample.center_x_norm),
            'center_y_norm': float(sample.center_y_norm),
            'center_radius_norm': float(sample.center_radius_norm),
            'region_bucket': sample.region_bucket,
            'gripper_rpy_deg': _float_list(sample.gripper_rpy_deg),
            't_base_gripper': transform_to_json_dict(sample.t_base_gripper),
            't_cam_target': transform_to_json_dict(sample.t_cam_target),
            't_base_target': transform_to_json_dict(sample.t_base_target),
            'raw_image_path': sample.raw_image_path,
            'overlay_image_path': sample.overlay_image_path,
        }
        if per_sample_metrics and sample.sample_index in per_sample_metrics:
            payload['report_metrics'] = per_sample_metrics[sample.sample_index]
        return payload

    def _write_point_lock_snapshot(self) -> None:
        payload = {
            'created_at': datetime.now().isoformat(),
            'mode': 'point_check',
            'calibration_result_path': str(self.calibration.calibration_path),
            'image_topic': self.image_topic,
            'camera_info_topic': self.camera_info_topic,
            'oculus_topic': self.oculus_topic,
            'arm': self.arm,
            'pose_frame_id': self.calibration.pose_frame_id,
            'gripper_frame': self.calibration.gripper_frame,
            'camera_frame_id': self.calibration.camera_frame_id,
            'active_lock_id': self.active_lock_id,
            'lock_count': len(self.point_locks),
            'locks': [self._point_lock_to_json(lock) for lock in self.point_locks],
        }
        output_path = self.session_dir / 'point_check_locks.json'
        with output_path.open('w', encoding='utf-8') as file_obj:
            json.dump(payload, file_obj, ensure_ascii=True, indent=2)

    def _point_lock_to_json(self, lock: PointLock) -> dict[str, Any]:
        return {
            'lock_id': int(lock.lock_id),
            'point_name': lock.point_name,
            'point_kind': lock.point_kind,
            'point_source_id': lock.point_source_id,
            'point_t': _float_list(lock.point_t),
            'image_stamp_s': float(lock.image_stamp_s),
            'pose_stamp_s': float(lock.pose_stamp_s),
            'pose_offset_s': float(lock.pose_offset_s),
            'pose_frame_id': lock.pose_frame_id,
            'image_width': int(lock.image_width),
            'image_height': int(lock.image_height),
            'charuco_count': int(lock.charuco_count),
            'reproj_mean_px': float(lock.reproj_mean_px),
            'board_center_x_px': float(lock.board_center_x_px),
            'board_center_y_px': float(lock.board_center_y_px),
            'center_x_norm': float(lock.center_x_norm),
            'center_y_norm': float(lock.center_y_norm),
            'center_radius_norm': float(lock.center_radius_norm),
            'region_bucket': lock.region_bucket,
            't_base_gripper_lock': transform_to_json_dict(lock.t_base_gripper_lock),
            't_cam_target_lock': transform_to_json_dict(lock.t_cam_target_lock),
            'p_base_target': _float_list(lock.p_base_target),
            'raw_image_path': lock.raw_image_path,
            'overlay_image_path': lock.overlay_image_path,
        }

    def _write_point_measurement_snapshot(self) -> None:
        payload = {
            'created_at': datetime.now().isoformat(),
            'mode': 'point_check',
            'calibration_result_path': str(self.calibration.calibration_path),
            'measurement_count': len(self.point_measurements),
            'measurements': [
                self._point_measurement_to_json(measurement)
                for measurement in self.point_measurements
            ],
        }
        output_path = self.session_dir / 'point_check_measurements.json'
        with output_path.open('w', encoding='utf-8') as file_obj:
            json.dump(payload, file_obj, ensure_ascii=True, indent=2)

    def _point_measurement_to_json(
        self,
        measurement: PointMeasurement,
    ) -> dict[str, Any]:
        error_norm_mm = float(np.linalg.norm(measurement.error_m) * 1000.0)
        return {
            'measurement_index': int(measurement.measurement_index),
            'lock_id': int(measurement.lock_id),
            'point_name': measurement.point_name,
            'point_kind': measurement.point_kind,
            'point_source_id': measurement.point_source_id,
            'pose_stamp_s': float(measurement.pose_stamp_s),
            'pose_frame_id': measurement.pose_frame_id,
            'p_base_measured': _float_list(measurement.p_base_measured),
            't_base_gripper_meas': transform_to_json_dict(
                measurement.t_base_gripper_meas
            ),
            'error_m': _float_list(measurement.error_m),
            'error_mm': _float_list(measurement.error_m * 1000.0),
            'error_norm_mm': error_norm_mm,
        }

    def _generate_consistency_report(self) -> None:
        if not self.consistency_samples:
            self.get_logger().warn('No consistency samples captured yet.')
            return

        transforms = [sample.t_base_target for sample in self.consistency_samples]
        t_mean = average_transforms(transforms)
        mean_translation = t_mean[:3, 3]
        residual_vectors_m = np.asarray(
            [sample.t_base_target[:3, 3] - mean_translation for sample in self.consistency_samples],
            dtype=np.float64,
        )
        residual_norms_mm = np.linalg.norm(residual_vectors_m, axis=1) * 1000.0
        rotation_errors_deg = np.asarray(
            [
                rotation_error_deg(t_mean[:3, :3], sample.t_base_target[:3, :3])
                for sample in self.consistency_samples
            ],
            dtype=np.float64,
        )

        per_sample_metrics: dict[int, dict[str, Any]] = {}
        for sample, residual_vector_m, residual_norm_mm, rot_error_deg in zip(
            self.consistency_samples,
            residual_vectors_m,
            residual_norms_mm,
            rotation_errors_deg,
        ):
            per_sample_metrics[sample.sample_index] = {
                'translation_residual_mm': _float_list(residual_vector_m * 1000.0),
                'translation_residual_norm_mm': float(residual_norm_mm),
                'rotation_error_to_mean_deg': float(rot_error_deg),
            }

        region_metrics: dict[str, list[float]] = {bucket: [] for bucket in REGION_BUCKETS}
        for sample, residual_norm_mm in zip(
            self.consistency_samples,
            residual_norms_mm,
        ):
            region_metrics[sample.region_bucket].append(float(residual_norm_mm))

        region_summary: dict[str, dict[str, float | int]] = {}
        for bucket in REGION_BUCKETS:
            values = np.asarray(region_metrics[bucket], dtype=np.float64)
            if values.size == 0:
                region_summary[bucket] = {
                    'count': 0,
                    'mean_error_norm_mm': 0.0,
                    'std_error_norm_mm': 0.0,
                    'max_error_norm_mm': 0.0,
                }
            else:
                region_summary[bucket] = {
                    'count': int(values.size),
                    'mean_error_norm_mm': float(np.mean(values)),
                    'std_error_norm_mm': float(np.std(values)),
                    'max_error_norm_mm': float(np.max(values)),
                }

        worst_entries = []
        for sample, residual_norm_mm, rot_error_deg in zip(
            self.consistency_samples,
            residual_norms_mm,
            rotation_errors_deg,
        ):
            worst_entries.append(
                {
                    'sample_index': int(sample.sample_index),
                    'region_bucket': sample.region_bucket,
                    'reproj_mean_px': float(sample.reproj_mean_px),
                    'translation_residual_norm_mm': float(residual_norm_mm),
                    'rotation_error_to_mean_deg': float(rot_error_deg),
                    'center_x_norm': float(sample.center_x_norm),
                    'center_y_norm': float(sample.center_y_norm),
                    'center_radius_norm': float(sample.center_radius_norm),
                    'gripper_rpy_deg': _float_list(sample.gripper_rpy_deg),
                    'overlay_image_path': sample.overlay_image_path,
                }
            )
        worst_entries.sort(
            key=lambda item: (
                item['translation_residual_norm_mm'],
                item['rotation_error_to_mean_deg'],
            ),
            reverse=True,
        )
        worst_entries = worst_entries[:TOP_WORST_COUNT]

        summary = {
            'created_at': datetime.now().isoformat(),
            'mode': 'consistency',
            'calibration_result_path': str(self.calibration.calibration_path),
            'image_topic': self.image_topic,
            'camera_info_topic': self.camera_info_topic,
            'oculus_topic': self.oculus_topic,
            'arm': self.arm,
            'pose_frame_id': self.calibration.pose_frame_id,
            'gripper_frame': self.calibration.gripper_frame,
            'camera_frame_id': self.calibration.camera_frame_id,
            'sample_count': len(self.consistency_samples),
            't_base_target_mean': transform_to_json_dict(t_mean),
            'translation_mean_m': _float_list(mean_translation),
            'translation_axis_std_mm': {
                axis: values['std']
                for axis, values in _stats_xyz_mm(residual_vectors_m).items()
            },
            'translation_axis_max_abs_dev_mm': {
                axis: values['max_abs']
                for axis, values in _stats_xyz_mm(residual_vectors_m).items()
            },
            'translation_residual_norm_mm': _stats_mean_std_max(residual_norms_mm),
            'rotation_mean_quaternion_xyzw': _float_list(
                rotation_matrix_to_quaternion(t_mean[:3, :3])
            ),
            'rotation_error_to_mean_deg': _stats_mean_std_max(rotation_errors_deg),
            'region_stats': region_summary,
            'worst_samples': worst_entries,
        }

        summary_path = self.session_dir / 'consistency_summary.json'
        with summary_path.open('w', encoding='utf-8') as file_obj:
            json.dump(summary, file_obj, ensure_ascii=True, indent=2)
        self._write_consistency_samples_snapshot(per_sample_metrics)
        self._write_consistency_summary_text(
            summary=summary,
            worst_entries=worst_entries,
        )
        self._save_consistency_overview_png(
            residual_norms_mm=residual_norms_mm,
        )
        self._save_consistency_heatmap_png(region_summary)
        self.get_logger().info(
            'Consistency report written to '
            f'{self.session_dir / "consistency_summary.json"}'
        )

    def _write_consistency_summary_text(
        self,
        *,
        summary: dict[str, Any],
        worst_entries: list[dict[str, Any]],
    ) -> None:
        axis_std = summary['translation_axis_std_mm']
        axis_max = summary['translation_axis_max_abs_dev_mm']
        trans_norm = summary['translation_residual_norm_mm']
        rot_norm = summary['rotation_error_to_mean_deg']
        lines = [
            f'mode: consistency',
            f'calibration_result_path: {self.calibration.calibration_path}',
            f'sample_count: {summary["sample_count"]}',
            f'pose_frame_id: {self.calibration.pose_frame_id}',
            f'gripper_frame: {self.calibration.gripper_frame}',
            f'camera_frame_id: {self.calibration.camera_frame_id}',
            'translation_mean_m: '
            + json.dumps(summary['translation_mean_m'], ensure_ascii=True),
            'translation_axis_std_mm: '
            f'x={axis_std["x"]:.4f}, y={axis_std["y"]:.4f}, z={axis_std["z"]:.4f}',
            'translation_axis_max_abs_dev_mm: '
            f'x={axis_max["x"]:.4f}, y={axis_max["y"]:.4f}, z={axis_max["z"]:.4f}',
            'translation_residual_norm_mm: '
            f'mean={trans_norm["mean"]:.4f}, std={trans_norm["std"]:.4f}, '
            f'max={trans_norm["max"]:.4f}',
            'rotation_error_to_mean_deg: '
            f'mean={rot_norm["mean"]:.4f}, std={rot_norm["std"]:.4f}, '
            f'max={rot_norm["max"]:.4f}',
            'region_stats:',
        ]
        for bucket in REGION_BUCKETS:
            bucket_stats = summary['region_stats'][bucket]
            lines.append(
                f'  {bucket}: count={bucket_stats["count"]}, '
                f'mean={bucket_stats["mean_error_norm_mm"]:.4f}mm, '
                f'std={bucket_stats["std_error_norm_mm"]:.4f}mm, '
                f'max={bucket_stats["max_error_norm_mm"]:.4f}mm'
            )
        lines.append('worst_samples:')
        for item in worst_entries:
            lines.append(
                f'  sample={item["sample_index"]}, region={item["region_bucket"]}, '
                f'translation_residual_norm_mm='
                f'{item["translation_residual_norm_mm"]:.4f}, '
                f'rotation_error_to_mean_deg='
                f'{item["rotation_error_to_mean_deg"]:.4f}, '
                f'reproj={item["reproj_mean_px"]:.4f}px, '
                f'overlay={item["overlay_image_path"]}'
            )
        output_path = self.session_dir / 'consistency_summary.txt'
        with output_path.open('w', encoding='utf-8') as file_obj:
            file_obj.write('\n'.join(lines) + '\n')

    def _save_consistency_overview_png(
        self,
        *,
        residual_norms_mm: np.ndarray,
    ) -> None:
        center_x = np.asarray(
            [sample.center_x_norm for sample in self.consistency_samples],
            dtype=np.float64,
        )
        center_y = np.asarray(
            [sample.center_y_norm for sample in self.consistency_samples],
            dtype=np.float64,
        )
        radius = np.asarray(
            [sample.center_radius_norm for sample in self.consistency_samples],
            dtype=np.float64,
        )
        rpy = np.asarray(
            [sample.gripper_rpy_deg for sample in self.consistency_samples],
            dtype=np.float64,
        )
        indices = np.arange(len(self.consistency_samples), dtype=np.float64)

        figure, axes = plt.subplots(2, 3, figsize=(18, 10))
        scatter = axes[0, 0].scatter(
            center_x,
            center_y,
            c=residual_norms_mm,
            cmap='viridis',
            s=70,
        )
        axes[0, 0].set_title('Image Center Position')
        axes[0, 0].set_xlabel('center_x_norm')
        axes[0, 0].set_ylabel('center_y_norm')
        axes[0, 0].set_xlim(0.0, 1.0)
        axes[0, 0].set_ylim(1.0, 0.0)
        figure.colorbar(scatter, ax=axes[0, 0], label='error_norm_mm')

        axes[0, 1].scatter(radius, residual_norms_mm, c='tab:blue')
        axes[0, 1].set_title('Error vs Radius')
        axes[0, 1].set_xlabel('center_radius_norm')
        axes[0, 1].set_ylabel('error_norm_mm')

        axes[0, 2].plot(indices, residual_norms_mm, 'o-', color='tab:red')
        axes[0, 2].set_title('Translation Residual by Sample')
        axes[0, 2].set_xlabel('sample_index')
        axes[0, 2].set_ylabel('error_norm_mm')

        for plot_index, axis_name in enumerate(('roll', 'pitch', 'yaw')):
            axis = axes[1, plot_index]
            axis.scatter(rpy[:, plot_index], residual_norms_mm, c='tab:green')
            axis.set_title(f'Error vs {axis_name}')
            axis.set_xlabel(f'{axis_name}_deg')
            axis.set_ylabel('error_norm_mm')

        figure.tight_layout()
        figure.savefig(
            self.session_dir / 'consistency_overview.png',
            dpi=180,
            bbox_inches='tight',
        )
        plt.close(figure)

    def _save_consistency_heatmap_png(
        self,
        region_summary: dict[str, dict[str, float | int]],
    ) -> None:
        heatmap = np.zeros((3, 3), dtype=np.float64)
        counts = np.zeros((3, 3), dtype=np.int32)
        for index, bucket in enumerate(REGION_BUCKETS):
            row_index = index // 3
            col_index = index % 3
            bucket_stats = region_summary[bucket]
            heatmap[row_index, col_index] = float(
                bucket_stats['mean_error_norm_mm']
            )
            counts[row_index, col_index] = int(bucket_stats['count'])

        figure, axis = plt.subplots(figsize=(6, 5))
        image = axis.imshow(heatmap, cmap='magma')
        axis.set_title('Mean Error Norm by 3x3 Region')
        axis.set_xticks([0, 1, 2], ['left', 'center', 'right'])
        axis.set_yticks([0, 1, 2], ['top', 'center', 'bottom'])
        for row_index in range(3):
            for col_index in range(3):
                axis.text(
                    col_index,
                    row_index,
                    f'{heatmap[row_index, col_index]:.2f}\n(n={counts[row_index, col_index]})',
                    ha='center',
                    va='center',
                    color='white',
                )
        figure.colorbar(image, ax=axis, label='mean_error_norm_mm')
        figure.tight_layout()
        figure.savefig(
            self.session_dir / 'consistency_region_heatmap.png',
            dpi=180,
            bbox_inches='tight',
        )
        plt.close(figure)

    def _generate_point_check_report(self) -> None:
        self._write_point_lock_snapshot()
        self._write_point_measurement_snapshot()

        if not self.point_measurements:
            self.get_logger().warn('No point-check measurements recorded yet.')
            return

        error_vectors_m = np.asarray(
            [measurement.error_m for measurement in self.point_measurements],
            dtype=np.float64,
        )
        error_norms_mm = np.linalg.norm(error_vectors_m, axis=1) * 1000.0
        overall_summary = {
            'count': len(self.point_measurements),
            'axis_error_mm': _stats_xyz_mm(error_vectors_m),
            'error_norm_mm': _stats_mean_std_max(error_norms_mm),
        }

        by_point: dict[str, Any] = {}
        grouped_indices_by_point: dict[str, list[int]] = defaultdict(list)
        grouped_indices_by_lock: dict[int, list[int]] = defaultdict(list)
        for index, measurement in enumerate(self.point_measurements):
            grouped_indices_by_point[measurement.point_name].append(index)
            grouped_indices_by_lock[measurement.lock_id].append(index)

        for point_name, indices in grouped_indices_by_point.items():
            vectors = error_vectors_m[indices]
            norms = error_norms_mm[indices]
            by_point[point_name] = {
                'count': len(indices),
                'axis_error_mm': _stats_xyz_mm(vectors),
                'error_norm_mm': _stats_mean_std_max(norms),
            }

        by_lock: dict[str, Any] = {}
        repeatability_by_lock: dict[str, Any] = {}
        lock_lookup = {lock.lock_id: lock for lock in self.point_locks}
        for lock_id, indices in grouped_indices_by_lock.items():
            vectors = error_vectors_m[indices]
            norms = error_norms_mm[indices]
            lock = lock_lookup[lock_id]
            by_lock[str(lock_id)] = {
                'count': len(indices),
                'point_name': lock.point_name,
                'axis_error_mm': _stats_xyz_mm(vectors),
                'error_norm_mm': _stats_mean_std_max(norms),
            }
            repeatability_by_lock[str(lock_id)] = {
                'count': len(indices),
                'point_name': lock.point_name,
                'axis_std_mm': {
                    axis: values['std']
                    for axis, values in _stats_xyz_mm(vectors).items()
                },
                'error_norm_std_mm': float(np.std(norms)),
            }

        summary = {
            'created_at': datetime.now().isoformat(),
            'mode': 'point_check',
            'calibration_result_path': str(self.calibration.calibration_path),
            'image_topic': self.image_topic,
            'camera_info_topic': self.camera_info_topic,
            'oculus_topic': self.oculus_topic,
            'arm': self.arm,
            'pose_frame_id': self.calibration.pose_frame_id,
            'gripper_frame': self.calibration.gripper_frame,
            'camera_frame_id': self.calibration.camera_frame_id,
            'lock_count': len(self.point_locks),
            'measurement_count': len(self.point_measurements),
            'overall': overall_summary,
            'by_point_name': by_point,
            'by_lock_id': by_lock,
            'repeatability_by_lock': repeatability_by_lock,
        }

        output_path = self.session_dir / 'point_check_summary.json'
        with output_path.open('w', encoding='utf-8') as file_obj:
            json.dump(summary, file_obj, ensure_ascii=True, indent=2)

        self._write_point_check_summary_text(summary)
        self._save_point_check_overview_png(
            error_vectors_m=error_vectors_m,
            error_norms_mm=error_norms_mm,
        )
        self.get_logger().info(
            'Point-check report written to '
            f'{self.session_dir / "point_check_summary.json"}'
        )

    def _write_point_check_summary_text(self, summary: dict[str, Any]) -> None:
        overall = summary['overall']
        lines = [
            'mode: point_check',
            f'calibration_result_path: {self.calibration.calibration_path}',
            f'lock_count: {summary["lock_count"]}',
            f'measurement_count: {summary["measurement_count"]}',
            'overall axis_error_mm:',
        ]
        for axis, axis_stats in overall['axis_error_mm'].items():
            lines.append(
                f'  {axis}: mean={axis_stats["mean"]:.4f}, '
                f'std={axis_stats["std"]:.4f}, max_abs={axis_stats["max_abs"]:.4f}'
            )
        lines.append(
            'overall error_norm_mm: '
            f'mean={overall["error_norm_mm"]["mean"]:.4f}, '
            f'std={overall["error_norm_mm"]["std"]:.4f}, '
            f'max={overall["error_norm_mm"]["max"]:.4f}'
        )
        lines.append('by_point_name:')
        for point_name, point_stats in summary['by_point_name'].items():
            norm = point_stats['error_norm_mm']
            lines.append(
                f'  {point_name}: count={point_stats["count"]}, '
                f'mean_norm={norm["mean"]:.4f}, std_norm={norm["std"]:.4f}, '
                f'max_norm={norm["max"]:.4f}'
            )
        lines.append('repeatability_by_lock:')
        for lock_id, lock_stats in summary['repeatability_by_lock'].items():
            axis_std = lock_stats['axis_std_mm']
            lines.append(
                f'  lock={lock_id} ({lock_stats["point_name"]}): '
                f'count={lock_stats["count"]}, '
                f'std_mm=[{axis_std["x"]:.4f}, {axis_std["y"]:.4f}, {axis_std["z"]:.4f}], '
                f'norm_std={lock_stats["error_norm_std_mm"]:.4f}'
            )

        output_path = self.session_dir / 'point_check_summary.txt'
        with output_path.open('w', encoding='utf-8') as file_obj:
            file_obj.write('\n'.join(lines) + '\n')

    def _save_point_check_overview_png(
        self,
        *,
        error_vectors_m: np.ndarray,
        error_norms_mm: np.ndarray,
    ) -> None:
        error_vectors_mm = error_vectors_m * 1000.0
        indices = np.arange(len(self.point_measurements), dtype=np.float64)
        point_names = [measurement.point_name for measurement in self.point_measurements]
        unique_points = list(dict.fromkeys(point_names))
        colors = plt.cm.tab10(np.linspace(0.0, 1.0, max(1, len(unique_points))))
        point_to_color = {
            point_name: colors[index]
            for index, point_name in enumerate(unique_points)
        }

        figure, axes = plt.subplots(2, 3, figsize=(18, 10))
        for axis_index, axis_name in enumerate(('x', 'y', 'z')):
            axis = axes[0, axis_index]
            for point_name in unique_points:
                point_indices = [
                    index
                    for index, measurement in enumerate(self.point_measurements)
                    if measurement.point_name == point_name
                ]
                axis.scatter(
                    indices[point_indices],
                    error_vectors_mm[point_indices, axis_index],
                    label=point_name,
                    color=point_to_color[point_name],
                )
            axis.axhline(0.0, color='black', linewidth=1.0)
            axis.set_title(f'{axis_name.upper()} Error')
            axis.set_xlabel('measurement_index')
            axis.set_ylabel('error_mm')

        axes[1, 0].axvline(0.0, color='black', linewidth=1.0)
        axes[1, 0].axhline(0.0, color='black', linewidth=1.0)
        for point_name in unique_points:
            point_indices = [
                index
                for index, measurement in enumerate(self.point_measurements)
                if measurement.point_name == point_name
            ]
            axes[1, 0].scatter(
                error_vectors_mm[point_indices, 0],
                error_vectors_mm[point_indices, 1],
                label=point_name,
                color=point_to_color[point_name],
            )
        axes[1, 0].set_title('XY Error Scatter')
        axes[1, 0].set_xlabel('x_error_mm')
        axes[1, 0].set_ylabel('y_error_mm')

        axes[1, 1].plot(indices, error_norms_mm, 'o-', color='tab:red')
        axes[1, 1].set_title('Error Norm by Measurement')
        axes[1, 1].set_xlabel('measurement_index')
        axes[1, 1].set_ylabel('error_norm_mm')

        mean_norms = [
            np.mean(
                [
                    error_norms_mm[index]
                    for index, measurement in enumerate(self.point_measurements)
                    if measurement.point_name == point_name
                ]
            )
            for point_name in unique_points
        ]
        axes[1, 2].bar(unique_points, mean_norms, color=colors[: len(unique_points)])
        axes[1, 2].set_title('Mean Error Norm by Point')
        axes[1, 2].set_ylabel('mean_error_norm_mm')
        axes[1, 2].tick_params(axis='x', rotation=30)

        handles, labels = axes[0, 0].get_legend_handles_labels()
        if handles:
            figure.legend(handles, labels, loc='upper right')
        figure.tight_layout()
        figure.savefig(
            self.session_dir / 'point_check_overview.png',
            dpi=180,
            bbox_inches='tight',
        )
        plt.close(figure)

    def _active_lock(self) -> PointLock | None:
        if self.active_lock_id is None:
            return None
        for lock in self.point_locks:
            if lock.lock_id == self.active_lock_id:
                return lock
        return None

    def _shutdown_with_error(self, message: str) -> None:
        if self._fatal_error_message:
            return
        self._fatal_error_message = message
        self.get_logger().error(message)
        if rclpy.ok():
            rclpy.shutdown()

    def _log_help_once(self, force: bool = False) -> None:
        if self._printed_help and not force:
            return
        self._printed_help = True
        if self.mode == 'consistency':
            self.get_logger().info(
                'Controls: '
                '[s]/[space]=capture sample, '
                '[c]=generate report, '
                '[r]=clear samples, '
                '[q]=quit.'
            )
        else:
            self.get_logger().info(
                'Controls: '
                '[n]/[p]=switch point, '
                '[l]=lock current point, '
                '[m]=record manual measurement, '
                '[c]=generate report, '
                '[r]=clear session, '
                '[q]=quit.'
            )
        if not self.show_preview:
            self.get_logger().info(
                'show_preview=false: node will keep running, but keyboard controls '
                'will not be available.'
            )


def main(args: list[str] | None = None) -> int:
    rclpy.init(args=args)
    node: EyeInHandValidatorNode | None = None
    try:
        node = EyeInHandValidatorNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
