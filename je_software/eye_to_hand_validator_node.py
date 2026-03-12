#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import time
from collections import deque
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
    invert_transform,
    load_eye_to_hand_calibration_result,
    make_transform,
    opencv_supports_charuco,
    pose_to_transform,
    region_bucket_name,
    rotation_error_deg,
    rotation_matrix_to_rpy_deg,
    stamp_to_seconds,
    transform_point,
    transform_to_json_dict,
)

matplotlib.use('Agg')
from matplotlib import pyplot as plt  # noqa: E402


TOP_WORST_COUNT = 5
SUPPORTED_MODES = ('consistency', 'point_check')


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
    t_gripper_target: np.ndarray
    raw_image_path: str
    overlay_image_path: str


@dataclass
class PointTarget:
    target_index: int
    point_name: str
    point_kind: str
    point_source_id: int | None
    point_t: np.ndarray
    image_stamp_s: float
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
    t_cam_target: np.ndarray
    t_base_target: np.ndarray
    t_base_point: np.ndarray
    raw_image_path: str
    overlay_image_path: str


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


class EyeToHandValidatorNode(Node):
    def __init__(self) -> None:
        super().__init__('eye_to_hand_validator')

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
        self.endpose_sub_topic = str(
            self.get_parameter('endpose_sub_topic').value
        ).strip()
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
        self.point_z_offset_m = float(
            self.get_parameter('point_z_offset_m').value
        )
        self.charuco_ids_csv = str(
            self.get_parameter('charuco_ids_csv').value
        ).strip()

        self._validate_parameters()
        self.calibration = load_eye_to_hand_calibration_result(
            self.calibration_result_path
        )
        self.charuco_helper = CharucoBoardHelper.from_board_config(
            self.calibration.board_config
        )
        self._extra_charuco_ids = self._parse_charuco_ids_csv()
        self.board_points: list[BoardPoint] = self.charuco_helper.default_point_set(
            point_set=self.point_set,
            extra_charuco_ids=self._extra_charuco_ids,
            point_z_offset_m=self.point_z_offset_m,
        )

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        session_name = f'eye_to_hand_validation_{self.mode}_{timestamp}'
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
        self._printed_help = False
        self._fatal_error_message = ''

        self.consistency_samples: list[ConsistencySample] = []
        self.point_targets: list[PointTarget] = []
        self.selected_point_index = 0

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
        self.endpose_sub = None
        if self.mode == 'consistency':
            self.endpose_sub = self.create_subscription(
                OculusControllers,
                self.endpose_sub_topic,
                self._endpose_callback,
                qos_profile_sensor_data,
            )
        self.status_timer = self.create_timer(2.0, self._status_timer_callback)

        self.get_logger().info(
            'Eye-to-hand validator started: '
            f'mode={self.mode}, '
            f'image_topic={self.image_topic}, '
            f'camera_info_topic={self.camera_info_topic}, '
            f'endpose_sub_topic={self.endpose_sub_topic}, '
            f'arm={self.arm}, '
            f'calibration_result={self.calibration.calibration_path}, '
            f'output_dir={self.session_dir}'
        )
        self.get_logger().info(
            'Loaded calibration frames: '
            f'pose_frame={self.calibration.pose_frame_id}, '
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
        self.declare_parameter('endpose_sub_topic', '/endpose_states_double_arm')
        self.declare_parameter('arm', 'left')
        self.declare_parameter(
            'output_dir',
            str(Path.home() / 'eye_to_hand_validation'),
        )
        self.declare_parameter('show_preview', True)
        self.declare_parameter('preview_max_width', 1280)
        self.declare_parameter('image_is_rectified', True)
        self.declare_parameter('pose_sync_tolerance_s', 0.08)
        self.declare_parameter('process_interval_s', 0.15)
        self.declare_parameter('max_reproj_error_px', 3.0)
        self.declare_parameter('min_charuco_corners', 8)
        self.declare_parameter('point_set', 'center_corners')
        self.declare_parameter('point_z_offset_m', 0.0)
        self.declare_parameter('charuco_ids_csv', '')

    def _validate_parameters(self) -> None:
        if self.mode not in SUPPORTED_MODES:
            supported = ', '.join(SUPPORTED_MODES)
            raise ValueError(f'Unsupported mode. Supported values: {supported}.')
        if not self.calibration_result_path:
            raise ValueError('calibration_result_path must not be empty.')
        if self.mode == 'consistency' and not self.endpose_sub_topic:
            raise ValueError('endpose_sub_topic must not be empty.')
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
        if not np.isfinite(self.point_z_offset_m):
            raise ValueError('point_z_offset_m must be finite.')

    def _parse_charuco_ids_csv(self) -> list[int]:
        if not self.charuco_ids_csv:
            return []
        values = []
        max_id = int(
            self.charuco_helper.charuco_corner_object_points.shape[0]
        ) - 1
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

    def _endpose_callback(self, msg: OculusControllers) -> None:
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
            cv2.imshow('Eye-To-Hand Validation', display_image)
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

    def _status_timer_callback(self) -> None:
        if self._fatal_error_message:
            return
        details = [
            f'camera_info={"yes" if self.camera_matrix is not None else "no"}',
            f'image_msgs={self.total_image_msgs}',
        ]
        if self.mode == 'consistency':
            details.append(f'pose_buffer={len(self.pose_buffer)}')
            details.append(f'samples={len(self.consistency_samples)}')
        else:
            details.append(f'targets={len(self.point_targets)}')
            details.append(
                f'selected={self.board_points[self.selected_point_index].name}'
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
            f'camera_info: {"ready" if self.camera_matrix is not None else "waiting"}',
            f'markers: {marker_count}  charuco: {observation.detection.charuco_count}',
            f'pose_frame: {self.pose_frame_id_live or "waiting"}',
            f'camera_frame: {self.camera_frame_id_live or "waiting"}',
        ]
        if self.mode == 'consistency':
            lines.insert(
                1,
                f'pose_buffer: {len(self.pose_buffer)}  samples: {len(self.consistency_samples)}',
            )
        else:
            lines.insert(
                1,
                f'targets: {len(self.point_targets)}',
            )
            lines.append(
                f'selected_point: {self.board_points[self.selected_point_index].name}'
            )
        if observation.reproj_mean_px is not None:
            lines.append(f'reproj: {observation.reproj_mean_px:.3f}px')
        else:
            lines.append('reproj: n/a')
        if observation.pose_offset_s is not None:
            lines.append(f'pose_dt: {observation.pose_offset_s * 1000.0:.1f} ms')
        else:
            lines.append('pose_dt: n/a')
        if self.mode == 'consistency':
            lines.append('keys: s/space=sample  c=report  r=reset  q=quit')
        else:
            lines.append('keys: n/p=point  s/space/l=save  c=report  r=reset  q=quit')

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
            if self.mode == 'consistency':
                self.consistency_samples.clear()
                self._write_consistency_samples_snapshot()
                self.get_logger().info('Cleared consistency samples.')
            else:
                self.point_targets.clear()
                self._write_point_target_snapshot()
                self.get_logger().info('Cleared point-check targets.')
            return
        if self.mode == 'consistency':
            if key in (ord('s'), ord(' ')):
                self._capture_consistency_sample(observation, preview_image)
                return
            if key == ord('c'):
                self._generate_consistency_report()
            return

        if key == ord('n'):
            self.selected_point_index = (
                self.selected_point_index + 1
            ) % len(self.board_points)
            self.get_logger().info(
                f'Selected point: {self.board_points[self.selected_point_index].name}'
            )
            return
        if key == ord('p'):
            self.selected_point_index = (
                self.selected_point_index - 1
            ) % len(self.board_points)
            self.get_logger().info(
                f'Selected point: {self.board_points[self.selected_point_index].name}'
            )
            return
        if key in (ord('l'), ord('s'), ord(' ')):
            self._save_point_target(observation, preview_image)
            return
        if key == ord('c'):
            self._generate_point_check_report()

    def _capture_consistency_sample(
        self,
        observation: LiveObservation,
        preview_image: np.ndarray,
    ) -> None:
        rejection = self._validate_visual_sample(
            observation,
            require_pose=True,
        )
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
        t_gripper_target = (
            invert_transform(observation.pose_snapshot.transform)
            @ self.calibration.t_base_cam
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
        raw_image_path = (
            self.images_dir / f'consistency_sample_{sample_index:03d}_raw.png'
        )
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
            t_gripper_target=t_gripper_target,
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

    def _validate_visual_sample(
        self,
        observation: LiveObservation,
        *,
        require_pose: bool,
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
        if require_pose and observation.pose_snapshot is None:
            return 'no synchronized end-effector pose'
        if (
            require_pose
            and observation.pose_snapshot is not None
            and observation.pose_snapshot.frame_id != self.calibration.pose_frame_id
        ):
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
            'endpose_sub_topic': self.endpose_sub_topic,
            'arm': self.arm,
            'pose_frame_id': self.calibration.pose_frame_id,
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
            't_gripper_target': transform_to_json_dict(sample.t_gripper_target),
            'raw_image_path': sample.raw_image_path,
            'overlay_image_path': sample.overlay_image_path,
        }
        if per_sample_metrics and sample.sample_index in per_sample_metrics:
            payload['report_metrics'] = per_sample_metrics[sample.sample_index]
        return payload

    def _save_point_target(
        self,
        observation: LiveObservation,
        preview_image: np.ndarray,
    ) -> None:
        rejection = self._validate_visual_sample(
            observation,
            require_pose=False,
        )
        if rejection is not None:
            self.get_logger().warn(f'Measurement rejected: {rejection}')
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
        t_base_target = self.calibration.t_base_cam @ t_cam_target
        p_base_point = transform_point(
            t_base_target,
            selected.point_t,
        )
        t_base_point = make_transform(
            t_base_target[:3, :3],
            p_base_point,
        )
        region_bucket = region_bucket_name(
            observation.center_x_norm,
            observation.center_y_norm,
        )

        target_index = len(self.point_targets)
        raw_image_path = (
            self.images_dir / f'point_target_{target_index:03d}_raw.png'
        )
        overlay_image_path = (
            self.images_dir / f'point_target_{target_index:03d}_overlay.png'
        )
        cv2.imwrite(str(raw_image_path), observation.bgr_image)
        cv2.imwrite(str(overlay_image_path), preview_image)

        target = PointTarget(
            target_index=target_index,
            point_name=selected.name,
            point_kind=selected.kind,
            point_source_id=selected.source_id,
            point_t=selected.point_t.copy(),
            image_stamp_s=observation.image_stamp_s,
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
            t_cam_target=t_cam_target,
            t_base_target=t_base_target,
            t_base_point=t_base_point,
            raw_image_path=str(raw_image_path),
            overlay_image_path=str(overlay_image_path),
        )
        self.point_targets.append(target)
        self._write_point_target_snapshot()
        self.get_logger().info(
            'Saved point target '
            f'#{target.target_index} ({target.point_name}): '
            f't=[{target.t_base_point[0, 3]:.6f}, '
            f'{target.t_base_point[1, 3]:.6f}, {target.t_base_point[2, 3]:.6f}] m'
        )

    def _write_point_target_snapshot(self) -> None:
        payload = {
            'created_at': datetime.now().isoformat(),
            'mode': 'point_check',
            'calibration_result_path': str(self.calibration.calibration_path),
            'image_topic': self.image_topic,
            'camera_info_topic': self.camera_info_topic,
            'arm': self.arm,
            'pose_frame_id': self.calibration.pose_frame_id,
            'camera_frame_id': self.calibration.camera_frame_id,
            'selected_point_name': self.board_points[self.selected_point_index].name,
            'target_count': len(self.point_targets),
            'targets': [
                self._point_target_to_json(target)
                for target in self.point_targets
            ],
        }
        output_path = self.session_dir / 'point_check_targets.json'
        with output_path.open('w', encoding='utf-8') as file_obj:
            json.dump(payload, file_obj, ensure_ascii=True, indent=2)

    def _point_target_to_json(
        self,
        target: PointTarget,
    ) -> dict[str, Any]:
        return {
            'target_index': int(target.target_index),
            'point_name': target.point_name,
            'point_kind': target.point_kind,
            'point_source_id': target.point_source_id,
            'point_t': _float_list(target.point_t),
            'image_stamp_s': float(target.image_stamp_s),
            'image_width': int(target.image_width),
            'image_height': int(target.image_height),
            'charuco_count': int(target.charuco_count),
            'reproj_mean_px': float(target.reproj_mean_px),
            'board_center_x_px': float(target.board_center_x_px),
            'board_center_y_px': float(target.board_center_y_px),
            'center_x_norm': float(target.center_x_norm),
            'center_y_norm': float(target.center_y_norm),
            'center_radius_norm': float(target.center_radius_norm),
            'region_bucket': target.region_bucket,
            't_cam_target': transform_to_json_dict(target.t_cam_target),
            't_base_target': transform_to_json_dict(target.t_base_target),
            't_base_point': transform_to_json_dict(target.t_base_point),
            'raw_image_path': target.raw_image_path,
            'overlay_image_path': target.overlay_image_path,
        }

    def _generate_consistency_report(self) -> None:
        if not self.consistency_samples:
            self.get_logger().warn('No consistency samples captured yet.')
            return

        transforms = [
            sample.t_gripper_target for sample in self.consistency_samples
        ]
        t_mean = average_transforms(transforms)
        mean_translation = t_mean[:3, 3]
        calibration_translation = self.calibration.t_gripper_target[:3, 3]
        residual_vectors_m = np.asarray(
            [
                sample.t_gripper_target[:3, 3] - mean_translation
                for sample in self.consistency_samples
            ],
            dtype=np.float64,
        )
        residual_norms_mm = np.linalg.norm(residual_vectors_m, axis=1) * 1000.0
        rotation_errors_deg = np.asarray(
            [
                rotation_error_deg(
                    t_mean[:3, :3],
                    sample.t_gripper_target[:3, :3],
                )
                for sample in self.consistency_samples
            ],
            dtype=np.float64,
        )
        error_to_calibration_mm = np.asarray(
            [
                np.linalg.norm(
                    sample.t_gripper_target[:3, 3] - calibration_translation
                )
                * 1000.0
                for sample in self.consistency_samples
            ],
            dtype=np.float64,
        )
        rotation_to_calibration_deg = np.asarray(
            [
                rotation_error_deg(
                    self.calibration.t_gripper_target[:3, :3],
                    sample.t_gripper_target[:3, :3],
                )
                for sample in self.consistency_samples
            ],
            dtype=np.float64,
        )
        mean_to_calibration_vector_m = mean_translation - calibration_translation
        mean_to_calibration_norm_mm = float(
            np.linalg.norm(mean_to_calibration_vector_m) * 1000.0
        )
        mean_to_calibration_rotation_deg = float(
            rotation_error_deg(
                self.calibration.t_gripper_target[:3, :3],
                t_mean[:3, :3],
            )
        )

        per_sample_metrics: dict[int, dict[str, Any]] = {}
        for (
            sample,
            residual_vector_m,
            residual_norm_mm,
            rot_error_deg,
            translation_to_calibration_mm,
            rotation_to_calibration_deg,
        ) in zip(
            self.consistency_samples,
            residual_vectors_m,
            residual_norms_mm,
            rotation_errors_deg,
            error_to_calibration_mm,
            rotation_to_calibration_deg,
        ):
            per_sample_metrics[sample.sample_index] = {
                'translation_residual_mm': _float_list(residual_vector_m * 1000.0),
                'translation_residual_norm_mm': float(residual_norm_mm),
                'rotation_error_to_mean_deg': float(rot_error_deg),
                'translation_error_to_calibration_mm': float(
                    translation_to_calibration_mm
                ),
                'rotation_error_to_calibration_deg': float(
                    rotation_to_calibration_deg
                ),
            }

        region_metrics: dict[str, list[float]] = {
            bucket: [] for bucket in REGION_BUCKETS
        }
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
            'endpose_sub_topic': self.endpose_sub_topic,
            'arm': self.arm,
            'pose_frame_id': self.calibration.pose_frame_id,
            'camera_frame_id': self.calibration.camera_frame_id,
            'sample_count': len(self.consistency_samples),
            'calibration_t_gripper_target': transform_to_json_dict(
                self.calibration.t_gripper_target
            ),
            't_gripper_target_mean': transform_to_json_dict(t_mean),
            'translation_mean_m': _float_list(mean_translation),
            'translation_mean_to_calibration_mm': _float_list(
                mean_to_calibration_vector_m * 1000.0
            ),
            'translation_mean_to_calibration_norm_mm': (
                mean_to_calibration_norm_mm
            ),
            'rotation_mean_to_calibration_deg': mean_to_calibration_rotation_deg,
            'translation_axis_std_mm': {
                axis: values['std']
                for axis, values in _stats_xyz_mm(residual_vectors_m).items()
            },
            'translation_axis_max_abs_dev_mm': {
                axis: values['max_abs']
                for axis, values in _stats_xyz_mm(residual_vectors_m).items()
            },
            'translation_residual_norm_mm': _stats_mean_std_max(residual_norms_mm),
            'rotation_error_to_mean_deg': _stats_mean_std_max(rotation_errors_deg),
            'translation_error_to_calibration_mm': _stats_mean_std_max(
                error_to_calibration_mm
            ),
            'rotation_error_to_calibration_deg': _stats_mean_std_max(
                rotation_to_calibration_deg
            ),
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
        trans_to_cal = summary['translation_error_to_calibration_mm']
        rot_to_cal = summary['rotation_error_to_calibration_deg']
        lines = [
            'mode: consistency',
            f'calibration_result_path: {self.calibration.calibration_path}',
            f'sample_count: {summary["sample_count"]}',
            f'pose_frame_id: {self.calibration.pose_frame_id}',
            f'camera_frame_id: {self.calibration.camera_frame_id}',
            'translation_mean_m: '
            + json.dumps(summary['translation_mean_m'], ensure_ascii=True),
            'translation_mean_to_calibration_mm: '
            + json.dumps(
                summary['translation_mean_to_calibration_mm'],
                ensure_ascii=True,
            ),
            'translation_mean_to_calibration_norm_mm: '
            f'{summary["translation_mean_to_calibration_norm_mm"]:.4f}',
            'rotation_mean_to_calibration_deg: '
            f'{summary["rotation_mean_to_calibration_deg"]:.4f}',
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
            'translation_error_to_calibration_mm: '
            f'mean={trans_to_cal["mean"]:.4f}, std={trans_to_cal["std"]:.4f}, '
            f'max={trans_to_cal["max"]:.4f}',
            'rotation_error_to_calibration_deg: '
            f'mean={rot_to_cal["mean"]:.4f}, std={rot_to_cal["std"]:.4f}, '
            f'max={rot_to_cal["max"]:.4f}',
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
                    f'{heatmap[row_index, col_index]:.2f}\n'
                    f'(n={counts[row_index, col_index]})',
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
        self._write_point_target_snapshot()

        if not self.point_targets:
            self.get_logger().warn('No point-check targets saved yet.')
            return

        base_positions_m = np.asarray(
            [target.t_base_point[:3, 3] for target in self.point_targets],
            dtype=np.float64,
        )
        by_point: dict[str, Any] = {}
        for target in self.point_targets:
            entry = by_point.setdefault(
                target.point_name,
                {
                    'count': 0,
                    'target_indices': [],
                    'positions_m': [],
                },
            )
            entry['count'] += 1
            entry['target_indices'].append(int(target.target_index))
            entry['positions_m'].append(
                [float(value) for value in target.t_base_point[:3, 3]]
            )

        position_min_m = _float_list(np.min(base_positions_m, axis=0))
        position_max_m = _float_list(np.max(base_positions_m, axis=0))
        position_span_m = _float_list(
            np.max(base_positions_m, axis=0) - np.min(base_positions_m, axis=0)
        )

        summary_targets = []
        for target in self.point_targets:
            summary_targets.append(
                {
                    'target_index': int(target.target_index),
                    'point_name': target.point_name,
                    'point_kind': target.point_kind,
                    'point_source_id': target.point_source_id,
                    't_base_point': transform_to_json_dict(target.t_base_point),
                    'region_bucket': target.region_bucket,
                    'reproj_mean_px': float(target.reproj_mean_px),
                    'overlay_image_path': target.overlay_image_path,
                }
            )

        summary = {
            'created_at': datetime.now().isoformat(),
            'mode': 'point_check',
            'calibration_result_path': str(self.calibration.calibration_path),
            'image_topic': self.image_topic,
            'camera_info_topic': self.camera_info_topic,
            'arm': self.arm,
            'pose_frame_id': self.calibration.pose_frame_id,
            'camera_frame_id': self.calibration.camera_frame_id,
            'target_count': len(self.point_targets),
            'position_min_m': position_min_m,
            'position_max_m': position_max_m,
            'position_span_m': position_span_m,
            'by_point_name': by_point,
            'targets': summary_targets,
        }

        output_path = self.session_dir / 'point_check_summary.json'
        with output_path.open('w', encoding='utf-8') as file_obj:
            json.dump(summary, file_obj, ensure_ascii=True, indent=2)

        self._write_point_check_summary_text(summary)
        self._save_point_check_overview_png(base_positions_m=base_positions_m)
        self.get_logger().info(
            'Point-check report written to '
            f'{self.session_dir / "point_check_summary.json"}'
        )

    def _write_point_check_summary_text(self, summary: dict[str, Any]) -> None:
        lines = [
            'mode: point_check',
            f'calibration_result_path: {self.calibration.calibration_path}',
            f'target_count: {summary["target_count"]}',
            'position_min_m: ' + json.dumps(summary['position_min_m'], ensure_ascii=True),
            'position_max_m: ' + json.dumps(summary['position_max_m'], ensure_ascii=True),
            'position_span_m: ' + json.dumps(summary['position_span_m'], ensure_ascii=True),
            'targets:',
        ]
        for target in summary['targets']:
            pose = target['t_base_point']
            translation = pose['translation_m']
            quaternion = pose['quaternion_xyzw']
            lines.append(
                f'  target={target["target_index"]}, point={target["point_name"]}, '
                f't=[{translation[0]:.6f}, {translation[1]:.6f}, {translation[2]:.6f}], '
                f'q=[{quaternion[0]:.6f}, {quaternion[1]:.6f}, {quaternion[2]:.6f}, '
                f'{quaternion[3]:.6f}], reproj={target["reproj_mean_px"]:.4f}px, '
                f'region={target["region_bucket"]}'
            )

        output_path = self.session_dir / 'point_check_summary.txt'
        with output_path.open('w', encoding='utf-8') as file_obj:
            file_obj.write('\n'.join(lines) + '\n')

    def _save_point_check_overview_png(
        self,
        *,
        base_positions_m: np.ndarray,
    ) -> None:
        positions = np.asarray(base_positions_m, dtype=np.float64).reshape(-1, 3)
        indices = np.arange(len(self.point_targets), dtype=np.float64)
        point_names = [target.point_name for target in self.point_targets]
        unique_points = list(dict.fromkeys(point_names))
        colors = plt.cm.tab10(np.linspace(0.0, 1.0, max(1, len(unique_points))))
        point_to_color = {
            point_name: colors[index]
            for index, point_name in enumerate(unique_points)
        }

        figure, axes = plt.subplots(2, 3, figsize=(18, 10))
        for point_name in unique_points:
            point_indices = [
                index
                for index, target in enumerate(self.point_targets)
                if target.point_name == point_name
            ]
            color = point_to_color[point_name]
            axes[0, 0].scatter(
                positions[point_indices, 0],
                positions[point_indices, 1],
                label=point_name,
                color=color,
            )
            axes[0, 1].scatter(
                positions[point_indices, 0],
                positions[point_indices, 2],
                label=point_name,
                color=color,
            )
            axes[0, 2].scatter(
                positions[point_indices, 1],
                positions[point_indices, 2],
                label=point_name,
                color=color,
            )

        axes[0, 0].set_title('Base XY Targets')
        axes[0, 0].set_xlabel('x_m')
        axes[0, 0].set_ylabel('y_m')
        axes[0, 1].set_title('Base XZ Targets')
        axes[0, 1].set_xlabel('x_m')
        axes[0, 1].set_ylabel('z_m')
        axes[0, 2].set_title('Base YZ Targets')
        axes[0, 2].set_xlabel('y_m')
        axes[0, 2].set_ylabel('z_m')

        axes[1, 0].plot(indices, positions[:, 0], 'o-', color='tab:red')
        axes[1, 0].set_title('Target X by Index')
        axes[1, 0].set_xlabel('target_index')
        axes[1, 0].set_ylabel('x_m')

        axes[1, 1].plot(indices, positions[:, 1], 'o-', color='tab:green')
        axes[1, 1].set_title('Target Y by Index')
        axes[1, 1].set_xlabel('target_index')
        axes[1, 1].set_ylabel('y_m')

        point_counts = [
            len([target for target in self.point_targets if target.point_name == point_name])
            for point_name in unique_points
        ]
        axes[1, 2].bar(unique_points, point_counts, color=colors[: len(unique_points)])
        axes[1, 2].set_title('Saved Targets by Point')
        axes[1, 2].set_ylabel('count')
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
                '[l]/[s]/[space]=save target pose, '
                '[c]=generate report, '
                '[r]=clear saved targets, '
                '[q]=quit.'
            )
        if not self.show_preview:
            self.get_logger().info(
                'show_preview=false: node will keep running, but keyboard controls '
                'will not be available.'
            )


def main(args: list[str] | None = None) -> int:
    rclpy.init(args=args)
    node: EyeToHandValidatorNode | None = None
    try:
        node = EyeToHandValidatorNode()
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
