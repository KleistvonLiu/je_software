#!/usr/bin/env python3
from __future__ import annotations

import json
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import rclpy
from common.msg import OculusControllers
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image

from je_software.calibration_common import (
    METHOD_NAME_TO_FLAG,
    CharucoBoardHelper,
    CharucoDetection,
    PoseSnapshot,
    average_transforms,
    draw_detection_overlay,
    image_msg_to_bgr,
    invert_transform,
    load_charuco_config,
    make_transform,
    opencv_supports_charuco,
    orthonormalize_rotation,
    pose_to_transform,
    rotation_error_deg,
    rotation_matrix_to_quaternion,
    stamp_to_seconds,
    transform_to_json_dict,
)


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


@dataclass
class CapturedSample:
    sample_index: int
    image_stamp_s: float
    pose_stamp_s: float
    pose_offset_s: float
    pose_frame_id: str
    image_width: int
    image_height: int
    charuco_count: int
    reproj_mean_px: float
    t_base_gripper: np.ndarray
    t_cam_target: np.ndarray
    raw_image_path: str
    preview_image_path: str


class EyeInHandCalibrationNode(Node):
    def __init__(self) -> None:
        super().__init__('eye_in_hand_calibration')

        self._declare_parameters()

        if not opencv_supports_charuco():
            raise RuntimeError(
                'Current OpenCV build does not expose cv2.aruco Charuco APIs.'
            )

        self.image_topic = str(self.get_parameter('image_topic').value)
        self.camera_info_topic = str(self.get_parameter('camera_info_topic').value)
        self.endpose_sub_topic = str(
            self.get_parameter('endpose_sub_topic').value
        ).strip()
        self.arm = str(self.get_parameter('arm').value).strip().lower()
        self.gripper_frame = str(
            self.get_parameter('gripper_frame').value
        ).strip()
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
        self.min_pose_delta_translation_m = float(
            self.get_parameter('min_pose_delta_translation_m').value
        )
        self.min_pose_delta_rotation_deg = float(
            self.get_parameter('min_pose_delta_rotation_deg').value
        )
        self.min_samples = int(self.get_parameter('min_samples').value)
        self.max_reproj_error_px = float(
            self.get_parameter('max_reproj_error_px').value
        )
        self.image_is_rectified = bool(
            self.get_parameter('image_is_rectified').value
        )
        self.hand_eye_method = str(
            self.get_parameter('hand_eye_method').value
        ).strip().lower()

        self.charuco_config_path = str(
            self.get_parameter('charuco_config_path').value
        ).strip()
        self.squares_x = int(self.get_parameter('squares_x').value)
        self.squares_y = int(self.get_parameter('squares_y').value)
        self.square_length_m = float(
            self.get_parameter('square_length_m').value
        )
        self.marker_length_m = float(
            self.get_parameter('marker_length_m').value
        )
        self.aruco_dictionary_name = str(
            self.get_parameter('aruco_dictionary').value
        ).strip()
        self.min_charuco_corners = int(
            self.get_parameter('min_charuco_corners').value
        )

        self._load_charuco_config_if_needed()
        self._validate_parameters()

        self.charuco_helper = CharucoBoardHelper(
            squares_x=self.squares_x,
            squares_y=self.squares_y,
            square_length_m=self.square_length_m,
            marker_length_m=self.marker_length_m,
            aruco_dictionary_name=self.aruco_dictionary_name,
            charuco_config_path=self.charuco_config_path,
        )

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.session_dir = self.output_root / f'eye_in_hand_calibration_{timestamp}'
        self.images_dir = self.session_dir / 'images'
        self.images_dir.mkdir(parents=True, exist_ok=True)

        self.camera_matrix: np.ndarray | None = None
        self.dist_coeffs: np.ndarray | None = None
        self.camera_info_snapshot: dict[str, Any] = {}
        self.camera_frame_id = ''
        self.pose_frame_id = ''
        self.pose_buffer: deque[PoseSnapshot] = deque(maxlen=400)
        self.samples: list[CapturedSample] = []

        self.last_process_time_s = 0.0
        self.total_image_msgs = 0
        self.last_valid_detection: LiveObservation | None = None
        self._printed_help = False

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
        self.endpose_sub_ = self.create_subscription(
            OculusControllers,
            self.endpose_sub_topic,
            self._endpose_callback,
            qos_profile_sensor_data,
        )
        self.status_timer = self.create_timer(2.0, self._status_timer_callback)

        self.get_logger().info(
            'Eye-in-hand calibrator started: '
            f'image_topic={self.image_topic}, '
            f'camera_info_topic={self.camera_info_topic}, '
            f'endpose_sub_topic={self.endpose_sub_topic}, '
            f'arm={self.arm}, '
            f'gripper_frame={self.gripper_frame}, '
            f'rectified={self.image_is_rectified}, '
            f'output_dir={self.session_dir}'
        )
        self.get_logger().info(
            'Board config: '
            f'ChArUco {self.squares_x}x{self.squares_y}, '
            f'square={self.square_length_m:.4f}m, '
            f'marker={self.marker_length_m:.4f}m, '
            f'dictionary={self.aruco_dictionary_name}'
        )
        self._log_help_once()

    def destroy_node(self) -> bool:
        if self.show_preview:
            cv2.destroyAllWindows()
        return super().destroy_node()

    def _declare_parameters(self) -> None:
        self.declare_parameter('image_topic', '/camera/color/image_raw')
        self.declare_parameter(
            'camera_info_topic',
            '/camera/color/camera_info',
        )
        self.declare_parameter('endpose_sub_topic', '/endpose_states_double_arm')
        self.declare_parameter('arm', 'left')
        self.declare_parameter('gripper_frame', 'gripper_link')
        self.declare_parameter(
            'output_dir',
            str(Path.home() / 'eye_in_hand_calibration'),
        )
        self.declare_parameter('show_preview', True)
        self.declare_parameter('preview_max_width', 1280)
        self.declare_parameter('process_interval_s', 0.15)
        self.declare_parameter('pose_sync_tolerance_s', 0.08)
        self.declare_parameter('min_pose_delta_translation_m', 0.02)
        self.declare_parameter('min_pose_delta_rotation_deg', 5.0)
        self.declare_parameter('min_samples', 12)
        self.declare_parameter('max_reproj_error_px', 3.0)
        self.declare_parameter('image_is_rectified', True)
        self.declare_parameter('hand_eye_method', 'all')

        self.declare_parameter('charuco_config_path', '')
        self.declare_parameter('squares_x', 7)
        self.declare_parameter('squares_y', 5)
        self.declare_parameter('square_length_m', 0.05)
        self.declare_parameter('marker_length_m', 0.03)
        self.declare_parameter('aruco_dictionary', 'DICT_4X4_50')
        self.declare_parameter('min_charuco_corners', 8)

    def _validate_parameters(self) -> None:
        if self.arm not in ('left', 'right'):
            raise ValueError("arm must be 'left' or 'right'.")
        if not self.endpose_sub_topic:
            raise ValueError('endpose_sub_topic must not be empty.')
        if not self.gripper_frame:
            raise ValueError('gripper_frame must not be empty.')
        if self.squares_x <= 1 or self.squares_y <= 1:
            raise ValueError('squares_x and squares_y must both be > 1.')
        if self.square_length_m <= 0.0:
            raise ValueError('square_length_m must be positive.')
        if self.marker_length_m <= 0.0:
            raise ValueError('marker_length_m must be positive.')
        if self.marker_length_m >= self.square_length_m:
            raise ValueError(
                'marker_length_m must be positive and smaller than '
                'square_length_m.'
            )
        if self.min_charuco_corners < 4:
            raise ValueError('min_charuco_corners must be >= 4.')
        if self.process_interval_s <= 0.0:
            raise ValueError('process_interval_s must be positive.')
        if self.pose_sync_tolerance_s <= 0.0:
            raise ValueError('pose_sync_tolerance_s must be positive.')
        if self.min_pose_delta_translation_m < 0.0:
            raise ValueError('min_pose_delta_translation_m must be >= 0.')
        if self.min_pose_delta_rotation_deg < 0.0:
            raise ValueError('min_pose_delta_rotation_deg must be >= 0.')
        if self.min_samples < 3:
            raise ValueError('min_samples must be >= 3.')
        if self.max_reproj_error_px <= 0.0:
            raise ValueError('max_reproj_error_px must be positive.')
        if self.hand_eye_method == 'all':
            return
        if self.hand_eye_method not in METHOD_NAME_TO_FLAG:
            supported = ', '.join(['all', *METHOD_NAME_TO_FLAG.keys()])
            raise ValueError(
                'Unsupported hand_eye_method. '
                f'Supported values: {supported}.'
            )

    def _load_charuco_config_if_needed(self) -> None:
        if not self.charuco_config_path:
            return

        board_config = load_charuco_config(self.charuco_config_path)
        self.squares_x = int(board_config['squares_x'])
        self.squares_y = int(board_config['squares_y'])
        self.square_length_m = float(board_config['square_length_m'])
        self.marker_length_m = float(board_config['marker_length_m'])
        self.aruco_dictionary_name = str(
            board_config['aruco_dictionary']
        ).strip()
        self.charuco_config_path = str(
            board_config.get('charuco_config_path', '') or ''
        )

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

        self.camera_frame_id = str(msg.header.frame_id)
        self.camera_info_snapshot = {
            'frame_id': self.camera_frame_id,
            'width': int(msg.width),
            'height': int(msg.height),
            'distortion_model': str(msg.distortion_model),
            'k': [float(value) for value in msg.k],
            'd': [float(value) for value in msg.d],
            'p': [float(value) for value in msg.p],
            'used_rectified_model': bool(self.image_is_rectified),
        }

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
        self.pose_frame_id = frame_id
        self.pose_buffer.append(
            PoseSnapshot(
                stamp_s=stamp_s,
                frame_id=frame_id,
                transform=transform,
            )
        )

    def _status_timer_callback(self) -> None:
        pose_count = len(self.pose_buffer)
        sample_count = len(self.samples)
        camera_ready = self.camera_matrix is not None
        latest_status = 'none'
        if self.last_valid_detection is not None:
            latest_status = 'ready'
            if self.last_valid_detection.pose_snapshot is None:
                latest_status = 'waiting_pose_sync'

        self.get_logger().info(
            'Status: '
            f'camera_info={"yes" if camera_ready else "no"}, '
            f'pose_buffer={pose_count}, '
            f'samples={sample_count}, '
            f'last_detection={latest_status}, '
            f'image_msgs={self.total_image_msgs}'
        )

    def _image_callback(self, msg: Image) -> None:
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
        self.last_valid_detection = observation

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
                self.square_length_m * 2.0,
                2,
            )

        self._draw_status_overlay(preview_image, observation)
        if self.show_preview:
            display_image = self._resize_for_preview(preview_image)
            cv2.imshow('Eye-In-Hand Calibration', display_image)
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
            f'pose_buffer: {len(self.pose_buffer)}  samples: {len(self.samples)}',
            f'markers: {marker_count}  charuco: {observation.detection.charuco_count}',
            f'arm: {self.arm}  pose_frame: {self.pose_frame_id or "unknown"}',
            f'gripper_frame: {self.gripper_frame}',
        ]

        if observation.reproj_mean_px is not None:
            lines.append(f'reproj: {observation.reproj_mean_px:.3f}px')
        else:
            lines.append('reproj: n/a')

        if observation.pose_offset_s is not None:
            pose_ms = observation.pose_offset_s * 1000.0
            lines.append(f'pose_dt: {pose_ms:.1f} ms')
        else:
            lines.append('pose_dt: n/a')

        lines.append('keys: s/space=capture  c=calibrate  r=reset  q=quit')

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
        if key == ord('r'):
            self.samples.clear()
            self._write_samples_snapshot()
            self.get_logger().info('Cleared captured samples.')
            return
        if key in (ord('s'), ord(' ')):
            self._capture_sample(observation, preview_image)
            return
        if key == ord('c'):
            self._run_calibration()
            return
        if key == ord('h'):
            self._log_help_once(force=True)

    def _capture_sample(
        self,
        observation: LiveObservation,
        preview_image: np.ndarray,
    ) -> None:
        if observation.rvec is None or observation.tvec is None:
            self.get_logger().warn(
                'Capture rejected: no valid board pose in current frame.'
            )
            return
        if observation.reproj_mean_px is None:
            self.get_logger().warn(
                'Capture rejected: missing reprojection error.'
            )
            return
        if observation.reproj_mean_px > self.max_reproj_error_px:
            self.get_logger().warn(
                'Capture rejected: reprojection error '
                f'{observation.reproj_mean_px:.3f}px > '
                f'{self.max_reproj_error_px:.3f}px.'
            )
            return
        if observation.pose_snapshot is None:
            self.get_logger().warn(
                'Capture rejected: no synchronized end-effector pose.'
            )
            return

        if self.samples:
            previous = self.samples[-1].t_base_gripper
            relative = invert_transform(previous) @ observation.pose_snapshot.transform
            translation_delta_m = float(np.linalg.norm(relative[:3, 3]))
            rotation_delta_deg = rotation_error_deg(
                np.eye(3, dtype=np.float64),
                relative[:3, :3],
            )
            if (
                translation_delta_m < self.min_pose_delta_translation_m
                and rotation_delta_deg < self.min_pose_delta_rotation_deg
            ):
                self.get_logger().warn(
                    'Capture rejected: pose change is too small. '
                    f'delta_t={translation_delta_m:.4f}m, '
                    f'delta_r={rotation_delta_deg:.2f}deg.'
                )
                return

        target_rotation, _ = cv2.Rodrigues(observation.rvec)
        t_cam_target = make_transform(
            target_rotation,
            observation.tvec.reshape(3),
        )

        sample_index = len(self.samples)
        raw_image_path = self.images_dir / f'sample_{sample_index:03d}_raw.png'
        preview_image_path = self.images_dir / f'sample_{sample_index:03d}_preview.png'
        cv2.imwrite(str(raw_image_path), observation.bgr_image)
        cv2.imwrite(str(preview_image_path), preview_image)

        sample = CapturedSample(
            sample_index=sample_index,
            image_stamp_s=observation.image_stamp_s,
            pose_stamp_s=observation.pose_snapshot.stamp_s,
            pose_offset_s=float(observation.pose_offset_s or 0.0),
            pose_frame_id=observation.pose_snapshot.frame_id,
            image_width=observation.image_width,
            image_height=observation.image_height,
            charuco_count=int(observation.detection.charuco_count),
            reproj_mean_px=float(observation.reproj_mean_px),
            t_base_gripper=observation.pose_snapshot.transform.copy(),
            t_cam_target=t_cam_target,
            raw_image_path=str(raw_image_path),
            preview_image_path=str(preview_image_path),
        )
        self.samples.append(sample)
        self._write_samples_snapshot()

        self.get_logger().info(
            'Captured sample '
            f'#{sample_index}: '
            f'charuco={sample.charuco_count}, '
            f'reproj={sample.reproj_mean_px:.3f}px, '
            f'pose_dt={sample.pose_offset_s * 1000.0:.1f}ms'
        )

    def _write_samples_snapshot(self) -> None:
        payload = {
            'created_at': datetime.now().isoformat(),
            'image_topic': self.image_topic,
            'camera_info_topic': self.camera_info_topic,
            'endpose_sub_topic': self.endpose_sub_topic,
            'arm': self.arm,
            'pose_frame_id': self.pose_frame_id or None,
            'camera_frame_id': self.camera_frame_id or None,
            'camera_info': self.camera_info_snapshot,
            'samples': [self._sample_to_json_dict(sample) for sample in self.samples],
        }
        output_path = self.session_dir / 'samples.json'
        with output_path.open('w', encoding='utf-8') as file_obj:
            json.dump(payload, file_obj, ensure_ascii=True, indent=2)

    def _sample_to_json_dict(self, sample: CapturedSample) -> dict[str, Any]:
        return {
            'sample_index': int(sample.sample_index),
            'image_stamp_s': float(sample.image_stamp_s),
            'pose_stamp_s': float(sample.pose_stamp_s),
            'pose_offset_s': float(sample.pose_offset_s),
            'pose_frame_id': sample.pose_frame_id,
            'image_width': int(sample.image_width),
            'image_height': int(sample.image_height),
            'charuco_count': int(sample.charuco_count),
            'reproj_mean_px': float(sample.reproj_mean_px),
            't_base_gripper': transform_to_json_dict(sample.t_base_gripper),
            't_cam_target': transform_to_json_dict(sample.t_cam_target),
            'raw_image_path': sample.raw_image_path,
            'preview_image_path': sample.preview_image_path,
        }

    def _selected_method_names(self) -> list[str]:
        if self.hand_eye_method == 'all':
            return list(METHOD_NAME_TO_FLAG.keys())
        return [self.hand_eye_method]

    def _run_calibration(self) -> None:
        sample_count = len(self.samples)
        if sample_count < self.min_samples:
            self.get_logger().warn(
                f'Need at least {self.min_samples} samples before calibration; '
                f'current={sample_count}.'
            )
            return

        gripper_to_base_rotations: list[np.ndarray] = []
        gripper_to_base_translations: list[np.ndarray] = []
        target_to_cam_rotations: list[np.ndarray] = []
        target_to_cam_translations: list[np.ndarray] = []
        gripper_to_base_transforms: list[np.ndarray] = []

        for sample in self.samples:
            gripper_to_base_transforms.append(sample.t_base_gripper)
            gripper_to_base_rotations.append(sample.t_base_gripper[:3, :3])
            gripper_to_base_translations.append(
                sample.t_base_gripper[:3, 3].reshape(3, 1)
            )
            target_to_cam_rotations.append(sample.t_cam_target[:3, :3])
            target_to_cam_translations.append(
                sample.t_cam_target[:3, 3].reshape(3, 1)
            )

        results: list[dict[str, Any]] = []
        for method_name in self._selected_method_names():
            try:
                rot_gc, trans_gc = cv2.calibrateHandEye(
                    gripper_to_base_rotations,
                    gripper_to_base_translations,
                    target_to_cam_rotations,
                    target_to_cam_translations,
                    method=METHOD_NAME_TO_FLAG[method_name],
                )
            except cv2.error as exc:
                self.get_logger().warn(
                    f'Calibration method {method_name} failed: {exc}'
                )
                continue

            t_gripper_cam = make_transform(
                orthonormalize_rotation(rot_gc),
                np.asarray(trans_gc, dtype=np.float64).reshape(3),
            )
            t_cam_gripper = invert_transform(t_gripper_cam)

            base_to_target_candidates = []
            for sample, t_base_gripper in zip(
                self.samples,
                gripper_to_base_transforms,
            ):
                base_to_target_candidates.append(
                    t_base_gripper @ t_gripper_cam @ sample.t_cam_target
                )
            t_base_target = average_transforms(base_to_target_candidates)

            residuals = []
            mount_consistency = []
            for sample, candidate in zip(self.samples, base_to_target_candidates):
                t_gripper_base = invert_transform(sample.t_base_gripper)
                predicted = t_cam_gripper @ t_gripper_base @ t_base_target
                translation_error_mm = float(
                    np.linalg.norm(
                        predicted[:3, 3] - sample.t_cam_target[:3, 3]
                    ) * 1000.0
                )
                rotation_error = rotation_error_deg(
                    predicted[:3, :3],
                    sample.t_cam_target[:3, :3],
                )
                residuals.append(
                    {
                        'sample_index': int(sample.sample_index),
                        'translation_error_mm': translation_error_mm,
                        'rotation_error_deg': rotation_error,
                    }
                )
                mount_consistency.append(
                    {
                        'sample_index': int(sample.sample_index),
                        'translation_error_mm': float(
                            np.linalg.norm(
                                candidate[:3, 3] - t_base_target[:3, 3]
                            ) * 1000.0
                        ),
                        'rotation_error_deg': rotation_error_deg(
                            candidate[:3, :3],
                            t_base_target[:3, :3],
                        ),
                    }
                )

            translation_errors = np.asarray(
                [item['translation_error_mm'] for item in residuals],
                dtype=np.float64,
            )
            rotation_errors = np.asarray(
                [item['rotation_error_deg'] for item in residuals],
                dtype=np.float64,
            )
            mount_translation_errors = np.asarray(
                [item['translation_error_mm'] for item in mount_consistency],
                dtype=np.float64,
            )
            mount_rotation_errors = np.asarray(
                [item['rotation_error_deg'] for item in mount_consistency],
                dtype=np.float64,
            )

            results.append(
                {
                    'method': method_name,
                    't_gripper_cam': t_gripper_cam,
                    't_cam_gripper': t_cam_gripper,
                    't_base_target': t_base_target,
                    'residual_mean_translation_mm': float(
                        np.mean(translation_errors)
                    ),
                    'residual_max_translation_mm': float(
                        np.max(translation_errors)
                    ),
                    'residual_mean_rotation_deg': float(
                        np.mean(rotation_errors)
                    ),
                    'residual_max_rotation_deg': float(
                        np.max(rotation_errors)
                    ),
                    'mount_mean_translation_mm': float(
                        np.mean(mount_translation_errors)
                    ),
                    'mount_max_translation_mm': float(
                        np.max(mount_translation_errors)
                    ),
                    'mount_mean_rotation_deg': float(
                        np.mean(mount_rotation_errors)
                    ),
                    'mount_max_rotation_deg': float(
                        np.max(mount_rotation_errors)
                    ),
                    'residuals': residuals,
                    'mount_consistency': mount_consistency,
                }
            )

        if not results:
            self.get_logger().error('All calibration methods failed.')
            return

        best_result = min(
            results,
            key=lambda item: (
                item['residual_mean_translation_mm'],
                item['residual_mean_rotation_deg'],
            ),
        )
        result_path = self._write_calibration_result(best_result, results)

        translation = best_result['t_gripper_cam'][:3, 3]
        quaternion = rotation_matrix_to_quaternion(
            best_result['t_gripper_cam'][:3, :3]
        )
        self.get_logger().info(
            'Best result: '
            f'method={best_result["method"]}, '
            f'mean_t={best_result["residual_mean_translation_mm"]:.2f}mm, '
            f'mean_r={best_result["residual_mean_rotation_deg"]:.3f}deg'
        )
        self.get_logger().info(
            'gripper_T_camera: '
            f't=[{translation[0]:.6f}, {translation[1]:.6f}, '
            f'{translation[2]:.6f}] m, '
            f'q=[{quaternion[0]:.6f}, {quaternion[1]:.6f}, '
            f'{quaternion[2]:.6f}, {quaternion[3]:.6f}]'
        )
        self.get_logger().info(f'Saved calibration result to: {result_path}')

    def _write_calibration_result(
        self,
        best_result: dict[str, Any],
        all_results: list[dict[str, Any]],
    ) -> Path:
        base_frame = self.pose_frame_id or 'base_link'
        parent_frame = self.gripper_frame
        child_frame = self.camera_frame_id or 'camera_link'
        summary = {
            'created_at': datetime.now().isoformat(),
            'image_topic': self.image_topic,
            'camera_info_topic': self.camera_info_topic,
            'endpose_sub_topic': self.endpose_sub_topic,
            'arm': self.arm,
            'pose_frame_id': base_frame,
            'gripper_frame': parent_frame,
            'camera_frame_id': child_frame,
            'camera_info': self.camera_info_snapshot,
            'board': self.charuco_helper.board_config_snapshot(),
            'sample_count': int(len(self.samples)),
            'best_method': best_result['method'],
            'best_result': self._serialize_calibration_result(best_result),
            'all_results': [
                self._serialize_calibration_result(result)
                for result in all_results
            ],
            'samples': [self._sample_to_json_dict(sample) for sample in self.samples],
            'static_transform_publisher': self._build_static_transform_command(
                transform=best_result['t_gripper_cam'],
                parent_frame=parent_frame,
                child_frame=child_frame,
            ),
        }

        output_path = self.session_dir / 'eye_in_hand_result.json'
        with output_path.open('w', encoding='utf-8') as file_obj:
            json.dump(summary, file_obj, ensure_ascii=True, indent=2)

        text_lines = [
            f'best_method: {best_result["method"]}',
            f'sample_count: {len(self.samples)}',
            f'pose_frame: {base_frame}',
            f'gripper_frame: {parent_frame}',
            f'camera_frame: {child_frame}',
            'mean_translation_error_mm: '
            f'{best_result["residual_mean_translation_mm"]:.6f}',
            'mean_rotation_error_deg: '
            f'{best_result["residual_mean_rotation_deg"]:.6f}',
            'gripper_T_camera:',
            json.dumps(
                transform_to_json_dict(best_result['t_gripper_cam']),
                ensure_ascii=True,
                indent=2,
            ),
            'static_transform_publisher:',
            summary['static_transform_publisher'],
        ]
        text_path = self.session_dir / 'eye_in_hand_result.txt'
        with text_path.open('w', encoding='utf-8') as file_obj:
            file_obj.write('\n'.join(text_lines) + '\n')
        return output_path

    def _serialize_calibration_result(
        self,
        result: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            'method': result['method'],
            't_gripper_cam': transform_to_json_dict(result['t_gripper_cam']),
            't_cam_gripper': transform_to_json_dict(result['t_cam_gripper']),
            't_base_target': transform_to_json_dict(result['t_base_target']),
            'residual_mean_translation_mm': float(
                result['residual_mean_translation_mm']
            ),
            'residual_max_translation_mm': float(
                result['residual_max_translation_mm']
            ),
            'residual_mean_rotation_deg': float(
                result['residual_mean_rotation_deg']
            ),
            'residual_max_rotation_deg': float(
                result['residual_max_rotation_deg']
            ),
            'mount_mean_translation_mm': float(
                result['mount_mean_translation_mm']
            ),
            'mount_max_translation_mm': float(
                result['mount_max_translation_mm']
            ),
            'mount_mean_rotation_deg': float(result['mount_mean_rotation_deg']),
            'mount_max_rotation_deg': float(result['mount_max_rotation_deg']),
            'residuals': list(result['residuals']),
            'mount_consistency': list(result['mount_consistency']),
        }

    def _build_static_transform_command(
        self,
        transform: np.ndarray,
        parent_frame: str,
        child_frame: str,
    ) -> str:
        translation = transform[:3, 3]
        quaternion = rotation_matrix_to_quaternion(transform[:3, :3])
        return (
            'ros2 run tf2_ros static_transform_publisher '
            f'{translation[0]:.9f} {translation[1]:.9f} {translation[2]:.9f} '
            f'{quaternion[0]:.9f} {quaternion[1]:.9f} {quaternion[2]:.9f} '
            f'{quaternion[3]:.9f} {parent_frame} {child_frame}'
        )

    def _log_help_once(self, force: bool = False) -> None:
        if self._printed_help and not force:
            return
        self._printed_help = True
        self.get_logger().info(
            'Controls: '
            '[s]/[space]=capture sample, '
            '[c]=run calibration, '
            '[r]=clear samples, '
            '[q]=quit.'
        )


def main(args: list[str] | None = None) -> int:
    rclpy.init(args=args)
    node: EyeInHandCalibrationNode | None = None
    try:
        node = EyeInHandCalibrationNode()
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
