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
import numpy as np
import rclpy
from common.msg import OculusControllers
from geometry_msgs.msg import Pose
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image


METHOD_NAME_TO_FLAG = {
    'tsai': cv2.CALIB_HAND_EYE_TSAI,
    'park': cv2.CALIB_HAND_EYE_PARK,
    'horaud': cv2.CALIB_HAND_EYE_HORAUD,
    'andreff': cv2.CALIB_HAND_EYE_ANDREFF,
    'daniilidis': cv2.CALIB_HAND_EYE_DANIILIDIS,
}


@dataclass
class PoseSnapshot:
    stamp_s: float
    frame_id: str
    transform: np.ndarray


@dataclass
class CharucoDetection:
    marker_corners: list[np.ndarray]
    marker_ids: np.ndarray | None
    charuco_corners: np.ndarray | None
    charuco_ids: np.ndarray | None
    charuco_count: int = 0


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


def _stamp_to_seconds(stamp: Any) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def _clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(max_value, value))


def _make_transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    transform[:3, 3] = np.asarray(translation, dtype=np.float64).reshape(3)
    return transform


def _invert_transform(transform: np.ndarray) -> np.ndarray:
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    inverse = np.eye(4, dtype=np.float64)
    inverse[:3, :3] = rotation.T
    inverse[:3, 3] = -rotation.T @ translation
    return inverse


def _orthonormalize_rotation(rotation: np.ndarray) -> np.ndarray:
    u_matrix, _, v_transpose = np.linalg.svd(rotation)
    normalized = u_matrix @ v_transpose
    if np.linalg.det(normalized) < 0.0:
        u_matrix[:, -1] *= -1.0
        normalized = u_matrix @ v_transpose
    return normalized


def _rotation_error_deg(rotation_a: np.ndarray, rotation_b: np.ndarray) -> float:
    delta = rotation_a.T @ rotation_b
    trace = float(np.trace(delta))
    cos_theta = _clamp((trace - 1.0) * 0.5, -1.0, 1.0)
    return math.degrees(math.acos(cos_theta))


def _average_rotations(rotations: list[np.ndarray]) -> np.ndarray:
    if not rotations:
        raise ValueError('No rotations to average.')
    accumulator = np.zeros((3, 3), dtype=np.float64)
    for rotation in rotations:
        accumulator += rotation
    return _orthonormalize_rotation(accumulator)


def _average_transforms(transforms: list[np.ndarray]) -> np.ndarray:
    if not transforms:
        raise ValueError('No transforms to average.')
    mean_rotation = _average_rotations([transform[:3, :3] for transform in transforms])
    mean_translation = np.mean(
        [transform[:3, 3] for transform in transforms],
        axis=0,
    )
    return _make_transform(mean_rotation, mean_translation)


def _quaternion_to_rotation_matrix(quaternion_xyzw: np.ndarray) -> np.ndarray:
    x_value, y_value, z_value, w_value = quaternion_xyzw
    xx = x_value * x_value
    yy = y_value * y_value
    zz = z_value * z_value
    xy = x_value * y_value
    xz = x_value * z_value
    yz = y_value * z_value
    wx = w_value * x_value
    wy = w_value * y_value
    wz = w_value * z_value

    return np.asarray(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )


def _rotation_matrix_to_quaternion(rotation: np.ndarray) -> np.ndarray:
    matrix = _orthonormalize_rotation(rotation)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        w_value = 0.25 * scale
        x_value = (matrix[2, 1] - matrix[1, 2]) / scale
        y_value = (matrix[0, 2] - matrix[2, 0]) / scale
        z_value = (matrix[1, 0] - matrix[0, 1]) / scale
    elif matrix[0, 0] > matrix[1, 1] and matrix[0, 0] > matrix[2, 2]:
        scale = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
        w_value = (matrix[2, 1] - matrix[1, 2]) / scale
        x_value = 0.25 * scale
        y_value = (matrix[0, 1] + matrix[1, 0]) / scale
        z_value = (matrix[0, 2] + matrix[2, 0]) / scale
    elif matrix[1, 1] > matrix[2, 2]:
        scale = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
        w_value = (matrix[0, 2] - matrix[2, 0]) / scale
        x_value = (matrix[0, 1] + matrix[1, 0]) / scale
        y_value = 0.25 * scale
        z_value = (matrix[1, 2] + matrix[2, 1]) / scale
    else:
        scale = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
        w_value = (matrix[1, 0] - matrix[0, 1]) / scale
        x_value = (matrix[0, 2] + matrix[2, 0]) / scale
        y_value = (matrix[1, 2] + matrix[2, 1]) / scale
        z_value = 0.25 * scale
    quaternion = np.asarray([x_value, y_value, z_value, w_value], dtype=np.float64)
    quaternion /= np.linalg.norm(quaternion)
    return quaternion


def _pose_to_transform(pose: Pose) -> np.ndarray:
    quaternion = np.asarray(
        [
            float(pose.orientation.x),
            float(pose.orientation.y),
            float(pose.orientation.z),
            float(pose.orientation.w),
        ],
        dtype=np.float64,
    )
    norm = float(np.linalg.norm(quaternion))
    if norm < 1e-12:
        raise ValueError('Pose quaternion norm is zero.')
    quaternion /= norm
    rotation = _quaternion_to_rotation_matrix(quaternion)
    translation = np.asarray(
        [
            float(pose.position.x),
            float(pose.position.y),
            float(pose.position.z),
        ],
        dtype=np.float64,
    )
    return _make_transform(rotation, translation)


def _transform_to_json_dict(transform: np.ndarray) -> dict[str, Any]:
    return {
        'translation_m': [float(value) for value in transform[:3, 3]],
        'quaternion_xyzw': [
            float(value)
            for value in _rotation_matrix_to_quaternion(transform[:3, :3])
        ],
        'matrix': [
            [float(value) for value in row]
            for row in transform.tolist()
        ],
    }


def _opencv_supports_charuco() -> bool:
    if not hasattr(cv2, 'aruco'):
        return False
    aruco_module = cv2.aruco
    has_board = hasattr(aruco_module, 'CharucoBoard_create') or hasattr(aruco_module, 'CharucoBoard')
    has_dictionary = hasattr(aruco_module, 'getPredefinedDictionary')
    has_detector = hasattr(aruco_module, 'detectMarkers') or hasattr(aruco_module, 'ArucoDetector')
    has_interpolation = hasattr(aruco_module, 'interpolateCornersCharuco')
    return has_board and has_dictionary and has_detector and has_interpolation


class EyeToHandCalibrationNode(Node):
    def __init__(self) -> None:
        super().__init__('eye_to_hand_calibration')

        self._declare_parameters()

        if not _opencv_supports_charuco():
            raise RuntimeError(
                'Current OpenCV build does not expose cv2.aruco Charuco APIs.'
            )

        self.image_topic = str(self.get_parameter('image_topic').value)
        self.camera_info_topic = str(self.get_parameter('camera_info_topic').value)
        self.endpose_sub_topic = str(
            self.get_parameter('endpose_sub_topic').value
        ).strip()
        self.arm = str(self.get_parameter('arm').value).strip().lower()
        self.output_root = Path(str(self.get_parameter('output_dir').value)).expanduser()
        self.show_preview = bool(self.get_parameter('show_preview').value)
        self.preview_max_width = int(self.get_parameter('preview_max_width').value)
        self.process_interval_s = float(self.get_parameter('process_interval_s').value)
        self.pose_sync_tolerance_s = float(self.get_parameter('pose_sync_tolerance_s').value)
        self.min_pose_delta_translation_m = float(
            self.get_parameter('min_pose_delta_translation_m').value
        )
        self.min_pose_delta_rotation_deg = float(
            self.get_parameter('min_pose_delta_rotation_deg').value
        )
        self.min_samples = int(self.get_parameter('min_samples').value)
        self.max_reproj_error_px = float(self.get_parameter('max_reproj_error_px').value)
        self.image_is_rectified = bool(self.get_parameter('image_is_rectified').value)
        self.hand_eye_method = str(self.get_parameter('hand_eye_method').value).strip().lower()

        self.charuco_config_path = str(self.get_parameter('charuco_config_path').value).strip()
        self.squares_x = int(self.get_parameter('squares_x').value)
        self.squares_y = int(self.get_parameter('squares_y').value)
        self.square_length_m = float(self.get_parameter('square_length_m').value)
        self.marker_length_m = float(self.get_parameter('marker_length_m').value)
        self.aruco_dictionary_name = str(self.get_parameter('aruco_dictionary').value).strip()
        self.min_charuco_corners = int(self.get_parameter('min_charuco_corners').value)

        self._load_charuco_config_if_needed()
        self._validate_parameters()

        self.aruco_dictionary_id = self._resolve_aruco_dictionary_id(self.aruco_dictionary_name)
        self.aruco_dictionary = cv2.aruco.getPredefinedDictionary(self.aruco_dictionary_id)
        self.aruco_detector_params = self._create_aruco_detector_params()
        self.aruco_detector = self._create_aruco_detector()
        self.charuco_board = self._create_charuco_board()
        self.charuco_corner_object_points = self._get_charuco_board_corners()

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.session_dir = self.output_root / f'eye_to_hand_calibration_{timestamp}'
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
            'Eye-to-hand calibrator started: '
            f'image_topic={self.image_topic}, '
            f'camera_info_topic={self.camera_info_topic}, '
            f'endpose_sub_topic={self.endpose_sub_topic}, '
            f'arm={self.arm}, '
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
        self.declare_parameter('camera_info_topic', '/camera/color/camera_info')
        self.declare_parameter('endpose_sub_topic', '/endpose_states_double_arm')
        self.declare_parameter('arm', 'left')
        self.declare_parameter(
            'output_dir',
            str(Path.home() / 'eye_to_hand_calibration'),
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
        if self.squares_x <= 1 or self.squares_y <= 1:
            raise ValueError('squares_x and squares_y must both be > 1.')
        if self.square_length_m <= 0.0:
            raise ValueError('square_length_m must be positive.')
        if self.marker_length_m <= 0.0 or self.marker_length_m >= self.square_length_m:
            raise ValueError('marker_length_m must be positive and smaller than square_length_m.')
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
        if self.hand_eye_method != 'all' and self.hand_eye_method not in METHOD_NAME_TO_FLAG:
            supported = ', '.join(['all', *METHOD_NAME_TO_FLAG.keys()])
            raise ValueError(f'Unsupported hand_eye_method. Supported values: {supported}.')

    def _load_charuco_config_if_needed(self) -> None:
        if not self.charuco_config_path:
            return

        config_path = Path(self.charuco_config_path).expanduser()
        if not config_path.is_file():
            raise FileNotFoundError(f'ChArUco config not found: {config_path}')

        with config_path.open('r', encoding='utf-8') as file_obj:
            config = json.load(file_obj)

        board_type = str(config.get('board_type', 'charuco')).strip().lower()
        if board_type != 'charuco':
            raise ValueError('Only ChArUco board configs are supported.')

        self.squares_x = int(config['squares_x'])
        self.squares_y = int(config['squares_y'])
        self.square_length_m = self._config_length_to_meters(
            config=config,
            meters_key='square_length_m',
            millimeters_key='square_length_mm',
        )
        self.marker_length_m = self._config_length_to_meters(
            config=config,
            meters_key='marker_length_m',
            millimeters_key='marker_length_mm',
        )
        self.aruco_dictionary_name = str(config['dictionary']).strip()
        self.charuco_config_path = str(config_path)

    def _config_length_to_meters(
        self,
        config: dict[str, Any],
        meters_key: str,
        millimeters_key: str,
    ) -> float:
        if meters_key in config:
            return float(config[meters_key])
        if millimeters_key in config:
            return float(config[millimeters_key]) / 1000.0
        raise KeyError(f'Missing board length field: {meters_key} / {millimeters_key}')

    def _resolve_aruco_dictionary_id(self, dictionary_name: str) -> int:
        if not hasattr(cv2.aruco, dictionary_name):
            raise ValueError(f'Unsupported ArUco dictionary: {dictionary_name}')
        return int(getattr(cv2.aruco, dictionary_name))

    def _create_aruco_detector_params(self) -> Any:
        if hasattr(cv2.aruco, 'DetectorParameters_create'):
            return cv2.aruco.DetectorParameters_create()
        if hasattr(cv2.aruco, 'DetectorParameters'):
            return cv2.aruco.DetectorParameters()
        raise RuntimeError('OpenCV aruco detector parameters API is unavailable.')

    def _create_aruco_detector(self) -> Any | None:
        if hasattr(cv2.aruco, 'ArucoDetector'):
            return cv2.aruco.ArucoDetector(self.aruco_dictionary, self.aruco_detector_params)
        return None

    def _create_charuco_board(self) -> Any:
        if hasattr(cv2.aruco, 'CharucoBoard_create'):
            return cv2.aruco.CharucoBoard_create(
                self.squares_x,
                self.squares_y,
                self.square_length_m,
                self.marker_length_m,
                self.aruco_dictionary,
            )
        if hasattr(cv2.aruco, 'CharucoBoard'):
            try:
                return cv2.aruco.CharucoBoard(
                    (self.squares_x, self.squares_y),
                    self.square_length_m,
                    self.marker_length_m,
                    self.aruco_dictionary,
                )
            except TypeError:
                return cv2.aruco.CharucoBoard(
                    self.squares_x,
                    self.squares_y,
                    self.square_length_m,
                    self.marker_length_m,
                    self.aruco_dictionary,
                )
        raise RuntimeError('OpenCV Charuco board API is unavailable.')

    def _get_charuco_board_corners(self) -> np.ndarray:
        board_corners = getattr(self.charuco_board, 'chessboardCorners', None)
        if board_corners is None and hasattr(self.charuco_board, 'getChessboardCorners'):
            board_corners = self.charuco_board.getChessboardCorners()
        if board_corners is None:
            raise RuntimeError('Failed to read ChArUco board corners from OpenCV.')
        return np.asarray(board_corners, dtype=np.float32)

    def _charuco_object_points_from_ids(self, charuco_ids: np.ndarray) -> np.ndarray:
        ids = charuco_ids.reshape(-1).astype(np.int32)
        return self.charuco_corner_object_points[ids]

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
            transform = _pose_to_transform(pose)
        except ValueError as exc:
            self.get_logger().warn(f'Ignoring invalid {self.arm} pose: {exc}')
            return

        stamp_s = _stamp_to_seconds(msg.header.stamp)
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
            bgr_image = self._image_msg_to_bgr(msg)
        except ValueError as exc:
            self.get_logger().warn(f'Unsupported image encoding: {exc}')
            return

        preview_image = bgr_image.copy()
        observation = self._build_live_observation(msg, bgr_image)
        self.last_valid_detection = observation

        if observation.detection.marker_ids is not None:
            self._draw_detection_overlay(preview_image, observation.detection)
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
            cv2.imshow('Eye-To-Hand Calibration', display_image)
            self._handle_preview_key(observation, preview_image)

    def _build_live_observation(self, msg: Image, bgr_image: np.ndarray) -> LiveObservation:
        gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        detection = self._detect_charuco(gray_image)
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
            solve_result = self._solve_target_pose(
                charuco_corners=detection.charuco_corners,
                charuco_ids=detection.charuco_ids,
            )
            if solve_result is not None:
                rvec, tvec, reproj_mean_px = solve_result

        image_stamp_s = _stamp_to_seconds(msg.header.stamp)
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

    def _image_msg_to_bgr(self, msg: Image) -> np.ndarray:
        encoding = msg.encoding.lower()
        row_data = np.frombuffer(msg.data, dtype=np.uint8)

        if encoding == 'bgr8':
            array = row_data.reshape(msg.height, msg.step)[:, : msg.width * 3]
            return array.reshape(msg.height, msg.width, 3).copy()
        if encoding == 'rgb8':
            array = row_data.reshape(msg.height, msg.step)[:, : msg.width * 3]
            return cv2.cvtColor(
                array.reshape(msg.height, msg.width, 3),
                cv2.COLOR_RGB2BGR,
            )
        if encoding in ('mono8', '8uc1'):
            array = row_data.reshape(msg.height, msg.step)[:, : msg.width]
            return cv2.cvtColor(array, cv2.COLOR_GRAY2BGR)
        if encoding == 'bgra8':
            array = row_data.reshape(msg.height, msg.step)[:, : msg.width * 4]
            return cv2.cvtColor(
                array.reshape(msg.height, msg.width, 4),
                cv2.COLOR_BGRA2BGR,
            )
        if encoding == 'rgba8':
            array = row_data.reshape(msg.height, msg.step)[:, : msg.width * 4]
            return cv2.cvtColor(
                array.reshape(msg.height, msg.width, 4),
                cv2.COLOR_RGBA2BGR,
            )
        raise ValueError(encoding)

    def _detect_charuco(self, gray_image: np.ndarray) -> CharucoDetection:
        if self.aruco_detector is not None:
            marker_corners, marker_ids, _ = self.aruco_detector.detectMarkers(gray_image)
        else:
            marker_corners, marker_ids, _ = cv2.aruco.detectMarkers(
                gray_image,
                self.aruco_dictionary,
                parameters=self.aruco_detector_params,
            )

        if marker_ids is None or len(marker_ids) == 0:
            return CharucoDetection(
                marker_corners=[],
                marker_ids=None,
                charuco_corners=None,
                charuco_ids=None,
            )

        interpolate_kwargs: dict[str, Any] = {}
        if self.camera_matrix is not None and self.dist_coeffs is not None:
            interpolate_kwargs['cameraMatrix'] = self.camera_matrix
            interpolate_kwargs['distCoeffs'] = self.dist_coeffs

        charuco_count, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
            marker_corners,
            marker_ids,
            gray_image,
            self.charuco_board,
            **interpolate_kwargs,
        )

        return CharucoDetection(
            marker_corners=list(marker_corners),
            marker_ids=marker_ids.astype(np.int32),
            charuco_corners=(
                None if charuco_corners is None else charuco_corners.astype(np.float32)
            ),
            charuco_ids=None if charuco_ids is None else charuco_ids.astype(np.int32),
            charuco_count=int(charuco_count),
        )

    def _solve_target_pose(
        self,
        charuco_corners: np.ndarray,
        charuco_ids: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, float] | None:
        assert self.camera_matrix is not None
        assert self.dist_coeffs is not None

        object_points = self._charuco_object_points_from_ids(charuco_ids)
        image_points = charuco_corners.reshape(-1, 2)
        if object_points.shape[0] < 4:
            return None

        success, rvec, tvec = cv2.solvePnP(
            object_points,
            image_points,
            self.camera_matrix,
            self.dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not success:
            return None

        projected, _ = cv2.projectPoints(
            object_points,
            rvec,
            tvec,
            self.camera_matrix,
            self.dist_coeffs,
        )
        projected = projected.reshape(-1, 2)
        reprojection = float(np.mean(np.linalg.norm(projected - image_points, axis=1)))
        return rvec.reshape(3, 1), tvec.reshape(3, 1), reprojection

    def _lookup_pose(self, image_stamp_s: float) -> tuple[PoseSnapshot | None, float | None]:
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

    def _draw_detection_overlay(self, image: np.ndarray, detection: CharucoDetection) -> None:
        for marker in detection.marker_corners:
            corners = marker.reshape(-1, 2).astype(np.int32)
            cv2.polylines(image, [corners], True, (255, 0, 255), 2, cv2.LINE_AA)

        if detection.charuco_corners is None:
            return
        for point in detection.charuco_corners.reshape(-1, 2):
            center = (int(round(point[0])), int(round(point[1])))
            cv2.circle(image, center, 4, (0, 255, 0), -1)

    def _draw_status_overlay(self, image: np.ndarray, observation: LiveObservation) -> None:
        marker_count = 0
        if observation.detection.marker_ids is not None:
            marker_count = int(observation.detection.marker_ids.shape[0])

        lines = [
            f'camera_info: {"ready" if self.camera_matrix is not None else "waiting"}',
            f'pose_buffer: {len(self.pose_buffer)}  samples: {len(self.samples)}',
            f'markers: {marker_count}  charuco: {observation.detection.charuco_count}',
            f'arm: {self.arm}  pose_frame: {self.pose_frame_id or "unknown"}',
        ]

        if observation.reproj_mean_px is not None:
            lines.append(f'reproj: {observation.reproj_mean_px:.3f}px')
        else:
            lines.append('reproj: n/a')

        if observation.pose_offset_s is not None:
            lines.append(f'pose_dt: {observation.pose_offset_s * 1000.0:.1f} ms')
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
        if key in (ord('r'),):
            self.samples.clear()
            self._write_samples_snapshot()
            self.get_logger().info('Cleared captured samples.')
            return
        if key in (ord('s'), ord(' ')):
            self._capture_sample(observation, preview_image)
            return
        if key in (ord('c'),):
            self._run_calibration()
            return
        if key in (ord('h'),):
            self._log_help_once(force=True)

    def _capture_sample(
        self,
        observation: LiveObservation,
        preview_image: np.ndarray,
    ) -> None:
        if observation.rvec is None or observation.tvec is None:
            self.get_logger().warn('Capture rejected: no valid board pose in current frame.')
            return
        if observation.reproj_mean_px is None:
            self.get_logger().warn('Capture rejected: missing reprojection error.')
            return
        if observation.reproj_mean_px > self.max_reproj_error_px:
            self.get_logger().warn(
                'Capture rejected: reprojection error '
                f'{observation.reproj_mean_px:.3f}px > {self.max_reproj_error_px:.3f}px.'
            )
            return
        if observation.pose_snapshot is None:
            self.get_logger().warn('Capture rejected: no synchronized end-effector pose.')
            return

        if self.samples:
            previous = self.samples[-1].t_base_gripper
            relative = _invert_transform(previous) @ observation.pose_snapshot.transform
            translation_delta_m = float(np.linalg.norm(relative[:3, 3]))
            rotation_delta_deg = _rotation_error_deg(
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
        t_cam_target = _make_transform(target_rotation, observation.tvec.reshape(3))

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
            't_base_gripper': _transform_to_json_dict(sample.t_base_gripper),
            't_cam_target': _transform_to_json_dict(sample.t_cam_target),
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
            t_gripper_base = _invert_transform(sample.t_base_gripper)
            gripper_to_base_transforms.append(t_gripper_base)
            gripper_to_base_rotations.append(t_gripper_base[:3, :3])
            gripper_to_base_translations.append(t_gripper_base[:3, 3].reshape(3, 1))
            target_to_cam_rotations.append(sample.t_cam_target[:3, :3])
            target_to_cam_translations.append(sample.t_cam_target[:3, 3].reshape(3, 1))

        results: list[dict[str, Any]] = []
        for method_name in self._selected_method_names():
            try:
                rotation_base_cam, translation_base_cam = cv2.calibrateHandEye(
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

            t_base_cam = _make_transform(
                _orthonormalize_rotation(rotation_base_cam),
                np.asarray(translation_base_cam, dtype=np.float64).reshape(3),
            )
            t_cam_base = _invert_transform(t_base_cam)

            gripper_to_target_candidates = []
            for sample, t_gripper_base in zip(self.samples, gripper_to_base_transforms):
                gripper_to_target_candidates.append(
                    t_gripper_base @ t_base_cam @ sample.t_cam_target
                )
            t_gripper_target = _average_transforms(gripper_to_target_candidates)

            residuals = []
            mount_consistency = []
            for sample, candidate in zip(self.samples, gripper_to_target_candidates):
                predicted = t_cam_base @ sample.t_base_gripper @ t_gripper_target
                translation_error_mm = float(
                    np.linalg.norm(predicted[:3, 3] - sample.t_cam_target[:3, 3]) * 1000.0
                )
                rotation_error = _rotation_error_deg(
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
                            np.linalg.norm(candidate[:3, 3] - t_gripper_target[:3, 3]) * 1000.0
                        ),
                        'rotation_error_deg': _rotation_error_deg(
                            candidate[:3, :3],
                            t_gripper_target[:3, :3],
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
                    't_base_cam': t_base_cam,
                    't_cam_base': t_cam_base,
                    't_gripper_target': t_gripper_target,
                    'residual_mean_translation_mm': float(np.mean(translation_errors)),
                    'residual_max_translation_mm': float(np.max(translation_errors)),
                    'residual_mean_rotation_deg': float(np.mean(rotation_errors)),
                    'residual_max_rotation_deg': float(np.max(rotation_errors)),
                    'mount_mean_translation_mm': float(np.mean(mount_translation_errors)),
                    'mount_max_translation_mm': float(np.max(mount_translation_errors)),
                    'mount_mean_rotation_deg': float(np.mean(mount_rotation_errors)),
                    'mount_max_rotation_deg': float(np.max(mount_rotation_errors)),
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

        translation = best_result['t_base_cam'][:3, 3]
        quaternion = _rotation_matrix_to_quaternion(best_result['t_base_cam'][:3, :3])
        self.get_logger().info(
            'Best result: '
            f'method={best_result["method"]}, '
            f'mean_t={best_result["residual_mean_translation_mm"]:.2f}mm, '
            f'mean_r={best_result["residual_mean_rotation_deg"]:.3f}deg'
        )
        self.get_logger().info(
            'base_T_camera: '
            f't=[{translation[0]:.6f}, {translation[1]:.6f}, {translation[2]:.6f}] m, '
            f'q=[{quaternion[0]:.6f}, {quaternion[1]:.6f}, {quaternion[2]:.6f}, {quaternion[3]:.6f}]'
        )
        self.get_logger().info(f'Saved calibration result to: {result_path}')

    def _write_calibration_result(
        self,
        best_result: dict[str, Any],
        all_results: list[dict[str, Any]],
    ) -> Path:
        parent_frame = self.pose_frame_id or 'base_link'
        child_frame = self.camera_frame_id or 'camera_link'
        summary = {
            'created_at': datetime.now().isoformat(),
            'image_topic': self.image_topic,
            'camera_info_topic': self.camera_info_topic,
            'endpose_sub_topic': self.endpose_sub_topic,
            'arm': self.arm,
            'pose_frame_id': parent_frame,
            'camera_frame_id': child_frame,
            'camera_info': self.camera_info_snapshot,
            'board': {
                'type': 'charuco',
                'squares_x': int(self.squares_x),
                'squares_y': int(self.squares_y),
                'square_length_m': float(self.square_length_m),
                'marker_length_m': float(self.marker_length_m),
                'aruco_dictionary': self.aruco_dictionary_name,
                'charuco_config_path': self.charuco_config_path or None,
            },
            'sample_count': int(len(self.samples)),
            'best_method': best_result['method'],
            'best_result': self._serialize_calibration_result(best_result),
            'all_results': [
                self._serialize_calibration_result(result)
                for result in all_results
            ],
            'samples': [self._sample_to_json_dict(sample) for sample in self.samples],
            'static_transform_publisher': self._build_static_transform_command(
                transform=best_result['t_base_cam'],
                parent_frame=parent_frame,
                child_frame=child_frame,
            ),
        }

        output_path = self.session_dir / 'eye_to_hand_result.json'
        with output_path.open('w', encoding='utf-8') as file_obj:
            json.dump(summary, file_obj, ensure_ascii=True, indent=2)

        text_lines = [
            f'best_method: {best_result["method"]}',
            f'sample_count: {len(self.samples)}',
            f'pose_frame: {parent_frame}',
            f'camera_frame: {child_frame}',
            f'mean_translation_error_mm: {best_result["residual_mean_translation_mm"]:.6f}',
            f'mean_rotation_error_deg: {best_result["residual_mean_rotation_deg"]:.6f}',
            'base_T_camera:',
            json.dumps(
                _transform_to_json_dict(best_result['t_base_cam']),
                ensure_ascii=True,
                indent=2,
            ),
            'static_transform_publisher:',
            summary['static_transform_publisher'],
        ]
        with (self.session_dir / 'eye_to_hand_result.txt').open('w', encoding='utf-8') as file_obj:
            file_obj.write('\n'.join(text_lines) + '\n')
        return output_path

    def _serialize_calibration_result(self, result: dict[str, Any]) -> dict[str, Any]:
        return {
            'method': result['method'],
            't_base_cam': _transform_to_json_dict(result['t_base_cam']),
            't_cam_base': _transform_to_json_dict(result['t_cam_base']),
            't_gripper_target': _transform_to_json_dict(result['t_gripper_target']),
            'residual_mean_translation_mm': float(result['residual_mean_translation_mm']),
            'residual_max_translation_mm': float(result['residual_max_translation_mm']),
            'residual_mean_rotation_deg': float(result['residual_mean_rotation_deg']),
            'residual_max_rotation_deg': float(result['residual_max_rotation_deg']),
            'mount_mean_translation_mm': float(result['mount_mean_translation_mm']),
            'mount_max_translation_mm': float(result['mount_max_translation_mm']),
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
        quaternion = _rotation_matrix_to_quaternion(transform[:3, :3])
        return (
            'ros2 run tf2_ros static_transform_publisher '
            f'{translation[0]:.9f} {translation[1]:.9f} {translation[2]:.9f} '
            f'{quaternion[0]:.9f} {quaternion[1]:.9f} {quaternion[2]:.9f} {quaternion[3]:.9f} '
            f'{parent_frame} {child_frame}'
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
    node: EyeToHandCalibrationNode | None = None
    try:
        node = EyeToHandCalibrationNode()
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
