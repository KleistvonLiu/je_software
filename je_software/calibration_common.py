"""Shared helpers for eye-to-hand and eye-in-hand calibration workflows."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


METHOD_NAME_TO_FLAG = {
    'tsai': cv2.CALIB_HAND_EYE_TSAI,
    'park': cv2.CALIB_HAND_EYE_PARK,
    'horaud': cv2.CALIB_HAND_EYE_HORAUD,
    'andreff': cv2.CALIB_HAND_EYE_ANDREFF,
    'daniilidis': cv2.CALIB_HAND_EYE_DANIILIDIS,
}

REGION_BUCKETS = (
    'top_left',
    'top_center',
    'top_right',
    'center_left',
    'center',
    'center_right',
    'bottom_left',
    'bottom_center',
    'bottom_right',
)


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
class BoardPoint:
    name: str
    point_t: np.ndarray
    kind: str
    source_id: int | None = None


@dataclass
class EyeInHandCalibrationResult:
    calibration_path: Path
    pose_frame_id: str
    gripper_frame: str
    camera_frame_id: str
    board_config: dict[str, Any]
    t_gripper_cam: np.ndarray
    t_cam_gripper: np.ndarray
    raw: dict[str, Any]


def stamp_to_seconds(stamp: Any) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(max_value, value))


def make_transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    transform[:3, 3] = np.asarray(translation, dtype=np.float64).reshape(3)
    return transform


def invert_transform(transform: np.ndarray) -> np.ndarray:
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    inverse = np.eye(4, dtype=np.float64)
    inverse[:3, :3] = rotation.T
    inverse[:3, 3] = -rotation.T @ translation
    return inverse


def orthonormalize_rotation(rotation: np.ndarray) -> np.ndarray:
    u_matrix, _, v_transpose = np.linalg.svd(rotation)
    normalized = u_matrix @ v_transpose
    if np.linalg.det(normalized) < 0.0:
        u_matrix[:, -1] *= -1.0
        normalized = u_matrix @ v_transpose
    return normalized


def rotation_error_deg(rotation_a: np.ndarray, rotation_b: np.ndarray) -> float:
    delta = rotation_a.T @ rotation_b
    trace = float(np.trace(delta))
    cos_theta = clamp((trace - 1.0) * 0.5, -1.0, 1.0)
    return math.degrees(math.acos(cos_theta))


def average_rotations(rotations: list[np.ndarray]) -> np.ndarray:
    if not rotations:
        raise ValueError('No rotations to average.')
    accumulator = np.zeros((3, 3), dtype=np.float64)
    for rotation in rotations:
        accumulator += np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    return orthonormalize_rotation(accumulator)


def average_transforms(transforms: list[np.ndarray]) -> np.ndarray:
    if not transforms:
        raise ValueError('No transforms to average.')
    mean_rotation = average_rotations([transform[:3, :3] for transform in transforms])
    mean_translation = np.mean(
        [transform[:3, 3] for transform in transforms],
        axis=0,
    )
    return make_transform(mean_rotation, mean_translation)


def quaternion_to_rotation_matrix(quaternion_xyzw: np.ndarray) -> np.ndarray:
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


def rotation_matrix_to_quaternion(rotation: np.ndarray) -> np.ndarray:
    matrix = orthonormalize_rotation(rotation)
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


def rotation_matrix_to_rpy_deg(rotation: np.ndarray) -> np.ndarray:
    matrix = orthonormalize_rotation(rotation)
    sy = math.sqrt(matrix[0, 0] * matrix[0, 0] + matrix[1, 0] * matrix[1, 0])
    singular = sy < 1e-9
    if not singular:
        roll = math.atan2(matrix[2, 1], matrix[2, 2])
        pitch = math.atan2(-matrix[2, 0], sy)
        yaw = math.atan2(matrix[1, 0], matrix[0, 0])
    else:
        roll = math.atan2(-matrix[1, 2], matrix[1, 1])
        pitch = math.atan2(-matrix[2, 0], sy)
        yaw = 0.0
    return np.asarray(
        [math.degrees(roll), math.degrees(pitch), math.degrees(yaw)],
        dtype=np.float64,
    )


def pose_to_transform(pose: Any) -> np.ndarray:
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
    rotation = quaternion_to_rotation_matrix(quaternion)
    translation = np.asarray(
        [
            float(pose.position.x),
            float(pose.position.y),
            float(pose.position.z),
        ],
        dtype=np.float64,
    )
    return make_transform(rotation, translation)


def transform_to_json_dict(transform: np.ndarray) -> dict[str, Any]:
    matrix = np.asarray(transform, dtype=np.float64).reshape(4, 4)
    return {
        'translation_m': [float(value) for value in matrix[:3, 3]],
        'quaternion_xyzw': [
            float(value)
            for value in rotation_matrix_to_quaternion(matrix[:3, :3])
        ],
        'matrix': [
            [float(value) for value in row]
            for row in matrix.tolist()
        ],
    }


def transform_from_json_dict(payload: dict[str, Any]) -> np.ndarray:
    if 'matrix' in payload:
        matrix = np.asarray(payload['matrix'], dtype=np.float64)
        if matrix.shape != (4, 4):
            raise ValueError('Transform matrix must be 4x4.')
        return matrix
    if 'translation_m' not in payload or 'quaternion_xyzw' not in payload:
        raise KeyError('Transform JSON must contain matrix or translation_m + quaternion_xyzw.')
    translation = np.asarray(payload['translation_m'], dtype=np.float64).reshape(3)
    quaternion = np.asarray(payload['quaternion_xyzw'], dtype=np.float64).reshape(4)
    rotation = quaternion_to_rotation_matrix(quaternion / np.linalg.norm(quaternion))
    return make_transform(rotation, translation)


def transform_point(transform: np.ndarray, point_xyz: np.ndarray) -> np.ndarray:
    point = np.asarray(point_xyz, dtype=np.float64).reshape(3)
    return transform[:3, :3] @ point + transform[:3, 3]


def vector_norm_mm(vector: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(vector, dtype=np.float64).reshape(-1)) * 1000.0)


def image_msg_to_bgr(msg: Any) -> np.ndarray:
    encoding = str(msg.encoding).lower()
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


def opencv_supports_charuco() -> bool:
    if not hasattr(cv2, 'aruco'):
        return False
    aruco_module = cv2.aruco
    has_board = hasattr(aruco_module, 'CharucoBoard_create') or hasattr(aruco_module, 'CharucoBoard')
    has_dictionary = hasattr(aruco_module, 'getPredefinedDictionary')
    has_detector = hasattr(aruco_module, 'detectMarkers') or hasattr(aruco_module, 'ArucoDetector')
    has_interpolation = hasattr(aruco_module, 'interpolateCornersCharuco')
    return has_board and has_dictionary and has_detector and has_interpolation


def config_length_to_meters(
    config: dict[str, Any],
    meters_key: str,
    millimeters_key: str,
) -> float:
    if meters_key in config:
        return float(config[meters_key])
    if millimeters_key in config:
        return float(config[millimeters_key]) / 1000.0
    raise KeyError(f'Missing board length field: {meters_key} / {millimeters_key}')


def load_charuco_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f'ChArUco config not found: {path}')
    with path.open('r', encoding='utf-8') as file_obj:
        config = json.load(file_obj)
    board_type = str(config.get('board_type', 'charuco')).strip().lower()
    if board_type != 'charuco':
        raise ValueError('Only ChArUco board configs are supported.')
    return {
        'type': 'charuco',
        'squares_x': int(config['squares_x']),
        'squares_y': int(config['squares_y']),
        'square_length_m': config_length_to_meters(
            config=config,
            meters_key='square_length_m',
            millimeters_key='square_length_mm',
        ),
        'marker_length_m': config_length_to_meters(
            config=config,
            meters_key='marker_length_m',
            millimeters_key='marker_length_mm',
        ),
        'aruco_dictionary': str(config['dictionary']).strip(),
        'charuco_config_path': str(path),
    }


def region_bucket_indices(center_x_norm: float, center_y_norm: float) -> tuple[int, int]:
    x_value = clamp(center_x_norm, 0.0, 0.999999)
    y_value = clamp(center_y_norm, 0.0, 0.999999)
    col_index = min(2, int(x_value * 3.0))
    row_index = min(2, int(y_value * 3.0))
    return row_index, col_index


def region_bucket_name(center_x_norm: float, center_y_norm: float) -> str:
    row_index, col_index = region_bucket_indices(center_x_norm, center_y_norm)
    return REGION_BUCKETS[row_index * 3 + col_index]


def draw_detection_overlay(image: np.ndarray, detection: CharucoDetection) -> None:
    for marker in detection.marker_corners:
        corners = marker.reshape(-1, 2).astype(np.int32)
        cv2.polylines(image, [corners], True, (255, 0, 255), 2, cv2.LINE_AA)

    if detection.charuco_corners is None:
        return
    for point in detection.charuco_corners.reshape(-1, 2):
        center = (int(round(point[0])), int(round(point[1])))
        cv2.circle(image, center, 4, (0, 255, 0), -1)


class CharucoBoardHelper:
    def __init__(
        self,
        *,
        squares_x: int,
        squares_y: int,
        square_length_m: float,
        marker_length_m: float,
        aruco_dictionary_name: str,
        charuco_config_path: str = '',
    ) -> None:
        if not opencv_supports_charuco():
            raise RuntimeError('Current OpenCV build does not expose cv2.aruco Charuco APIs.')

        self.squares_x = int(squares_x)
        self.squares_y = int(squares_y)
        self.square_length_m = float(square_length_m)
        self.marker_length_m = float(marker_length_m)
        self.aruco_dictionary_name = str(aruco_dictionary_name).strip()
        self.charuco_config_path = str(charuco_config_path).strip()

        self.aruco_dictionary_id = self._resolve_aruco_dictionary_id(self.aruco_dictionary_name)
        self.aruco_dictionary = cv2.aruco.getPredefinedDictionary(self.aruco_dictionary_id)
        self.aruco_detector_params = self._create_aruco_detector_params()
        self.aruco_detector = self._create_aruco_detector()
        self.charuco_board = self._create_charuco_board()
        self.charuco_corner_object_points = self._get_charuco_board_corners()

        self.board_width_m = float(self.squares_x) * self.square_length_m
        self.board_height_m = float(self.squares_y) * self.square_length_m

    @classmethod
    def from_board_config(cls, board_config: dict[str, Any]) -> CharucoBoardHelper:
        board_type = str(board_config.get('type', 'charuco')).strip().lower()
        if board_type != 'charuco':
            raise ValueError('Only ChArUco board configs are supported.')
        dictionary_name = str(
            board_config.get('aruco_dictionary', board_config.get('dictionary', ''))
        ).strip()
        return cls(
            squares_x=int(board_config['squares_x']),
            squares_y=int(board_config['squares_y']),
            square_length_m=float(board_config['square_length_m']),
            marker_length_m=float(board_config['marker_length_m']),
            aruco_dictionary_name=dictionary_name,
            charuco_config_path=str(board_config.get('charuco_config_path', '') or ''),
        )

    def board_config_snapshot(self) -> dict[str, Any]:
        return {
            'type': 'charuco',
            'squares_x': int(self.squares_x),
            'squares_y': int(self.squares_y),
            'square_length_m': float(self.square_length_m),
            'marker_length_m': float(self.marker_length_m),
            'aruco_dictionary': self.aruco_dictionary_name,
            'charuco_config_path': self.charuco_config_path or None,
        }

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

    def charuco_object_points_from_ids(self, charuco_ids: np.ndarray) -> np.ndarray:
        ids = np.asarray(charuco_ids, dtype=np.int32).reshape(-1)
        return self.charuco_corner_object_points[ids]

    def get_charuco_corner_by_id(self, charuco_id: int) -> np.ndarray:
        return np.asarray(
            self.charuco_corner_object_points[int(charuco_id)],
            dtype=np.float64,
        ).reshape(3)

    def detect(
        self,
        gray_image: np.ndarray,
        camera_matrix: np.ndarray | None = None,
        dist_coeffs: np.ndarray | None = None,
    ) -> CharucoDetection:
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
        if camera_matrix is not None and dist_coeffs is not None:
            interpolate_kwargs['cameraMatrix'] = camera_matrix
            interpolate_kwargs['distCoeffs'] = dist_coeffs

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

    def solve_target_pose(
        self,
        *,
        charuco_corners: np.ndarray,
        charuco_ids: np.ndarray,
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, float] | None:
        object_points = self.charuco_object_points_from_ids(charuco_ids)
        image_points = np.asarray(charuco_corners, dtype=np.float32).reshape(-1, 2)
        if object_points.shape[0] < 4:
            return None

        success, rvec, tvec = cv2.solvePnP(
            object_points,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not success:
            return None

        projected, _ = cv2.projectPoints(
            object_points,
            rvec,
            tvec,
            camera_matrix,
            dist_coeffs,
        )
        projected = projected.reshape(-1, 2)
        reprojection = float(np.mean(np.linalg.norm(projected - image_points, axis=1)))
        return rvec.reshape(3, 1), tvec.reshape(3, 1), reprojection

    def board_outer_corners(self) -> dict[str, np.ndarray]:
        width = self.board_width_m
        height = self.board_height_m
        return {
            'board_corner_top_left': np.asarray([0.0, 0.0, 0.0], dtype=np.float64),
            'board_corner_top_right': np.asarray([width, 0.0, 0.0], dtype=np.float64),
            'board_corner_bottom_right': np.asarray([width, height, 0.0], dtype=np.float64),
            'board_corner_bottom_left': np.asarray([0.0, height, 0.0], dtype=np.float64),
        }

    def default_point_set(
        self,
        *,
        point_set: str,
        extra_charuco_ids: list[int] | None = None,
    ) -> list[BoardPoint]:
        if point_set != 'center_corners':
            raise ValueError("Only point_set='center_corners' is currently supported.")

        points = [
            BoardPoint(
                name='board_center',
                point_t=np.asarray(
                    [self.board_width_m * 0.5, self.board_height_m * 0.5, 0.0],
                    dtype=np.float64,
                ),
                kind='board_center',
            ),
        ]
        for name, point in self.board_outer_corners().items():
            points.append(
                BoardPoint(
                    name=name,
                    point_t=np.asarray(point, dtype=np.float64).reshape(3),
                    kind='board_corner',
                )
            )

        if extra_charuco_ids:
            for charuco_id in sorted(set(int(value) for value in extra_charuco_ids)):
                points.append(
                    BoardPoint(
                        name=f'charuco_id_{charuco_id:03d}',
                        point_t=self.get_charuco_corner_by_id(charuco_id),
                        kind='charuco_corner',
                        source_id=int(charuco_id),
                    )
                )
        return points

    def project_points(
        self,
        object_points_t: np.ndarray,
        rvec: np.ndarray,
        tvec: np.ndarray,
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
    ) -> np.ndarray:
        object_points = np.asarray(object_points_t, dtype=np.float64).reshape(-1, 3)
        projected, _ = cv2.projectPoints(
            object_points,
            rvec,
            tvec,
            camera_matrix,
            dist_coeffs,
        )
        return projected.reshape(-1, 2)


def draw_projected_board_point(
    image: np.ndarray,
    *,
    board_helper: CharucoBoardHelper,
    point_t: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    color: tuple[int, int, int] = (0, 215, 255),
    label: str | None = None,
) -> tuple[int, int]:
    projected = board_helper.project_points(
        object_points_t=np.asarray(point_t, dtype=np.float64).reshape(1, 3),
        rvec=rvec,
        tvec=tvec,
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
    )[0]
    center = (int(round(projected[0])), int(round(projected[1])))
    cv2.circle(image, center, 8, color, 2, cv2.LINE_AA)
    cv2.drawMarker(image, center, color, cv2.MARKER_CROSS, 18, 2, cv2.LINE_AA)
    if label:
        cv2.putText(
            image,
            label,
            (center[0] + 10, center[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            label,
            (center[0] + 10, center[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            1,
            cv2.LINE_AA,
        )
    return center


def load_eye_in_hand_calibration_result(
    calibration_result_path: str | Path,
) -> EyeInHandCalibrationResult:
    path = Path(calibration_result_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f'Calibration result not found: {path}')

    with path.open('r', encoding='utf-8') as file_obj:
        payload = json.load(file_obj)

    best_result = payload.get('best_result')
    if not isinstance(best_result, dict):
        raise ValueError('Calibration result missing best_result.')
    if 't_gripper_cam' not in best_result:
        raise ValueError('Calibration result missing best_result.t_gripper_cam.')

    board_config = payload.get('board')
    if not isinstance(board_config, dict):
        raise ValueError('Calibration result missing board config.')

    pose_frame_id = str(payload.get('pose_frame_id', '')).strip()
    gripper_frame = str(payload.get('gripper_frame', '')).strip()
    camera_frame_id = str(payload.get('camera_frame_id', '')).strip()
    if not pose_frame_id:
        raise ValueError('Calibration result missing pose_frame_id.')
    if not gripper_frame:
        raise ValueError('Calibration result missing gripper_frame.')
    if not camera_frame_id:
        raise ValueError('Calibration result missing camera_frame_id.')

    t_gripper_cam = transform_from_json_dict(best_result['t_gripper_cam'])

    return EyeInHandCalibrationResult(
        calibration_path=path,
        pose_frame_id=pose_frame_id,
        gripper_frame=gripper_frame,
        camera_frame_id=camera_frame_id,
        board_config=board_config,
        t_gripper_cam=t_gripper_cam,
        t_cam_gripper=invert_transform(t_gripper_cam),
        raw=payload,
    )
