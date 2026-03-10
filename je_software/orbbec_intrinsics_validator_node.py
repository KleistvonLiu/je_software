#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image


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

REGION_TARGETS = {
    'top_left': 2,
    'top_center': 2,
    'top_right': 2,
    'center_left': 2,
    'center': 4,
    'center_right': 2,
    'bottom_left': 2,
    'bottom_center': 2,
    'bottom_right': 2,
}

DISTANCE_BUCKETS = ('near', 'mid', 'far')
TILT_BUCKETS = ('low', 'mid', 'high')


@dataclass
class CandidateFrame:
    frame_index: int
    stamp_s: float
    region_bucket: str
    region_class: str
    z_m: float
    distance_m: float
    tilt_deg: float
    board_center_x_px: float
    board_center_y_px: float
    center_x_norm: float
    center_y_norm: float
    board_area_ratio: float
    corner_count: int
    reproj_mean_px: float
    reproj_max_px: float
    focus_score: float
    rvec: np.ndarray
    tvec: np.ndarray
    corners_px: np.ndarray
    projected_px: np.ndarray
    image_width: int
    image_height: int
    jpeg_bytes: bytes
    distance_bucket: str = 'mid'
    tilt_bucket: str = 'mid'
    selected_rank: int | None = None
    saved_image_path: str = ''


@dataclass
class CharucoDetection:
    marker_corners: list[np.ndarray]
    marker_ids: np.ndarray | None
    charuco_corners: np.ndarray | None
    charuco_ids: np.ndarray | None
    charuco_count: int = 0


def _stamp_to_seconds(stamp: Any) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def _clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(max_value, value))


def _rotation_matrix_from_rvec(rvec: np.ndarray) -> np.ndarray:
    matrix, _ = cv2.Rodrigues(rvec)
    return matrix


def _rotation_delta_deg(rvec_a: np.ndarray, rvec_b: np.ndarray) -> float:
    rot_a = _rotation_matrix_from_rvec(rvec_a)
    rot_b = _rotation_matrix_from_rvec(rvec_b)
    delta = rot_a.T @ rot_b
    trace = float(np.trace(delta))
    cos_theta = _clamp((trace - 1.0) * 0.5, -1.0, 1.0)
    return math.degrees(math.acos(cos_theta))


def _vector_norm_mm(vector: np.ndarray) -> float:
    return float(np.linalg.norm(vector) * 1000.0)


class OrbbecIntrinsicsValidator(Node):
    def __init__(self) -> None:
        super().__init__('orbbec_intrinsics_validator')

        self._declare_parameters()
        self.image_topic = self.get_parameter('image_topic').value
        self.camera_info_topic = self.get_parameter('camera_info_topic').value
        self.output_root = Path(self.get_parameter('output_dir').value).expanduser()
        self.capture_duration_s = float(self.get_parameter('capture_duration_s').value)
        self.process_interval_s = float(self.get_parameter('process_interval_s').value)
        self.target_selected_count = int(self.get_parameter('target_selected_count').value)
        self.max_selected_count = int(self.get_parameter('max_selected_count').value)
        self.show_preview = bool(self.get_parameter('show_preview').value)
        self.preview_max_width = int(self.get_parameter('preview_max_width').value)
        self.charuco_config_path = str(self.get_parameter('charuco_config_path').value).strip()
        self.squares_x = int(self.get_parameter('squares_x').value)
        self.squares_y = int(self.get_parameter('squares_y').value)
        self.square_length_m = float(self.get_parameter('square_length_m').value)
        self.marker_length_m = float(self.get_parameter('marker_length_m').value)
        self.aruco_dictionary_name = str(self.get_parameter('aruco_dictionary').value).strip()
        self.min_charuco_corners = int(self.get_parameter('min_charuco_corners').value)
        self.max_reproj_for_candidate_px = float(self.get_parameter('max_reproj_for_candidate_px').value)
        self.stability_min_frames = int(self.get_parameter('stability_min_frames').value)
        self.stability_max_gap_s = float(self.get_parameter('stability_max_gap_s').value)
        self.stability_translation_gate_m = float(self.get_parameter('stability_translation_gate_m').value)
        self.stability_rotation_gate_deg = float(self.get_parameter('stability_rotation_gate_deg').value)
        self.charuco_config_snapshot: dict[str, Any] = {}
        self._load_charuco_config_if_needed()
        self._validate_parameters()
        self.aruco_dictionary_id = self._resolve_aruco_dictionary_id(self.aruco_dictionary_name)
        self.aruco_dictionary = cv2.aruco.getPredefinedDictionary(self.aruco_dictionary_id)
        self.aruco_detector_params = self._create_aruco_detector_params()
        self.aruco_detector = self._create_aruco_detector()
        self.charuco_board = self._create_charuco_board()
        self.charuco_corner_object_points = self._get_charuco_board_corners()

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.session_dir = self.output_root / f'orbbec_intrinsics_validation_{timestamp}'
        self.images_dir = self.session_dir / 'images'
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.camera_matrix: np.ndarray | None = None
        self.dist_coeffs: np.ndarray | None = None
        self.camera_info_snapshot: dict[str, Any] = {}
        self.latest_info_stamp_s: float | None = None

        self.start_monotonic_s = time.monotonic()
        self.last_process_monotonic_s = 0.0
        self.last_progress_log_s = 0.0
        self.last_camera_info_wait_log_s = 0.0
        self.camera_info_logged = False
        self.preview_available = self.show_preview
        self.preview_window_name = 'Orbbec Intrinsics Validator'
        self.total_image_msgs = 0
        self.candidate_frames: list[CandidateFrame] = []
        self.finalized = False
        self.finalize_reason = ''

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
        self.status_timer = self.create_timer(0.5, self._status_timer_callback)

        self.get_logger().info(
            '开始 RGB 内参验证：'
            f'图像话题={self.image_topic}，'
            f'相机内参话题={self.camera_info_topic}，'
            f'标定板=ChArUco {self.squares_x}x{self.squares_y}，'
            f'方格边长={self.square_length_m:.4f}m，'
            f'marker 边长={self.marker_length_m:.4f}m，'
            f'字典={self.aruco_dictionary_name}，'
            f'采集时长={self.capture_duration_s:.1f}s，'
            f'实时预览={"开启" if self.show_preview else "关闭"}，'
            f'输出目录={self.session_dir}'
        )
        if self.charuco_config_path:
            self.get_logger().info(f'已加载 ChArUco 标定板配置：{self.charuco_config_path}')
        self.get_logger().info(
            '本工具只验证视觉链路和 T_cam_board 稳定性，不混入 hand-eye 标定结果。'
        )

    def _declare_parameters(self) -> None:
        # 必填/最常修改参数：
        # `image_topic`：RGB 图像话题，脚本从这里读取待验证图像。
        self.declare_parameter('image_topic', '/camera/color/image_raw')
        # `camera_info_topic`：相机内参话题，脚本从这里读取当前 SDK/驱动发布的内参。
        self.declare_parameter('camera_info_topic', '/camera/color/camera_info')
        # `output_dir`：验证结果输出根目录，最终会在其下新建时间戳子目录。
        self.declare_parameter(
            'output_dir',
            str(Path.home() / 'orbbec_intrinsics_validation'),
        )
        # `capture_duration_s`：总采集时长，建议给操作者留够时间覆盖中心/四角/边缘和近中远距离。
        self.declare_parameter('capture_duration_s', 4500.0)

        # 运行策略参数：
        # `process_interval_s`：处理图像的最小时间间隔，用于降采样，避免每帧都做检测导致负载过高。
        self.declare_parameter('process_interval_s', 0.2)
        # `target_selected_count`：目标输出验证图数量，通常设为 20~30 张中的一个值。
        self.declare_parameter('target_selected_count', 24)
        # `max_selected_count`：允许输出的最大验证图数量上限。
        self.declare_parameter('max_selected_count', 30)
        # `show_preview`：是否实时显示图像和检测结果预览窗口。
        self.declare_parameter('show_preview', True)
        # `preview_max_width`：预览窗口最大宽度，原图更宽时会按比例缩放显示。
        self.declare_parameter('preview_max_width', 1280)

        # ChArUco 板参数：
        # `charuco_config_path`：可选，指向生成板子的 JSON；如果提供，会用 JSON 中的配置覆盖下面的手填参数。
        self.declare_parameter('charuco_config_path', '')
        # `squares_x` / `squares_y`：ChArUco 棋盘格方块数量，不是内角点数量。
        self.declare_parameter('squares_x', 7)
        self.declare_parameter('squares_y', 5)
        # `square_length_m`：大方格边长，单位米，例如 50mm 就填 0.05。
        self.declare_parameter('square_length_m', 0.05)
        # `marker_length_m`：Aruco marker 边长，单位米，例如 30mm 就填 0.03。
        self.declare_parameter('marker_length_m', 0.03)
        # `aruco_dictionary`：Aruco 字典名，需与生成板子时保持一致，例如 `DICT_4X4_50`。
        self.declare_parameter('aruco_dictionary', 'DICT_4X4_50')
        # `min_charuco_corners`：一帧至少插值出多少个 Charuco 角点才认为是有效样本。
        self.declare_parameter('min_charuco_corners', 8)

        # 质量门限参数：
        # `max_reproj_for_candidate_px`：单帧平均重投影误差超过该值时，不纳入有效样本。
        self.declare_parameter('max_reproj_for_candidate_px', 3.0)
        # `stability_min_frames`：静止姿态分析至少需要连续多少帧有效样本。
        self.declare_parameter('stability_min_frames', 5)
        # `stability_max_gap_s`：判定“同一静止姿态窗口”时，相邻帧允许的最大时间间隔。
        self.declare_parameter('stability_max_gap_s', 0.5)
        # `stability_translation_gate_m`：静止窗口内，相邻帧平移变化的门限，单位米。
        self.declare_parameter('stability_translation_gate_m', 0.015)
        # `stability_rotation_gate_deg`：静止窗口内，相邻帧旋转变化的门限，单位度。
        self.declare_parameter('stability_rotation_gate_deg', 1.0)

    def _validate_parameters(self) -> None:
        if not self._opencv_supports_charuco():
            raise RuntimeError(
                '当前 OpenCV 构建不支持 ChArUco 标定板 '
                f'(python={sys.executable}, cv2={getattr(cv2, "__version__", "unknown")})。'
                '请使用 ROS 系统 Python，或安装带 aruco/charuco 支持的 OpenCV。'
            )
        if self.squares_x <= 1 or self.squares_y <= 1:
            raise ValueError('squares_x 和 squares_y 都必须大于 1。')
        if self.square_length_m <= 0.0:
            raise ValueError('square_length_m 必须为正数。')
        if self.marker_length_m <= 0.0:
            raise ValueError('marker_length_m 必须为正数。')
        if self.marker_length_m >= self.square_length_m:
            raise ValueError('marker_length_m 必须小于 square_length_m。')
        if self.min_charuco_corners < 4:
            raise ValueError('min_charuco_corners 至少应为 4。')
        if self.capture_duration_s <= 0.0:
            raise ValueError('capture_duration_s 必须为正数。')
        if self.process_interval_s <= 0.0:
            raise ValueError('process_interval_s 必须为正数。')
        if self.target_selected_count <= 0 or self.max_selected_count <= 0:
            raise ValueError('target_selected_count 和 max_selected_count 都必须为正数。')
        if self.target_selected_count > self.max_selected_count:
            raise ValueError('target_selected_count 不能大于 max_selected_count。')
        if self.max_reproj_for_candidate_px <= 0.0:
            raise ValueError('max_reproj_for_candidate_px 必须为正数。')

    def _opencv_supports_charuco(self) -> bool:
        if not hasattr(cv2, 'aruco'):
            return False

        aruco_module = cv2.aruco
        has_board = hasattr(aruco_module, 'CharucoBoard_create') or hasattr(aruco_module, 'CharucoBoard')
        has_dictionary = hasattr(aruco_module, 'getPredefinedDictionary')
        has_detector = hasattr(aruco_module, 'detectMarkers') or hasattr(aruco_module, 'ArucoDetector')
        has_interpolation = hasattr(aruco_module, 'interpolateCornersCharuco')
        return has_board and has_dictionary and has_detector and has_interpolation

    def _create_aruco_detector_params(self) -> Any:
        aruco_module = cv2.aruco
        if hasattr(aruco_module, 'DetectorParameters_create'):
            return aruco_module.DetectorParameters_create()
        if hasattr(aruco_module, 'DetectorParameters'):
            return aruco_module.DetectorParameters()
        raise RuntimeError('当前 OpenCV 构建未暴露 ArUco 检测参数接口。')

    def _create_aruco_detector(self) -> Any | None:
        if hasattr(cv2.aruco, 'ArucoDetector'):
            return cv2.aruco.ArucoDetector(self.aruco_dictionary, self.aruco_detector_params)
        return None

    def _create_charuco_board(self) -> Any:
        aruco_module = cv2.aruco
        if hasattr(aruco_module, 'CharucoBoard_create'):
            return aruco_module.CharucoBoard_create(
                self.squares_x,
                self.squares_y,
                self.square_length_m,
                self.marker_length_m,
                self.aruco_dictionary,
            )
        if hasattr(aruco_module, 'CharucoBoard'):
            try:
                return aruco_module.CharucoBoard(
                    (self.squares_x, self.squares_y),
                    self.square_length_m,
                    self.marker_length_m,
                    self.aruco_dictionary,
                )
            except TypeError:
                return aruco_module.CharucoBoard(
                    self.squares_x,
                    self.squares_y,
                    self.square_length_m,
                    self.marker_length_m,
                    self.aruco_dictionary,
                )
        raise RuntimeError('当前 OpenCV 构建未暴露 ChArUco 标定板构造接口。')

    def _get_charuco_board_corners(self) -> np.ndarray:
        board_corners = getattr(self.charuco_board, 'chessboardCorners', None)
        if board_corners is None and hasattr(self.charuco_board, 'getChessboardCorners'):
            board_corners = self.charuco_board.getChessboardCorners()
        if board_corners is None:
            raise RuntimeError('无法从 OpenCV 中读取 ChArUco 标定板角点坐标。')
        return np.asarray(board_corners, dtype=np.float32)

    def _load_charuco_config_if_needed(self) -> None:
        if not self.charuco_config_path:
            return

        config_path = Path(self.charuco_config_path).expanduser()
        if not config_path.is_file():
            raise FileNotFoundError(f'未找到 ChArUco 配置文件：{config_path}')

        with config_path.open('r', encoding='utf-8') as file_obj:
            config = json.load(file_obj)

        board_type = str(config.get('board_type', '')).strip().lower()
        if board_type and board_type != 'charuco':
            raise ValueError(f'配置文件中的 board_type 不受支持：{config.get("board_type")}')

        try:
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
        except KeyError as exc:
            raise KeyError(f'ChArUco 配置缺少字段：{exc.args[0]}') from exc

        self.charuco_config_path = str(config_path)
        self.charuco_config_snapshot = dict(config)
        self.charuco_config_snapshot['config_path'] = str(config_path)

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
        raise KeyError(f'缺少长度字段：{meters_key} / {millimeters_key}')

    def _resolve_aruco_dictionary_id(self, dictionary_name: str) -> int:
        if not hasattr(cv2.aruco, dictionary_name):
            raise ValueError(f'不支持的 ArUco 字典：{dictionary_name}')
        return int(getattr(cv2.aruco, dictionary_name))

    def _charuco_object_points_from_ids(self, charuco_ids: np.ndarray) -> np.ndarray:
        ids = charuco_ids.reshape(-1).astype(np.int32)
        return self.charuco_corner_object_points[ids]

    def _camera_info_callback(self, msg: CameraInfo) -> None:
        if len(msg.k) != 9:
            self.get_logger().warn('收到的 CameraInfo.k 长度不是 9，已忽略该消息。')
            return

        self.camera_matrix = np.asarray(msg.k, dtype=np.float64).reshape(3, 3)
        self.dist_coeffs = np.asarray(msg.d, dtype=np.float64).reshape(-1, 1)
        self.latest_info_stamp_s = _stamp_to_seconds(msg.header.stamp)
        self.camera_info_snapshot = {
            'topic': self.camera_info_topic,
            'frame_id': msg.header.frame_id,
            'width': int(msg.width),
            'height': int(msg.height),
            'distortion_model': msg.distortion_model,
            'k': list(msg.k),
            'd': list(msg.d),
        }
        if not self.camera_info_logged:
            self.camera_info_logged = True
            self.get_logger().info(
                '已收到 CameraInfo：'
                f'frame_id={msg.header.frame_id}，分辨率={msg.width}x{msg.height}，'
                f'畸变模型={msg.distortion_model}'
            )

    def _status_timer_callback(self) -> None:
        if self.finalized:
            return

        elapsed_s = time.monotonic() - self.start_monotonic_s
        if self.camera_matrix is None:
            if elapsed_s - self.last_camera_info_wait_log_s >= 2.0:
                self.last_camera_info_wait_log_s = elapsed_s
                self.get_logger().info(
                    f'等待 {self.camera_info_topic} 发布 CameraInfo，'
                    f'已等待 {elapsed_s:.1f}s'
                )
            if elapsed_s >= self.capture_duration_s:
                self.finalize('timeout_without_camera_info')
                if rclpy.ok():
                    rclpy.shutdown()
            return

        if elapsed_s >= self.capture_duration_s:
            self.finalize('timeout')
            if rclpy.ok():
                rclpy.shutdown()

    def _image_callback(self, msg: Image) -> None:
        if self.finalized:
            return

        self.total_image_msgs += 1

        now_s = time.monotonic()
        if now_s - self.last_process_monotonic_s < self.process_interval_s:
            return
        self.last_process_monotonic_s = now_s

        try:
            bgr_image = self._image_msg_to_bgr(msg)
        except ValueError as exc:
            self.get_logger().warn(f'暂不支持的图像编码：{exc}')
            return

        gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        if self.camera_matrix is None or self.dist_coeffs is None:
            self._show_preview(
                self._build_preview_image(
                    bgr_image=bgr_image,
                    detection=None,
                    observation=None,
                    status_text='waiting_camera_info',
                )
            )
            self._maybe_log_progress(now_s)
            return

        detection = self._detect_charuco(gray_image)
        observation: CandidateFrame | None = None
        preview_status = self._preview_status_from_detection(detection)
        if detection.charuco_corners is not None and detection.charuco_ids is not None:
            observation, preview_status = self._analyze_detection(
                msg=msg,
                bgr_image=bgr_image,
                gray_image=gray_image,
                charuco_corners=detection.charuco_corners,
                charuco_ids=detection.charuco_ids,
            )
            if observation is not None:
                self.candidate_frames.append(observation)

        self._show_preview(
            self._build_preview_image(
                bgr_image=bgr_image,
                detection=detection,
                observation=observation,
                status_text=preview_status,
            )
        )
        if self.finalized:
            return
        self._maybe_log_progress(now_s)

    def _maybe_log_progress(self, now_s: float) -> None:
        if self.camera_matrix is None or self.dist_coeffs is None:
            return
        if now_s - self.last_progress_log_s < 2.0:
            return
        self.last_progress_log_s = now_s

        if not self.candidate_frames:
            self.get_logger().info(
                f'已处理图像消息={self.total_image_msgs}，有效检测=0。'
                '请继续移动 ChArUco 板，覆盖中心/边缘/角落以及近/中/远距离。'
            )
            return

        z_values = np.asarray([frame.z_m for frame in self.candidate_frames], dtype=np.float64)
        tilt_values = np.asarray([frame.tilt_deg for frame in self.candidate_frames], dtype=np.float64)
        region_counts = Counter(frame.region_bucket for frame in self.candidate_frames)
        covered_regions = sum(1 for name in REGION_BUCKETS if region_counts.get(name, 0) > 0)
        self.get_logger().info(
            f'已处理图像消息={self.total_image_msgs}，'
            f'有效检测={len(self.candidate_frames)}，'
            f'区域覆盖={covered_regions}/9，'
            f'z 范围=[{z_values.min():.3f}, {z_values.max():.3f}]m，'
            f'倾角范围=[{tilt_values.min():.1f}, {tilt_values.max():.1f}]deg'
        )

    def _image_msg_to_bgr(self, msg: Image) -> np.ndarray:
        encoding = msg.encoding.lower()
        row_data = np.frombuffer(msg.data, dtype=np.uint8)

        if encoding == 'bgr8':
            array = row_data.reshape(msg.height, msg.step)[:, : msg.width * 3].reshape(msg.height, msg.width, 3)
            return array.copy()
        if encoding == 'rgb8':
            array = row_data.reshape(msg.height, msg.step)[:, : msg.width * 3].reshape(msg.height, msg.width, 3)
            return cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
        if encoding in ('mono8', '8uc1'):
            array = row_data.reshape(msg.height, msg.step)[:, : msg.width]
            return cv2.cvtColor(array, cv2.COLOR_GRAY2BGR)
        if encoding == 'bgra8':
            array = row_data.reshape(msg.height, msg.step)[:, : msg.width * 4].reshape(msg.height, msg.width, 4)
            return cv2.cvtColor(array, cv2.COLOR_BGRA2BGR)
        if encoding == 'rgba8':
            array = row_data.reshape(msg.height, msg.step)[:, : msg.width * 4].reshape(msg.height, msg.width, 4)
            return cv2.cvtColor(array, cv2.COLOR_RGBA2BGR)
        raise ValueError(f'未知编码 {encoding}')

    def _detect_charuco(
        self,
        gray_image: np.ndarray,
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
            return CharucoDetection(marker_corners=[], marker_ids=None, charuco_corners=None, charuco_ids=None)

        # OpenCV's Python binding places output arrays before camera intrinsics in the
        # positional signature. Use keywords here to avoid passing intrinsics into the
        # wrong native slots, which can segfault the interpreter on some builds.
        charuco_count, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
            marker_corners,
            marker_ids,
            gray_image,
            self.charuco_board,
            cameraMatrix=self.camera_matrix,
            distCoeffs=self.dist_coeffs,
        )
        normalized_charuco_corners = None
        normalized_charuco_ids = None
        if charuco_corners is not None and charuco_ids is not None:
            normalized_charuco_corners = charuco_corners.astype(np.float32)
            normalized_charuco_ids = charuco_ids.astype(np.int32)

        return CharucoDetection(
            marker_corners=list(marker_corners),
            marker_ids=marker_ids.astype(np.int32) if marker_ids is not None else None,
            charuco_corners=normalized_charuco_corners,
            charuco_ids=normalized_charuco_ids,
            charuco_count=int(charuco_count),
        )

    def _analyze_detection(
        self,
        msg: Image,
        bgr_image: np.ndarray,
        gray_image: np.ndarray,
        charuco_corners: np.ndarray,
        charuco_ids: np.ndarray,
    ) -> tuple[CandidateFrame | None, str]:
        assert self.camera_matrix is not None
        assert self.dist_coeffs is not None

        if int(charuco_ids.shape[0]) < self.min_charuco_corners:
            return None, f'charuco<{self.min_charuco_corners}'

        observed = charuco_corners.reshape(-1, 2)
        object_points = self._charuco_object_points_from_ids(charuco_ids)
        if object_points.shape[0] < 4:
            return None, 'objpts<4'

        success, rvec, tvec = cv2.solvePnP(
            object_points,
            observed,
            self.camera_matrix,
            self.dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not success:
            return None, 'solvepnp_failed'

        projected, _ = cv2.projectPoints(
            object_points,
            rvec,
            tvec,
            self.camera_matrix,
            self.dist_coeffs,
        )
        projected = projected.reshape(-1, 2)
        reproj_errors = np.linalg.norm(projected - observed, axis=1)
        reproj_mean_px = float(np.mean(reproj_errors))
        if reproj_mean_px > self.max_reproj_for_candidate_px:
            return None, f'reproj>{self.max_reproj_for_candidate_px:.1f}'

        reproj_max_px = float(np.max(reproj_errors))
        board_center = np.mean(observed, axis=0)
        region_bucket = self._classify_region(
            center_x_px=float(board_center[0]),
            center_y_px=float(board_center[1]),
            image_width=int(msg.width),
            image_height=int(msg.height),
        )
        region_class = self._region_bucket_to_class(region_bucket)

        contour = cv2.convexHull(observed.astype(np.float32))
        board_area_ratio = float(cv2.contourArea(contour) / max(1.0, float(msg.width * msg.height)))
        focus_score = float(cv2.Laplacian(gray_image, cv2.CV_64F).var())
        z_m = abs(float(tvec[2]))
        distance_m = float(np.linalg.norm(tvec))
        tilt_deg = self._compute_tilt_deg(rvec)
        center_x_norm = float(board_center[0] / max(1, msg.width))
        center_y_norm = float(board_center[1] / max(1, msg.height))

        ok, encoded = cv2.imencode('.jpg', bgr_image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        if not ok:
            return None, 'jpeg_failed'

        return (
            CandidateFrame(
                frame_index=len(self.candidate_frames),
                stamp_s=_stamp_to_seconds(msg.header.stamp),
                region_bucket=region_bucket,
                region_class=region_class,
                z_m=z_m,
                distance_m=distance_m,
                tilt_deg=tilt_deg,
                board_center_x_px=float(board_center[0]),
                board_center_y_px=float(board_center[1]),
                center_x_norm=center_x_norm,
                center_y_norm=center_y_norm,
                board_area_ratio=board_area_ratio,
                corner_count=int(observed.shape[0]),
                reproj_mean_px=reproj_mean_px,
                reproj_max_px=reproj_max_px,
                focus_score=focus_score,
                rvec=rvec.reshape(3, 1),
                tvec=tvec.reshape(3, 1),
                corners_px=observed,
                projected_px=projected,
                image_width=int(msg.width),
                image_height=int(msg.height),
                jpeg_bytes=encoded.tobytes(),
            ),
            'valid',
        )

    def _preview_status_from_detection(self, detection: CharucoDetection) -> str:
        if detection.marker_ids is None or len(detection.marker_ids) == 0:
            return 'no_marker'
        if detection.charuco_corners is None or detection.charuco_ids is None:
            return 'charuco_failed'
        if detection.charuco_count < self.min_charuco_corners:
            return f'charuco={detection.charuco_count}/{self.min_charuco_corners}'
        return 'charuco_ready'

    def _build_preview_image(
        self,
        bgr_image: np.ndarray,
        detection: CharucoDetection | None,
        observation: CandidateFrame | None,
        status_text: str,
    ) -> np.ndarray:
        preview = bgr_image.copy()

        if detection is not None:
            self._draw_detected_markers(preview, detection)

        if observation is not None:
            self._draw_candidate_overlay(preview, observation, show_rank=False)

        marker_count = 0 if detection is None or detection.marker_ids is None else int(len(detection.marker_ids))
        charuco_count = 0 if detection is None else int(detection.charuco_count)
        info_lines = [
            f'status: {status_text}',
            f'markers: {marker_count}  charuco: {charuco_count}',
            f'valid: {len(self.candidate_frames)}  msgs: {self.total_image_msgs}',
        ]

        if self.camera_matrix is None:
            info_lines.append('camera_info: waiting')
        elif observation is not None:
            info_lines.append(
                f'region: {observation.region_bucket}  reproj: {observation.reproj_mean_px:.3f}px'
            )
            info_lines.append(
                f'z: {observation.z_m:.3f}m  tilt: {observation.tilt_deg:.1f}deg'
            )

        self._draw_overlay_text(preview, info_lines)
        return preview

    def _draw_detected_markers(self, image: np.ndarray, detection: CharucoDetection) -> None:
        for index, marker in enumerate(detection.marker_corners):
            points = marker.reshape(-1, 2).astype(np.int32)
            cv2.polylines(image, [points], True, (255, 0, 255), 2, cv2.LINE_AA)
            label_anchor = tuple(points[0])
            marker_label = 'id=?'
            if detection.marker_ids is not None and index < len(detection.marker_ids):
                marker_label = f'id={int(detection.marker_ids[index])}'
            cv2.putText(
                image,
                marker_label,
                (int(label_anchor[0]), int(label_anchor[1]) - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 255),
                1,
                cv2.LINE_AA,
            )

        if detection.charuco_corners is None or detection.charuco_ids is None:
            return

        for point, point_id in zip(detection.charuco_corners.reshape(-1, 2), detection.charuco_ids.reshape(-1)):
            center = (int(round(point[0])), int(round(point[1])))
            cv2.circle(image, center, 4, (0, 255, 0), -1)
            cv2.putText(
                image,
                str(int(point_id)),
                (center[0] + 4, center[1] - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 220, 0),
                1,
                cv2.LINE_AA,
            )

    def _draw_candidate_overlay(
        self,
        image: np.ndarray,
        frame: CandidateFrame,
        show_rank: bool,
    ) -> None:
        for point in frame.corners_px:
            center = (int(round(point[0])), int(round(point[1])))
            cv2.circle(image, center, 4, (0, 255, 0), -1)
        for point in frame.projected_px:
            projected_center = (int(round(point[0])), int(round(point[1])))
            cv2.circle(image, projected_center, 3, (0, 255, 255), -1)

        for observed, projected in zip(frame.corners_px, frame.projected_px):
            cv2.line(
                image,
                (int(round(observed[0])), int(round(observed[1]))),
                (int(round(projected[0])), int(round(projected[1]))),
                (255, 128, 0),
                1,
                cv2.LINE_AA,
            )

        if not show_rank:
            return

        info_lines = [
            f'rank={frame.selected_rank or "-"} region={frame.region_bucket}',
            (
                f'distance={frame.distance_bucket} z={frame.z_m:.3f}m '
                f'tilt={frame.tilt_deg:.1f}deg corners={frame.corner_count}'
            ),
            f'reproj_mean={frame.reproj_mean_px:.3f}px reproj_max={frame.reproj_max_px:.3f}px',
        ]
        self._draw_overlay_text(image, info_lines, origin=(16, 28), line_height=28)

    def _draw_overlay_text(
        self,
        image: np.ndarray,
        lines: list[str],
        origin: tuple[int, int] = (16, 28),
        line_height: int = 24,
    ) -> None:
        if not lines:
            return

        x, y = origin
        max_width = 0
        for line in lines:
            (width, _), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
            max_width = max(max_width, width)

        box_height = line_height * len(lines) + 12
        cv2.rectangle(
            image,
            (x - 10, y - 20),
            (x + max_width + 12, y - 20 + box_height),
            (16, 16, 16),
            -1,
        )

        for line in lines:
            cv2.putText(
                image,
                line,
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (32, 255, 32),
                2,
                cv2.LINE_AA,
            )
            y += line_height

    def _show_preview(self, image: np.ndarray) -> None:
        if not self.show_preview or not self.preview_available:
            return

        preview = image
        if self.preview_max_width > 0 and preview.shape[1] > self.preview_max_width:
            scale = self.preview_max_width / float(preview.shape[1])
            preview = cv2.resize(
                preview,
                (self.preview_max_width, int(round(preview.shape[0] * scale))),
                interpolation=cv2.INTER_AREA,
            )

        try:
            cv2.imshow(self.preview_window_name, preview)
            key = cv2.waitKey(1) & 0xFF
        except cv2.error as exc:
            self.preview_available = False
            self.get_logger().warn(f'无法打开实时预览窗口，已自动关闭预览：{exc}')
            return

        if key in (27, ord('q'), ord('Q')):
            self.get_logger().info('检测到预览窗口退出请求，正在结束本次验证。')
            self.finalize('preview_exit')
            if rclpy.ok():
                rclpy.shutdown()

    def _close_preview(self) -> None:
        if not self.show_preview:
            return
        try:
            cv2.destroyWindow(self.preview_window_name)
            cv2.waitKey(1)
        except cv2.error:
            pass

    def _classify_region(
        self,
        center_x_px: float,
        center_y_px: float,
        image_width: int,
        image_height: int,
    ) -> str:
        x_norm = center_x_px / max(1, image_width)
        y_norm = center_y_px / max(1, image_height)

        col_idx = 0 if x_norm < (1.0 / 3.0) else 1 if x_norm < (2.0 / 3.0) else 2
        row_idx = 0 if y_norm < (1.0 / 3.0) else 1 if y_norm < (2.0 / 3.0) else 2
        labels = (
            ('top_left', 'top_center', 'top_right'),
            ('center_left', 'center', 'center_right'),
            ('bottom_left', 'bottom_center', 'bottom_right'),
        )
        return labels[row_idx][col_idx]

    def _region_bucket_to_class(self, region_bucket: str) -> str:
        if region_bucket == 'center':
            return 'center'
        if region_bucket in ('top_center', 'center_left', 'center_right', 'bottom_center'):
            return 'edge'
        return 'corner'

    def _compute_tilt_deg(self, rvec: np.ndarray) -> float:
        rotation = _rotation_matrix_from_rvec(rvec)
        board_normal = rotation[:, 2]
        cos_tilt = _clamp(abs(float(board_normal[2])), -1.0, 1.0)
        return math.degrees(math.acos(cos_tilt))

    def _finalize_reason_label(self, reason: str) -> str:
        mapping = {
            'timeout_without_camera_info': '采集超时，且未收到 CameraInfo',
            'timeout': '采集超时',
            'preview_exit': '用户关闭了预览窗口',
            'keyboard_interrupt_or_shutdown': '收到退出信号',
        }
        return mapping.get(reason, reason)

    def finalize(self, reason: str) -> None:
        if self.finalized:
            return

        self.finalized = True
        self.finalize_reason = reason
        self._close_preview()
        summary = self._build_summary()
        self._write_outputs(summary)
        self.get_logger().info(
            f'验证结束，原因={self._finalize_reason_label(reason)}，有效检测={len(self.candidate_frames)}，'
            f'报告文件={self.session_dir / "report.txt"}'
        )

    def _build_summary(self) -> dict[str, Any]:
        if not self.candidate_frames:
            return {
                'status': 'no_valid_frames',
                'finalize_reason': self.finalize_reason,
                'message': '未记录到有效的 ChArUco 检测结果。',
                'conclusion': '建议重标 RGB 内参',
                'conclusion_reasons': [
                    '未采集到有效标定板图像，无法证明当前视觉链路稳定。',
                ],
            }

        self._assign_quantile_buckets(self.candidate_frames)
        selected_indices = self._select_frames(self.candidate_frames)
        selected_set = set(selected_indices)
        for rank, candidate_index in enumerate(selected_indices, start=1):
            self.candidate_frames[candidate_index].selected_rank = rank

        overall_errors = np.asarray(
            [frame.reproj_mean_px for frame in self.candidate_frames],
            dtype=np.float64,
        )
        region_class_stats = self._bucket_error_stats(
            self.candidate_frames,
            key=lambda frame: frame.region_class,
            ordered_keys=('center', 'edge', 'corner'),
        )
        region_grid_stats = self._bucket_error_stats(
            self.candidate_frames,
            key=lambda frame: frame.region_bucket,
            ordered_keys=REGION_BUCKETS,
        )
        distance_stats = self._bucket_error_stats(
            self.candidate_frames,
            key=lambda frame: frame.distance_bucket,
            ordered_keys=DISTANCE_BUCKETS,
        )
        tilt_stats = self._bucket_error_stats(
            self.candidate_frames,
            key=lambda frame: frame.tilt_bucket,
            ordered_keys=TILT_BUCKETS,
        )
        stability_summary = self._compute_stability_summary(self.candidate_frames)
        coverage_summary = self._build_coverage_summary(self.candidate_frames, selected_indices)
        conclusion, reasons = self._make_conclusion(
            region_class_stats=region_class_stats,
            distance_stats=distance_stats,
            stability_summary=stability_summary,
            coverage_summary=coverage_summary,
            overall_errors=overall_errors,
        )

        frame_rows = []
        for index, frame in enumerate(self.candidate_frames):
            frame_rows.append(
                {
                    'frame_index': index,
                    'stamp_s': round(frame.stamp_s, 6),
                    'selected': index in selected_set,
                    'selected_rank': frame.selected_rank or '',
                    'region_bucket': frame.region_bucket,
                    'region_class': frame.region_class,
                    'distance_bucket': frame.distance_bucket,
                    'tilt_bucket': frame.tilt_bucket,
                    'z_m': round(frame.z_m, 6),
                    'distance_m': round(frame.distance_m, 6),
                    'tilt_deg': round(frame.tilt_deg, 3),
                    'board_center_x_px': round(frame.board_center_x_px, 3),
                    'board_center_y_px': round(frame.board_center_y_px, 3),
                    'board_area_ratio': round(frame.board_area_ratio, 6),
                    'corner_count': frame.corner_count,
                    'reproj_mean_px': round(frame.reproj_mean_px, 6),
                    'reproj_max_px': round(frame.reproj_max_px, 6),
                    'focus_score': round(frame.focus_score, 6),
                    'rvec': [round(float(v), 6) for v in frame.rvec.reshape(-1)],
                    'tvec_m': [round(float(v), 6) for v in frame.tvec.reshape(-1)],
                    'saved_image_path': frame.saved_image_path,
                }
            )

        return {
            'status': 'ok',
            'finalize_reason': self.finalize_reason,
            'capture_duration_s': round(time.monotonic() - self.start_monotonic_s, 3),
            'image_topic': self.image_topic,
            'camera_info_topic': self.camera_info_topic,
            'camera_info': self.camera_info_snapshot,
            'board': {
                'type': 'charuco',
                'squares_x': self.squares_x,
                'squares_y': self.squares_y,
                'square_length_m': self.square_length_m,
                'marker_length_m': self.marker_length_m,
                'aruco_dictionary': self.aruco_dictionary_name,
                'config_path': self.charuco_config_path or None,
            },
            'charuco_config': self.charuco_config_snapshot,
            'counts': {
                'image_messages_seen': self.total_image_msgs,
                'valid_detections': len(self.candidate_frames),
                'selected_images': len(selected_indices),
            },
            'overall_reprojection_error_px': {
                'mean': round(float(np.mean(overall_errors)), 6),
                'median': round(float(np.median(overall_errors)), 6),
                'p90': round(float(np.percentile(overall_errors, 90)), 6),
                'max': round(float(np.max(overall_errors)), 6),
            },
            'coverage': coverage_summary,
            'region_class_stats': region_class_stats,
            'region_grid_stats': region_grid_stats,
            'distance_stats': distance_stats,
            'tilt_stats': tilt_stats,
            'stability': stability_summary,
            'selected_indices': selected_indices,
            'frames': frame_rows,
            'conclusion': conclusion,
            'conclusion_reasons': reasons,
        }

    def _assign_quantile_buckets(self, frames: list[CandidateFrame]) -> None:
        z_values = np.asarray([frame.z_m for frame in frames], dtype=np.float64)
        tilt_values = np.asarray([frame.tilt_deg for frame in frames], dtype=np.float64)
        z_edges = self._quantile_edges(z_values)
        tilt_edges = self._quantile_edges(tilt_values)

        for frame in frames:
            frame.distance_bucket = self._bucket_from_edges(frame.z_m, z_edges, DISTANCE_BUCKETS)
            frame.tilt_bucket = self._bucket_from_edges(frame.tilt_deg, tilt_edges, TILT_BUCKETS)

    def _quantile_edges(self, values: np.ndarray) -> tuple[float, float] | None:
        if values.size < 3:
            return None
        if float(np.max(values) - np.min(values)) < 1e-6:
            return None
        q1, q2 = np.percentile(values, [33.333, 66.666])
        if (q2 - q1) < 1e-6:
            return None
        return float(q1), float(q2)

    def _bucket_from_edges(
        self,
        value: float,
        edges: tuple[float, float] | None,
        labels: tuple[str, str, str],
    ) -> str:
        if edges is None:
            return labels[1]
        low_edge, high_edge = edges
        if value <= low_edge:
            return labels[0]
        if value <= high_edge:
            return labels[1]
        return labels[2]

    def _select_frames(self, frames: list[CandidateFrame]) -> list[int]:
        available_indices = list(range(len(frames)))
        if not available_indices:
            return []

        region_counts: Counter[str] = Counter()
        distance_counts: Counter[str] = Counter()
        tilt_counts: Counter[str] = Counter()
        selected_indices: list[int] = []

        while available_indices and len(selected_indices) < self.max_selected_count:
            best_index = None
            best_score = -1e9

            for candidate_index in available_indices:
                candidate = frames[candidate_index]
                score = self._selection_score(
                    candidate=candidate,
                    frames=frames,
                    selected_indices=selected_indices,
                    region_counts=region_counts,
                    distance_counts=distance_counts,
                    tilt_counts=tilt_counts,
                )
                if score > best_score:
                    best_score = score
                    best_index = candidate_index

            if best_index is None:
                break

            candidate = frames[best_index]
            selected_indices.append(best_index)
            region_counts[candidate.region_bucket] += 1
            distance_counts[candidate.distance_bucket] += 1
            tilt_counts[candidate.tilt_bucket] += 1
            available_indices.remove(best_index)

            if (
                len(selected_indices) >= self.target_selected_count
                and self._selection_targets_met(region_counts, distance_counts, tilt_counts)
            ):
                break

        return selected_indices

    def _selection_score(
        self,
        candidate: CandidateFrame,
        frames: list[CandidateFrame],
        selected_indices: list[int],
        region_counts: Counter[str],
        distance_counts: Counter[str],
        tilt_counts: Counter[str],
    ) -> float:
        region_need = max(REGION_TARGETS[candidate.region_bucket] - region_counts[candidate.region_bucket], 0)
        distance_need = max(3 - distance_counts[candidate.distance_bucket], 0)
        tilt_need = max(3 - tilt_counts[candidate.tilt_bucket], 0)
        score = 0.0
        score += 7.0 if region_counts[candidate.region_bucket] == 0 else 0.0
        score += 2.0 * float(region_need)
        score += 3.5 if distance_counts[candidate.distance_bucket] == 0 else 0.0
        score += 1.5 * float(distance_need)
        score += 3.0 if tilt_counts[candidate.tilt_bucket] == 0 else 0.0
        score += 1.2 * float(tilt_need)
        score += max(0.0, 2.0 - candidate.reproj_mean_px) * 2.0
        score += min(candidate.focus_score / 200.0, 1.5)

        if selected_indices:
            center_distance = min(
                math.hypot(
                    candidate.center_x_norm - frames[index].center_x_norm,
                    candidate.center_y_norm - frames[index].center_y_norm,
                )
                for index in selected_indices
            )
            pose_distance = min(
                math.hypot(
                    candidate.z_m - frames[index].z_m,
                    (candidate.tilt_deg - frames[index].tilt_deg) / 45.0,
                )
                for index in selected_indices
            )
            score += min(center_distance, 0.5) * 6.0
            score += min(pose_distance, 0.4) * 4.0

        return score

    def _selection_targets_met(
        self,
        region_counts: Counter[str],
        distance_counts: Counter[str],
        tilt_counts: Counter[str],
    ) -> bool:
        region_ok = all(region_counts.get(bucket, 0) >= 1 for bucket in REGION_BUCKETS)
        distance_ok = all(distance_counts.get(bucket, 0) >= 1 for bucket in DISTANCE_BUCKETS)
        tilt_ok = all(tilt_counts.get(bucket, 0) >= 1 for bucket in TILT_BUCKETS)
        return region_ok and distance_ok and tilt_ok

    def _bucket_error_stats(
        self,
        frames: list[CandidateFrame],
        key: Any,
        ordered_keys: tuple[str, ...],
    ) -> dict[str, dict[str, Any]]:
        grouped: defaultdict[str, list[float]] = defaultdict(list)
        for frame in frames:
            grouped[key(frame)].append(frame.reproj_mean_px)

        stats: dict[str, dict[str, Any]] = {}
        for bucket_name in ordered_keys:
            values = np.asarray(grouped.get(bucket_name, []), dtype=np.float64)
            if values.size == 0:
                stats[bucket_name] = {
                    'count': 0,
                    'mean_reproj_px': None,
                    'median_reproj_px': None,
                    'p90_reproj_px': None,
                }
                continue

            stats[bucket_name] = {
                'count': int(values.size),
                'mean_reproj_px': round(float(np.mean(values)), 6),
                'median_reproj_px': round(float(np.median(values)), 6),
                'p90_reproj_px': round(float(np.percentile(values, 90)), 6),
            }
        return stats

    def _compute_stability_summary(self, frames: list[CandidateFrame]) -> dict[str, Any]:
        if len(frames) < self.stability_min_frames:
            return {
                'window_count': 0,
                'message': '有效帧数量不足，无法进行稳定性分析。',
            }

        sorted_frames = sorted(frames, key=lambda frame: frame.stamp_s)
        windows: list[list[CandidateFrame]] = []
        current_window = [sorted_frames[0]]

        for frame in sorted_frames[1:]:
            previous = current_window[-1]
            time_gap = frame.stamp_s - previous.stamp_s
            translation_gap = float(np.linalg.norm(frame.tvec - previous.tvec))
            rotation_gap = _rotation_delta_deg(previous.rvec, frame.rvec)
            if (
                time_gap <= self.stability_max_gap_s
                and translation_gap <= self.stability_translation_gate_m
                and rotation_gap <= self.stability_rotation_gate_deg
            ):
                current_window.append(frame)
            else:
                if len(current_window) >= self.stability_min_frames:
                    windows.append(current_window)
                current_window = [frame]

        if len(current_window) >= self.stability_min_frames:
            windows.append(current_window)

        if not windows:
            return {
                'window_count': 0,
                'message': '没有任何静止姿态窗口满足稳定性门限。',
            }

        window_summaries = []
        for window_index, window in enumerate(windows):
            tvecs = np.stack([frame.tvec.reshape(3) for frame in window], axis=0)
            centers = np.stack(
                [[frame.board_center_x_px, frame.board_center_y_px] for frame in window],
                axis=0,
            )
            reference_rvec = window[0].rvec
            rotation_deltas_deg = np.asarray(
                [_rotation_delta_deg(reference_rvec, frame.rvec) for frame in window],
                dtype=np.float64,
            )
            translation_centered = tvecs - np.mean(tvecs, axis=0, keepdims=True)
            translation_norms_mm = np.asarray(
                [_vector_norm_mm(vector) for vector in translation_centered],
                dtype=np.float64,
            )

            window_summaries.append(
                {
                    'window_index': window_index,
                    'frame_count': len(window),
                    'start_stamp_s': round(window[0].stamp_s, 6),
                    'end_stamp_s': round(window[-1].stamp_s, 6),
                    'duration_s': round(window[-1].stamp_s - window[0].stamp_s, 6),
                    'translation_jitter_mm_mean': round(float(np.mean(translation_norms_mm)), 6),
                    'translation_jitter_mm_max': round(float(np.max(translation_norms_mm)), 6),
                    'rotation_jitter_deg_std': round(float(np.std(rotation_deltas_deg)), 6),
                    'rotation_jitter_deg_max': round(float(np.max(rotation_deltas_deg)), 6),
                    'center_jitter_px_mean': round(
                        float(np.mean(np.linalg.norm(centers - np.mean(centers, axis=0, keepdims=True), axis=1))),
                        6,
                    ),
                }
            )

        translation_means = np.asarray(
            [window['translation_jitter_mm_mean'] for window in window_summaries],
            dtype=np.float64,
        )
        rotation_stds = np.asarray(
            [window['rotation_jitter_deg_std'] for window in window_summaries],
            dtype=np.float64,
        )

        return {
            'window_count': len(window_summaries),
            'windows': window_summaries,
            'median_translation_jitter_mm_mean': round(float(np.median(translation_means)), 6),
            'max_translation_jitter_mm_mean': round(float(np.max(translation_means)), 6),
            'median_rotation_jitter_deg_std': round(float(np.median(rotation_stds)), 6),
            'max_rotation_jitter_deg_std': round(float(np.max(rotation_stds)), 6),
        }

    def _build_coverage_summary(
        self,
        frames: list[CandidateFrame],
        selected_indices: list[int],
    ) -> dict[str, Any]:
        region_counts = Counter(frame.region_bucket for frame in frames)
        distance_counts = Counter(frame.distance_bucket for frame in frames)
        tilt_counts = Counter(frame.tilt_bucket for frame in frames)
        selected_region_counts = Counter(frames[index].region_bucket for index in selected_indices)
        selected_distance_counts = Counter(frames[index].distance_bucket for index in selected_indices)
        selected_tilt_counts = Counter(frames[index].tilt_bucket for index in selected_indices)

        z_values = np.asarray([frame.z_m for frame in frames], dtype=np.float64)
        tilt_values = np.asarray([frame.tilt_deg for frame in frames], dtype=np.float64)

        warnings = []
        if sum(1 for bucket in REGION_BUCKETS if region_counts.get(bucket, 0) > 0) < len(REGION_BUCKETS):
            warnings.append('区域覆盖不完整，中心/边缘/角落至少有一部分未覆盖。')
        if sum(1 for bucket in DISTANCE_BUCKETS if distance_counts.get(bucket, 0) > 0) < len(DISTANCE_BUCKETS):
            warnings.append('距离变化不足，near/mid/far 未全部覆盖。')
        if sum(1 for bucket in TILT_BUCKETS if tilt_counts.get(bucket, 0) > 0) < len(TILT_BUCKETS):
            warnings.append('倾角变化不足，多种倾角覆盖不完整。')
        if len(selected_indices) < self.target_selected_count:
            warnings.append(
                f'最终仅选出 {len(selected_indices)} 张验证图，低于目标 {self.target_selected_count} 张。'
            )

        return {
            'region_counts_all_valid': {bucket: int(region_counts.get(bucket, 0)) for bucket in REGION_BUCKETS},
            'distance_counts_all_valid': {bucket: int(distance_counts.get(bucket, 0)) for bucket in DISTANCE_BUCKETS},
            'tilt_counts_all_valid': {bucket: int(tilt_counts.get(bucket, 0)) for bucket in TILT_BUCKETS},
            'region_counts_selected': {
                bucket: int(selected_region_counts.get(bucket, 0)) for bucket in REGION_BUCKETS
            },
            'distance_counts_selected': {
                bucket: int(selected_distance_counts.get(bucket, 0)) for bucket in DISTANCE_BUCKETS
            },
            'tilt_counts_selected': {
                bucket: int(selected_tilt_counts.get(bucket, 0)) for bucket in TILT_BUCKETS
            },
            'z_range_m': [round(float(np.min(z_values)), 6), round(float(np.max(z_values)), 6)],
            'tilt_range_deg': [
                round(float(np.min(tilt_values)), 6),
                round(float(np.max(tilt_values)), 6),
            ],
            'warnings': warnings,
        }

    def _make_conclusion(
        self,
        region_class_stats: dict[str, dict[str, Any]],
        distance_stats: dict[str, dict[str, Any]],
        stability_summary: dict[str, Any],
        coverage_summary: dict[str, Any],
        overall_errors: np.ndarray,
    ) -> tuple[str, list[str]]:
        reasons: list[str] = []

        center_mean = region_class_stats['center']['mean_reproj_px']
        edge_mean = region_class_stats['edge']['mean_reproj_px']
        corner_mean = region_class_stats['corner']['mean_reproj_px']
        near_mean = distance_stats['near']['mean_reproj_px']
        far_mean = distance_stats['far']['mean_reproj_px']

        if center_mean is None:
            reasons.append('中心区域没有有效样本，无法证明中心区域稳定。')
        elif center_mean > 0.8:
            reasons.append(f'中心区域平均重投影误差 {center_mean:.3f}px，偏大。')

        if stability_summary.get('window_count', 0) == 0:
            reasons.append('没有找到满足门限的静止姿态窗口，姿态抖动无法被证明足够稳定。')
        else:
            median_translation = float(stability_summary['median_translation_jitter_mm_mean'])
            median_rotation = float(stability_summary['median_rotation_jitter_deg_std'])
            if median_translation > 3.0:
                reasons.append(f'静止姿态平移抖动中位数 {median_translation:.3f}mm，偏大。')
            if median_rotation > 0.5:
                reasons.append(f'静止姿态旋转抖动中位数 {median_rotation:.3f}deg，偏大。')

        if center_mean is not None and edge_mean is not None:
            if edge_mean > center_mean * 1.5 and (edge_mean - center_mean) > 0.2:
                reasons.append(
                    f'边缘区域平均重投影误差 {edge_mean:.3f}px，明显高于中心区域 {center_mean:.3f}px。'
                )

        if center_mean is not None and corner_mean is not None:
            if corner_mean > center_mean * 1.8 and (corner_mean - center_mean) > 0.25:
                reasons.append(
                    f'角落区域平均重投影误差 {corner_mean:.3f}px，明显高于中心区域 {center_mean:.3f}px。'
                )

        if near_mean is not None and far_mean is not None:
            if far_mean > near_mean * 1.5 and (far_mean - near_mean) > 0.2:
                reasons.append(
                    f'远距离平均重投影误差 {far_mean:.3f}px，明显高于近距离 {near_mean:.3f}px。'
                )

        overall_mean = float(np.mean(overall_errors))
        overall_p90 = float(np.percentile(overall_errors, 90))
        if overall_mean > 0.9 or overall_p90 > 1.3:
            reasons.append(
                f'整体重投影误差偏大，mean={overall_mean:.3f}px, p90={overall_p90:.3f}px。'
            )

        if coverage_summary['warnings']:
            reasons.extend(coverage_summary['warnings'])

        if reasons:
            return '建议重标 RGB 内参', reasons

        return '当前内参可用', [
            f'中心区域平均重投影误差 {center_mean:.3f}px，未见明显异常。',
            f'整体重投影误差 mean={overall_mean:.3f}px, p90={overall_p90:.3f}px。',
            '静止姿态窗口与区域/距离覆盖未见明显恶化趋势。',
        ]

    def _write_outputs(self, summary: dict[str, Any]) -> None:
        self.session_dir.mkdir(parents=True, exist_ok=True)
        selected_indices = summary.get('selected_indices', [])
        frame_rows_by_index = {
            int(frame_row['frame_index']): frame_row
            for frame_row in summary.get('frames', [])
        }

        for candidate_index in selected_indices:
            frame = self.candidate_frames[candidate_index]
            image = cv2.imdecode(np.frombuffer(frame.jpeg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                continue
            annotated = self._annotate_frame(image, frame)
            image_name = (
                f'{frame.selected_rank:02d}_'
                f'{frame.region_bucket}_{frame.distance_bucket}_{frame.tilt_bucket}.png'
            )
            image_path = self.images_dir / image_name
            cv2.imwrite(str(image_path), annotated)
            frame.saved_image_path = str(image_path)
            if candidate_index in frame_rows_by_index:
                frame_rows_by_index[candidate_index]['saved_image_path'] = str(image_path)

        frames_csv_path = self.session_dir / 'frame_metrics.csv'
        with frames_csv_path.open('w', newline='', encoding='utf-8') as csv_file:
            fieldnames = [
                'frame_index',
                'stamp_s',
                'selected',
                'selected_rank',
                'region_bucket',
                'region_class',
                'distance_bucket',
                'tilt_bucket',
                'z_m',
                'distance_m',
                'tilt_deg',
                'board_center_x_px',
                'board_center_y_px',
                'board_area_ratio',
                'corner_count',
                'reproj_mean_px',
                'reproj_max_px',
                'focus_score',
                'rvec',
                'tvec_m',
                'saved_image_path',
            ]
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            for frame in summary.get('frames', []):
                writer.writerow(frame)

        if self.camera_info_snapshot:
            with (self.session_dir / 'camera_info_snapshot.json').open('w', encoding='utf-8') as file_obj:
                json.dump(self.camera_info_snapshot, file_obj, ensure_ascii=False, indent=2)

        with (self.session_dir / 'summary.json').open('w', encoding='utf-8') as file_obj:
            json.dump(summary, file_obj, ensure_ascii=False, indent=2)

        report_text = self._build_report_text(summary)
        with (self.session_dir / 'report.txt').open('w', encoding='utf-8') as file_obj:
            file_obj.write(report_text)

    def _annotate_frame(self, image: np.ndarray, frame: CandidateFrame) -> np.ndarray:
        annotated = image.copy()
        self._draw_candidate_overlay(annotated, frame, show_rank=True)
        return annotated

    def _build_report_text(self, summary: dict[str, Any]) -> str:
        if summary.get('status') != 'ok':
            return (
                '奥比中光 RGB 内参验证报告（ChArUco）\n'
                f'结论：{summary["conclusion"]}\n'
                f'原因：{summary["message"]}\n'
            )

        counts = summary['counts']
        overall = summary['overall_reprojection_error_px']
        stability = summary['stability']
        coverage = summary['coverage']
        region_stats = summary['region_class_stats']
        distance_stats = summary['distance_stats']
        reasons = summary['conclusion_reasons']

        lines = [
            '奥比中光 RGB 内参验证报告（ChArUco）',
            '',
            '范围说明：',
            '- 仅验证视觉链路与 T_cam_board 稳定性，不夹杂 hand-eye。',
            '',
            '采集概况：',
            f'- image_topic: {summary["image_topic"]}',
            f'- camera_info_topic: {summary["camera_info_topic"]}',
            (
                f'- ChArUco: {summary["board"]["squares_x"]}x{summary["board"]["squares_y"]}, '
                f'square={summary["board"]["square_length_m"]:.3f}m, '
                f'marker={summary["board"]["marker_length_m"]:.3f}m, '
                f'dict={summary["board"]["aruco_dictionary"]}'
            ),
            (
                f'- 有效检测 {counts["valid_detections"]} 帧，'
                f'输出验证图 {counts["selected_images"]} 张'
            ),
            (
                f'- 整体重投影误差: mean={overall["mean"]:.3f}px, '
                f'median={overall["median"]:.3f}px, p90={overall["p90"]:.3f}px, '
                f'max={overall["max"]:.3f}px'
            ),
            '',
            '观察点：',
            (
                f'- 中心区域: count={region_stats["center"]["count"]}, '
                f'mean={region_stats["center"]["mean_reproj_px"]}'
            ),
            (
                f'- 边缘区域: count={region_stats["edge"]["count"]}, '
                f'mean={region_stats["edge"]["mean_reproj_px"]}'
            ),
            (
                f'- 角落区域: count={region_stats["corner"]["count"]}, '
                f'mean={region_stats["corner"]["mean_reproj_px"]}'
            ),
            (
                f'- 近中远距离误差: near={distance_stats["near"]["mean_reproj_px"]}, '
                f'mid={distance_stats["mid"]["mean_reproj_px"]}, '
                f'far={distance_stats["far"]["mean_reproj_px"]}'
            ),
        ]

        if stability.get('window_count', 0) > 0:
            lines.extend(
                [
                    (
                        f'- 静止姿态窗口: {stability["window_count"]} 段, '
                        f'平移抖动中位数={stability["median_translation_jitter_mm_mean"]:.3f}mm, '
                        f'旋转抖动中位数={stability["median_rotation_jitter_deg_std"]:.3f}deg'
                    ),
                ]
            )
        else:
            lines.append(f'- 静止姿态窗口: {stability.get("message", "无")}')

        lines.extend(
            [
                (
                    f'- 距离范围: {coverage["z_range_m"][0]:.3f}m ~ {coverage["z_range_m"][1]:.3f}m, '
                    f'倾角范围: {coverage["tilt_range_deg"][0]:.1f}deg ~ {coverage["tilt_range_deg"][1]:.1f}deg'
                ),
                '',
                f'结论：{summary["conclusion"]}',
            ]
        )

        if reasons:
            lines.append('依据：')
            for reason in reasons:
                lines.append(f'- {reason}')

        if coverage['warnings']:
            lines.append('')
            lines.append('覆盖提醒：')
            for warning in coverage['warnings']:
                lines.append(f'- {warning}')

        lines.append('')
        lines.append('建议顺序：先确认本次 T_cam_board 可信，再继续做机器人外参。')
        return '\n'.join(lines) + '\n'


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = OrbbecIntrinsicsValidator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('收到键盘中断，正在整理当前验证结果。')
    finally:
        node.finalize('keyboard_interrupt_or_shutdown')
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
