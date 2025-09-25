#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import threading
from typing import Optional, Tuple

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import Header

import pyorbbecsdk

# 尝试使用 cv_bridge（若没有也可走手工构造消息）
try:
    from cv_bridge import CvBridge
    _HAS_BRIDGE = True
except Exception:
    _HAS_BRIDGE = False
    CvBridge = None  # type: ignore

# ====== 这里假设你把 OrbbecCamera / OrbbecCameraConfig 放在可导入路径上 ======
# 例如 your_project.cameras.orbbec_camera import ...
# 按你的真实包路径修改下面两行：
from .cameras.orbbec import OrbbecCamera, OrbbecCameraConfig  # noqa
from .cameras.errors import DeviceNotConnectedError  # 若你的工程里有该异常

class OrbbecRos2Node(Node):
    def __init__(self):
        super().__init__('orbbec_publisher')

        # ---------------- 参数声明（可由 YAML 覆盖） ----------------
        self.declare_parameter('color_topic', '/camera/color/image_raw')
        self.declare_parameter('depth_topic', '/camera/depth/image_raw')
        self.declare_parameter('publish_depth', True)
        self.declare_parameter('frame_id_color', 'orbbec_color_optical_frame')
        self.declare_parameter('frame_id_depth', 'orbbec_depth_optical_frame')

        # 与 OrbbecCameraConfig 对应的常用相机参数（可选）
        self.declare_parameter('width', 640)
        self.declare_parameter('height', 480)
        self.declare_parameter('fps', 30)
        self.declare_parameter('use_depth', False)
        self.declare_parameter('index_or_path', '')   # 相机序列号或路径
        self.declare_parameter('warmup_s', 1)
        self.declare_parameter('color_mode', 'rgb')   # 'rgb' or 'bgr'

        # ---------------- 读取参数 ----------------
        self.color_topic: str = self.get_parameter('color_topic').get_parameter_value().string_value
        self.depth_topic: str = self.get_parameter('depth_topic').get_parameter_value().string_value
        self.publish_depth: bool = self.get_parameter('publish_depth').get_parameter_value().bool_value
        self.frame_id_color: str = self.get_parameter('frame_id_color').get_parameter_value().string_value
        self.frame_id_depth: str = self.get_parameter('frame_id_depth').get_parameter_value().string_value

        width = int(self.get_parameter('width').get_parameter_value().integer_value)
        height = int(self.get_parameter('height').get_parameter_value().integer_value)
        fps = int(self.get_parameter('fps').get_parameter_value().integer_value)
        use_depth = bool(self.get_parameter('use_depth').get_parameter_value().bool_value)
        index_or_path = self.get_parameter('index_or_path').get_parameter_value().string_value
        warmup_s = int(self.get_parameter('warmup_s').get_parameter_value().integer_value)
        color_mode = self.get_parameter('color_mode').get_parameter_value().string_value or 'rgb'

        # ---------------- Publisher（SensorData QoS） ----------------
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.color_pub = self.create_publisher(Image, self.color_topic, qos)
        self.depth_pub = self.create_publisher(Image, self.depth_topic, qos) if (use_depth and self.publish_depth) else None

        if _HAS_BRIDGE:
            self.bridge = CvBridge()
        else:
            self.bridge = None

        # ---------------- 相机初始化 ----------------
        # 依据你的 OrbbecCameraConfig 字段名初始化（与贴出来的一致）
        ctx = pyorbbecsdk.Context()
        device_list = ctx.query_devices()
        cfg = OrbbecCameraConfig(
            color_mode=color_mode,
            index_or_path=index_or_path if index_or_path else None,
            channels=3,
            TemporalFilter_alpha=0.5,
            warmup_s=warmup_s,
            width=width,
            height=height,
            fps=fps,
            use_depth=use_depth,
            device_list=device_list,  # 若你的 Config 需要，这里改成你的获取方式
        )

        # 注意：上面 device_list 的获取方式依你工程为准；若你的 Config 不需要可删除该参数。
        # 如果你工程内的 __post_init__ 有更强校验，确保 width/height/fps 都已给出。

        self.cam = OrbbecCamera(cfg, logger=self.get_logger())
        try:
            self.cam.connect(warmup=True)
        except Exception as e:
            self.get_logger().error(f'Failed to connect Orbbec camera: {e}')
            raise

        # 使用后台异步循环读取
        self._read_timeout_ms = max(1, int(1000.0 / fps / 2))  # 15~20ms
        self._timer = self.create_timer(max(1.0 / fps * 0.5, 0.005), self._on_timer)
        self._lock = threading.Lock()
        self.get_logger().info(
            f'Publishing color to "{self.color_topic}"'
            + (f', depth to "{self.depth_topic}"' if self.depth_pub else ', depth publishing disabled')
        )

    # --------- 工具：组装 Image 消息（无 cv_bridge 时） ----------
    def _to_image_msg_manual(self, np_img: np.ndarray, encoding: str, frame_id: str) -> Image:
        msg = Image()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = frame_id

        if np_img.ndim == 2:  # (H, W)
            h, w = np_img.shape
            msg.height = h
            msg.width = w
        elif np_img.ndim == 3:
            h, w, _ = np_img.shape
            msg.height = h
            msg.width = w
        else:
            raise ValueError(f'Unsupported image shape: {np_img.shape}')

        msg.encoding = encoding
        msg.is_bigendian = 0
        # 步幅：rgb8->3 字节, bgr8->3 字节, 16UC1->2 字节
        bpp = 3 if encoding in ('rgb8', 'bgr8') else 2
        msg.step = w * bpp
        if np_img.flags['C_CONTIGUOUS'] is False:
            np_img = np.ascontiguousarray(np_img)
        msg.data = np_img.tobytes()
        return msg

    def _publish_color(self, color_np: np.ndarray):
        enc = 'rgb8'  # 你给的转换函数默认输出 RGB ndarray
        if self.bridge:
            msg = self.bridge.cv2_to_imgmsg(color_np, encoding=enc)
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = self.frame_id_color
        else:
            msg = self._to_image_msg_manual(color_np, enc, self.frame_id_color)
        self.color_pub.publish(msg)

    def _publish_depth(self, depth_np: np.ndarray):
        """
        兼容两种深度格式：
        1) (H,W) 或 (H,W,1) uint16：直接 16UC1；
        2) (H,W,3) uint8：按默认 HI_LO（R=高8位，G=低8位）解包为 uint16 再发 16UC1。
        """
        if depth_np is None or self.depth_pub is None:
            return

        # 解包为 uint16
        if depth_np.dtype == np.uint16:
            if depth_np.ndim == 3 and depth_np.shape[2] == 1:
                depth_u16 = depth_np[..., 0]
            else:
                depth_u16 = depth_np
        elif depth_np.dtype == np.uint8 and depth_np.ndim == 3 and depth_np.shape[2] == 3:
            # 假设默认 HI_LO：R=高8位，G=低8位
            r = depth_np[..., 0].astype(np.uint16)
            g = depth_np[..., 1].astype(np.uint16)
            depth_u16 = (r << 8) | g
        else:
            self.get_logger().warn(f'Unexpected depth dtype/shape: {depth_np.dtype} {depth_np.shape}, skip publish.')
            return

        if self.bridge:
            msg = self.bridge.cv2_to_imgmsg(depth_u16, encoding='16UC1')
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = self.frame_id_depth
        else:
            msg = self._to_image_msg_manual(depth_u16, '16UC1', self.frame_id_depth)

        self.depth_pub.publish(msg)

    def _on_timer(self):
        with self._lock:
            try:
                # 用异步接口拿最新帧（不阻塞主线程）
                frame = None
                try:
                    frame = self.cam.async_read(timeout_ms=self._read_timeout_ms)
                except TimeoutError:
                    return  # 没新帧就下个周期再试
                except DeviceNotConnectedError as e:
                    self.get_logger().error(f'Camera disconnected: {e}')
                    rclpy.shutdown()
                    return

                if frame is None:
                    return

                if isinstance(frame, tuple):
                    color_np, depth_np = frame  # (H,W,3) u8, 以及深度
                else:
                    color_np, depth_np = frame, None

                if color_np is not None:
                    self._publish_color(color_np)

                if self.publish_depth and (depth_np is not None) and (self.depth_pub is not None):
                    self._publish_depth(depth_np)

            except Exception as e:
                self.get_logger().warn(f'Publish error: {e}')

    def destroy_node(self):
        try:
            if hasattr(self, 'cam') and self.cam and self.cam.is_connected:
                self.cam.disconnect()
        except Exception as e:
            self.get_logger().warn(f'Error while disconnecting camera: {e}')
        return super().destroy_node()


def main(argv=None):
    rclpy.init(args=argv)
    node = OrbbecRos2Node()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
