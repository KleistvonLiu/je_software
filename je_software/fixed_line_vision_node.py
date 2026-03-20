#!/usr/bin/env python3
"""流水线侧固定感知服务。

这个节点不做真实视觉处理，只做两件事：
1. 订阅键盘节点发布的 PCB 到位状态
2. 在收到抓取位姿请求时，返回 YAML 里配置好的固定 pose

这样可以先把“任务调度 -> 取位姿 -> 执行动作”整条链路跑通。
"""

from __future__ import annotations

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy
from rclpy.qos import QoSProfile
from rclpy.qos import ReliabilityPolicy

from je_software.msg import PcbPresence
from je_software.pcb_process_common import make_pose_stamped
from je_software.pcb_process_common import require_pose_list
from je_software.srv import GetPcbPickPose


class FixedLineVisionNode(Node):
    """在 PCB 被标记为 ready 时，返回固定的 PCB 位姿和抓取位姿。"""

    def __init__(self) -> None:
        super().__init__('fixed_line_vision_node')
        self.declare_parameter('presence_topic', '/vision/line/pcb_presence')
        self.declare_parameter('service_name', '/vision/line/get_pcb_pick_pose')
        self.declare_parameter('frame_id', 'base_link')
        self.declare_parameter('pcb_pose', [0.45, 0.00, 0.12, 3.14, 0.0, 0.0])
        self.declare_parameter('pick_pose', [0.45, 0.00, 0.12, 3.14, 0.0, 0.0])
        self.declare_parameter('confidence', 1.0)

        self.frame_id = str(self.get_parameter('frame_id').value)
        self.confidence = float(self.get_parameter('confidence').value)
        self._pcb_pose = require_pose_list(
            self.get_parameter('pcb_pose').value,
            'pcb_pose',
        )
        self._pick_pose = require_pose_list(
            self.get_parameter('pick_pose').value,
            'pick_pose',
        )
        # 缓存最近一次“PCB 是否到位”的状态。
        # service 回调只读这个缓存，不自己做图像处理。
        self._latest_presence = PcbPresence()

        qos = QoSProfile(depth=1)
        qos.reliability = ReliabilityPolicy.RELIABLE
        qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self.create_subscription(
            PcbPresence,
            str(self.get_parameter('presence_topic').value),
            self._presence_callback,
            qos,
        )
        self.create_service(
            GetPcbPickPose,
            str(self.get_parameter('service_name').value),
            self._handle_get_pick_pose,
        )
        self.get_logger().info(
            'Fixed line vision node ready. '
            f'loaded pcb_pose={self._format_values(self._pcb_pose)}, '
            f'pick_pose={self._format_values(self._pick_pose)}, '
            f'frame_id={self.frame_id}'
        )

    def _format_values(self, values) -> str:
        return '[' + ', '.join(f'{float(value):.6f}' for value in values) + ']'

    def _presence_callback(self, msg: PcbPresence) -> None:
        # 持续记录最新状态，供 service 请求时判断是否允许抓取。
        self._latest_presence = msg

    def _handle_get_pick_pose(
        self,
        request: GetPcbPickPose.Request,
        response: GetPcbPickPose.Response,
    ) -> GetPcbPickPose.Response:
        # 这三个判断对应的是最小流程门禁：
        # 1. 是否有板
        # 2. 是否稳定
        # 3. 是否允许抓取
        if not self._latest_presence.present:
            response.success = False
            response.reason = 'pcb_not_present'
            return response
        if request.require_stable and not self._latest_presence.stable:
            response.success = False
            response.reason = 'pcb_not_stable'
            return response
        if not self._latest_presence.ready_for_pick:
            response.success = False
            response.reason = 'pcb_not_ready'
            return response

        # 用当前 service 响应时刻打时间戳，方便消费方确认这是“当前有效结果”。
        now = self.get_clock().now().to_msg()
        response.success = True
        response.reason = 'ok'
        response.pcb_pose_base = make_pose_stamped(self._pcb_pose, self.frame_id)
        response.pick_pose_base = make_pose_stamped(self._pick_pose, self.frame_id)
        self.get_logger().info(
            'Using preloaded fixed poses for GetPcbPickPose: '
            f'pcb_pose={self._format_values(self._pcb_pose)}, '
            f'pick_pose={self._format_values(self._pick_pose)}'
        )
        response.pcb_pose_base.header.stamp = now
        response.pick_pose_base.header.stamp = now
        response.confidence = self.confidence
        return response


def main() -> None:
    rclpy.init()
    node = FixedLineVisionNode()
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
