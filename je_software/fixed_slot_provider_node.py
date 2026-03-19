#!/usr/bin/env python3
"""固定槽位服务。

这个节点不做真实料框检测和占用判断，只返回一个固定 good 槽位。
目的是先把“检测完成 -> 请求槽位 -> 执行放置”整条链路跑通。
"""

from __future__ import annotations

import rclpy
from rclpy.node import Node

from je_software.pcb_process_common import make_pose_stamped
from je_software.pcb_process_common import require_pose_list
from je_software.srv import GetAvailableSlot


class FixedSlotProviderNode(Node):
    """始终返回同一个固定槽位。"""

    def __init__(self) -> None:
        super().__init__('fixed_slot_provider_node')
        self.declare_parameter(
            'service_name',
            '/vision/slot/get_available_slot',
        )
        self.declare_parameter('frame_id', 'base_link')
        self.declare_parameter('fixed_box_type', 'good')
        self.declare_parameter('fixed_slot_id', 0)
        self.declare_parameter('slot_pose', [0.22, -0.28, 0.08, 3.14, 0.0, 0.0])
        self.declare_parameter('confidence', 1.0)

        self.frame_id = str(self.get_parameter('frame_id').value)
        self.fixed_box_type = str(self.get_parameter('fixed_box_type').value)
        self.fixed_slot_id = int(self.get_parameter('fixed_slot_id').value)
        self.slot_pose = require_pose_list(
            self.get_parameter('slot_pose').value,
            'slot_pose',
        )
        self.confidence = float(self.get_parameter('confidence').value)

        self.create_service(
            GetAvailableSlot,
            str(self.get_parameter('service_name').value),
            self._handle_slot_request,
        )
        self.get_logger().info('Fixed slot provider node ready.')

    def _handle_slot_request(
        self,
        request: GetAvailableSlot.Request,
        response: GetAvailableSlot.Response,
    ) -> GetAvailableSlot.Response:
        # 初版只支持一个固定框类型，默认是 good。
        # 如果任务管理器请求了别的框，直接返回失败，便于尽早暴露配置错误。
        if request.box_type and request.box_type != self.fixed_box_type:
            response.success = False
            response.reason = 'unsupported_box_type'
            return response

        response.success = True
        response.reason = 'ok'
        response.slot_id = self.fixed_slot_id
        # 槽位位姿固定来自 YAML，不考虑真实占用状态。
        response.slot_pose_base = make_pose_stamped(self.slot_pose, self.frame_id)
        response.slot_pose_base.header.stamp = self.get_clock().now().to_msg()
        response.slot_empty = True
        response.confidence = self.confidence
        return response


def main() -> None:
    rclpy.init()
    node = FixedSlotProviderNode()
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
