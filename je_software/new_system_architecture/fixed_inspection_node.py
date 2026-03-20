#!/usr/bin/env python3
"""固定检测服务。

这个节点用来模拟真实 AOI/检测设备：
1. 收到 TriggerInspection 请求后先返回 accepted
2. 延时一小段时间
3. 在 result topic 上发布固定结果，默认是 good

这样任务管理器就可以按真实系统的“请求 + 异步结果”模式工作。
"""

from __future__ import annotations

import threading
import time

import rclpy
from rclpy.node import Node

from je_software.msg import InspectionResult
from je_software.srv import TriggerInspection


class FixedInspectionNode(Node):
    """接受检测请求，并在稍后发布一个固定检测结果。"""

    def __init__(self) -> None:
        super().__init__('fixed_inspection_node')
        self.declare_parameter('service_name', '/inspection/trigger')
        self.declare_parameter('result_topic', '/inspection/result')
        self.declare_parameter('fixed_result', 'good')
        self.declare_parameter('response_delay_sec', 0.5)

        self.fixed_result = str(self.get_parameter('fixed_result').value)
        self.response_delay_sec = float(
            self.get_parameter('response_delay_sec').value
        )
        # 用一个简单的 pending 标志模拟“检测设备忙碌中”。
        # 初版不做排队，只允许同时处理一个请求。
        self._pending_lock = threading.Lock()
        self._pending = False

        self.publisher = self.create_publisher(
            InspectionResult,
            str(self.get_parameter('result_topic').value),
            10,
        )
        self.create_service(
            TriggerInspection,
            str(self.get_parameter('service_name').value),
            self._handle_trigger,
        )
        self.get_logger().info('Fixed inspection node ready.')

    def _handle_trigger(
        self,
        request: TriggerInspection.Request,
        response: TriggerInspection.Response,
    ) -> TriggerInspection.Response:
        # pcb_id 由任务管理器生成，用来在后续结果回调里做流程对齐。
        pcb_id = str(request.pcb_id).strip()
        if not pcb_id:
            response.accepted = False
            response.reason = 'empty_pcb_id'
            return response

        with self._pending_lock:
            if self._pending:
                response.accepted = False
                response.reason = 'inspection_busy'
                return response
            self._pending = True

        # 用后台线程延时发布结果，避免阻塞 service 回调线程。
        worker = threading.Thread(
            target=self._publish_result_after_delay,
            args=(pcb_id,),
            daemon=True,
        )
        worker.start()
        response.accepted = True
        response.reason = 'accepted'
        return response

    def _publish_result_after_delay(self, pcb_id: str) -> None:
        try:
            time.sleep(max(self.response_delay_sec, 0.0))
            msg = InspectionResult()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.pcb_id = pcb_id
            msg.result = self.fixed_result
            msg.valid = True
            # 这里固定发 good，后续接真实检测设备时只需要替换这个节点。
            self.publisher.publish(msg)
        finally:
            with self._pending_lock:
                self._pending = False


def main() -> None:
    rclpy.init()
    node = FixedInspectionNode()
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
