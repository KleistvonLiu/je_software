#!/usr/bin/env python3
import time

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import JointState
from common.msg import OculusInitJointState
from common_utils.ros2_qos import sensor_qos, reliable_qos_shallow


class JointRateMonitor(Node):
    def __init__(self) -> None:
        super().__init__("joint_rate_monitor")

        self.declare_parameter("topic", "/joint_states_double_arm")
        self.declare_parameter("msg_type", "oculus_init_joint_state")
        self.declare_parameter("log_period_s", 1.0)

        topic = str(self.get_parameter("topic").value)
        msg_type = str(self.get_parameter("msg_type").value).strip().lower()
        log_period_s = float(self.get_parameter("log_period_s").value)
        self._log_period_s = max(0.1, log_period_s)
        self._topic = topic

        if msg_type in ("oculus", "oculus_init_joint_state", "oculusinitjointstate"):
            msg_cls = OculusInitJointState
        elif msg_type in ("joint_state", "jointstate"):
            msg_cls = JointState
        else:
            raise RuntimeError(
                f"Unsupported msg_type '{msg_type}'. Use 'oculus_init_joint_state' or 'joint_state'."
            )

        self._count = 0
        self._last_count = 0
        self._last_log_time = time.perf_counter()
        self._last_msg_time = None
        self._last_interval = None

        self._sub = self.create_subscription(msg_cls, topic, self._cb, reliable_qos_shallow)
        self.get_logger().info(
            f"Monitoring {topic} (type={msg_cls.__name__}) log_period_s={self._log_period_s:.2f}"
        )

    def _cb(self, _msg) -> None:
        now = time.perf_counter()
        self._count += 1
        if self._last_msg_time is not None:
            self._last_interval = now - self._last_msg_time
        self._last_msg_time = now
        self.get_logger().info (
                        f"topic={self._topic} time={now:.6f} "
        )
        if now - self._last_log_time >= self._log_period_s:
            dt = now - self._last_log_time
            delta = self._count - self._last_count
            hz = (delta / dt) if dt > 0 else 0.0
            inst_hz = (
                (1.0 / self._last_interval)
                if self._last_interval and self._last_interval > 0.0
                else 0.0
            )
            self.get_logger().info(
                f"recv_rate={hz:.2f}Hz inst={inst_hz:.2f}Hz count={delta} period={dt:.2f}s"
            )
            self._last_log_time = now
            self._last_count = self._count


def main(args=None) -> None:
    rclpy.init(args=args)
    node = JointRateMonitor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
