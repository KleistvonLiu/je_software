#!/usr/bin/env python3
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import QoSProfile
from sensor_msgs.msg import JointState
from piper_sdk import C_PiperInterface
from rclpy.parameter import Parameter
import math

DEG2RAD = math.pi / 180.0


class AgilexRobotNode(Node):
    """
    可通过参数配置：
      - joint_sub_topic  (订阅对方关节角)
      - end_pose_topic   (订阅末端位姿)
      - joint_pub_topic  (发布本机关节/夹爪状态)
      - fps              (发布频率 Hz，字符串形式传入也可)
      - can_port         (如 'can_right')
    """
    def __init__(self):
        super().__init__('agilex_robot_node')

        # -------- 声明参数（全部以字符串/通用类型声明，避免类型不匹配）--------
        self.declare_parameter('joint_sub_topic', '/joint_states')
        self.declare_parameter('end_pose_topic', '/end_pose')
        self.declare_parameter('joint_pub_topic', '/joint_state')
        self.declare_parameter('fps', 50)                # 支持字符串/数字
        self.declare_parameter('can_port', 'can_right')

        # 读取参数
        joint_topic_name = self._get_param_str('joint_sub_topic')
        endpose_topic_name = self._get_param_str('end_pose_topic')
        jointstate_topic_name = self._get_param_str('joint_pub_topic')
        self.fps = int(self.get_parameter('fps').value)
        can_port = self._get_param_str('can_port')

        # QoS
        qos = QoSProfile(depth=10)

        # 订阅：来自外部的关节目标
        self.expected_names = [f'joint{i + 1}' for i in range(7)]
        self.current_joint_positions = [0.0] * 7
        self.joint_positions_received = False

        self.sub_jointstate = self.create_subscription(
            JointState, joint_topic_name, self.joint_states_callback, qos
        )
        self.get_logger().info(f"[SUB] joint states: {joint_topic_name}")

        # 订阅：末端位姿（如不使用可忽略）
        self.latest_ee_pose = None
        self.ee_pose_received = False
        self.sub_end_pose = self.create_subscription(
            PoseStamped, endpose_topic_name, self.end_pose_callback, qos
        )
        self.get_logger().info(f"[SUB] EE pose: {endpose_topic_name}")

        # 发布：自身关节/夹爪状态
        self.pub_end_state = self.create_publisher(JointState, jointstate_topic_name, qos)
        self.get_logger().info(f"[PUB] joint state: {jointstate_topic_name}")

        self.msg = JointState()
        self.msg.name = [f'joint{i + 1}' for i in range(7)]

        # 计时器（发布频率）
        period = 1.0 / max(1.0, float(self.fps))
        self.timer = self.create_timer(period, self.publish_cb)
        self.get_logger().info(f"Publish FPS: {1.0/period:.2f} Hz")

        # 硬件接口
        self.piper = C_PiperInterface(can_name=can_port)
        self.piper.ConnectPort()
        self.piper.EnableArm(7)
        self.piper.GripperCtrl(0, 1000, 0x01, 0)
        self.get_logger().info(f"Connected to Piper on CAN: {can_port}")

        self._cb_count = 0
        self._ee_log_count = 0

    # ---------- 参数读取工具 ----------
    def _get_param_str(self, name: str) -> str:
        p = self.get_parameter(name)
        # rclpy 会把 launch 传来的值保持其 YAML 类型；统一转成 str
        return str(p.value) if p.type_ != Parameter.Type.NOT_SET else str(self.get_parameter(name).value)

    def _get_param_float(self, name: str, default: float) -> float:
        p = self.get_parameter(name)
        val = p.value if p.type_ != Parameter.Type.NOT_SET else default
        try:
            return float(val)
        except Exception:
            self.get_logger().warn(f"Param '{name}' value '{val}' not float, fallback to {default}")
            return float(default)

    # ---------- 回调 ----------
    def joint_states_callback(self, msg: JointState):
        """订阅对方 JointState：重排到 joint1..joint7 的顺序并下发到硬件"""
        if not msg.name or not msg.position:
            self.get_logger().warn("JointState is empty.")
            return

        name_to_idx = {n: i for i, n in enumerate(msg.name)}
        positions = []
        missing = []
        for n in self.expected_names:
            if n in name_to_idx and name_to_idx[n] < len(msg.position):
                positions.append(float(msg.position[name_to_idx[n]]))
            else:
                missing.append(n)

        if missing:
            self.get_logger().warn(f"Missing joints in message: {missing}. Have names={msg.name}")
            return

        self.current_joint_positions = positions
        self.joint_positions_received = True

        # 下发到控制器（对方传入是弧度；硬件接口是“度*1000”和“夹爪微弧度”）
        self.piper.MotionCtrl_2(0x01, 0x01, 100)
        self.piper.JointCtrl(
            int(positions[0] * 1e3 * 180.0 / math.pi),
            int(positions[1] * 1e3 * 180.0 / math.pi),
            int(positions[2] * 1e3 * 180.0 / math.pi),
            int(positions[3] * 1e3 * 180.0 / math.pi),
            int(positions[4] * 1e3 * 180.0 / math.pi),
            int(positions[5] * 1e3 * 180.0 / math.pi),
        )
        self.piper.GripperCtrl(abs(int(positions[6] * 1e6)), 1000, 0x01, 0)

    def end_pose_callback(self, msg: PoseStamped):
        """订阅末端位姿（如需可在此做 RPY 解析或其他处理）"""
        self.latest_ee_pose = msg
        self.ee_pose_received = True
        # 仅节流打印
        self._ee_log_count = (self._ee_log_count + 1) % 50
        if self._ee_log_count == 0:
            p = msg.pose.position
            self.get_logger().info(f"EE pose @ {msg.header.frame_id}: ({p.x:.3f},{p.y:.3f},{p.z:.3f})")

    def publish_cb(self):
        """按 fps 发布自身的关节/夹爪状态（单位按你底层定义；如需可改成 SI 单位）"""
        arm_joint = self.piper.GetArmJointMsgs().joint_state
        highspd = self.piper.GetArmHighSpdInfoMsgs()
        gripper = self.piper.GetArmGripperMsgs().gripper_state

        # 如需发布为 SI 单位（rad / rad/s），可在此处做缩放；此处保持原始缩放一致：
        j = [
            float(arm_joint.joint_1),
            float(arm_joint.joint_2),
            float(arm_joint.joint_3),
            float(arm_joint.joint_4),
            float(arm_joint.joint_5),
            float(arm_joint.joint_6),
            float(gripper.grippers_angle),
        ]
        v = [
            float(highspd.motor_1.motor_speed),
            float(highspd.motor_2.motor_speed),
            float(highspd.motor_3.motor_speed),
            float(highspd.motor_4.motor_speed),
            float(highspd.motor_5.motor_speed),
            float(highspd.motor_6.motor_speed),
            float(0.0),
        ]
        e = [float(0.0), float(0.0), float(0.0), float(0.0), float(0.0), float(0.0), float(gripper.grippers_effort)]

        self.msg.header.stamp = self.get_clock().now().to_msg()
        self.msg.position = j
        self.msg.velocity = v
        self.msg.effort = e
        self.pub_end_state.publish(self.msg)


def main():
    rclpy.init()
    node = AgilexRobotNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
