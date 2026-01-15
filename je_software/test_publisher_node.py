import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Float32MultiArray, Header
from cv_bridge import CvBridge
from common_utils.ros2_qos import reliable_qos
import numpy as np
import time


class TestPublisher(Node):
    def __init__(self):
        super().__init__('test_publisher')
        # Use the project's reliable_qos to match the manager's subscriptions
        # 4路彩色和4路深度相机
        self.color_pubs = [self.create_publisher(Image, f'cam{i}', reliable_qos) for i in range(4)]
        self.depth_pubs = [self.create_publisher(Image, f'dep{i}', reliable_qos) for i in range(4)]
        self.joint_pub = self.create_publisher(JointState, 'joint', 10)
        self.tactile_pub = self.create_publisher(Float32MultiArray, 'tactile', 10)
        self.timer = self.create_timer(1.0 / 30, self.publish_all)  # 30Hz
        self.frame_id = 0
        # 只生成一张彩色和一张深度图片（保留为 numpy 数组，使用 CvBridge 发布）
        self.img_height = 480
        self.img_width = 640
        arr = (np.random.rand(self.img_height, self.img_width, 3) * 255).astype(np.uint8)
        self.arr = arr  # keep numpy array
        # depth: create a single-channel uint16 depth image (16UC1) to mimic real depth topic
        depth_u16 = (np.random.rand(self.img_height, self.img_width) * 65535).astype(np.uint16)
        self.depth_arr = depth_u16

        # CvBridge for creating sensor_msgs/Image consistently with real cameras
        self.bridge = CvBridge()

    def publish_all(self):
        start = time.time()
        # use ROS clock stamp for messages
        t_msg = self.get_clock().now().to_msg()

        # 4路彩色 — convert numpy -> Image via CvBridge
        for i in range(4):
            color_msg = self.bridge.cv2_to_imgmsg(self.arr, encoding='bgr8')
            color_msg.header.stamp = t_msg
            color_msg.header.frame_id = f'cam{i}'
            self.color_pubs[i].publish(color_msg)

        # 4路深度 — publish as bgr8 to match test behavior (manager handles encoding)
        for i in range(4):
            # publish depth as 16UC1 (uint16 single-channel)
            depth_msg = self.bridge.cv2_to_imgmsg(self.depth_arr, encoding='16UC1')
            depth_msg.header.stamp = t_msg
            depth_msg.header.frame_id = f'dep{i}'
            self.depth_pubs[i].publish(depth_msg)

        # joint
        joint_msg = JointState()
        joint_msg.header.stamp = t_msg
        joint_msg.name = ['joint1', 'joint2']
        joint_msg.position = [float(self.frame_id), float(self.frame_id + 1)]
        joint_msg.velocity = [0.1, 0.2]
        joint_msg.effort = [0.0, 0.0]
        self.joint_pub.publish(joint_msg)
        # tactile
        tactile_msg = Float32MultiArray()
        tactile_msg.data = [0.1, 0.2, 0.3, 0.4]
        self.tactile_pub.publish(tactile_msg)
        self.frame_id += 1
        end = time.time()
        if self.frame_id % 30 == 0:
            print(f"publish_all duration: {end - start:.4f} seconds")
        # self.get_logger().info(f"Published frame {self.frame_id}")  # Commented out to improve performance


def main(args=None):
    rclpy.init(args=args)
    node = TestPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
