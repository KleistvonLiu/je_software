import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Float32MultiArray, Header
import numpy as np
import time

class TestPublisher(Node):
    def __init__(self):
        super().__init__('test_publisher')
        # 4路彩色和4路深度相机
        self.color_pubs = [self.create_publisher(Image, f'cam{i}', 10) for i in range(4)]
        self.depth_pubs = [self.create_publisher(Image, f'dep{i}', 10) for i in range(4)]
        self.joint_pub = self.create_publisher(JointState, 'joint', 10)
        self.tactile_pub = self.create_publisher(Float32MultiArray, 'tactile', 10)
        self.timer = self.create_timer(1.0 / 30, self.publish_all)  # 30Hz
        self.frame_id = 0
        # 只生成一张彩色和一张深度图片，所有帧复用（低分辨率 160x120）
        self.img_height = 120
        self.img_width = 160
        arr = (np.random.rand(self.img_height, self.img_width, 3) * 255).astype(np.uint8)
        self.arr_bytes = arr.tobytes()
        depth_arr = (np.random.rand(self.img_height, self.img_width, 3) * 255).astype(np.uint8)
        self.depth_bytes = depth_arr.tobytes()

    def publish_all(self):
        start = time.time()
        t = start
        t_sec = int(t)
        t_nsec = int((t % 1) * 1e9)
        # 4路彩色
        for i in range(4):
            color_msg = Image()
            color_msg.header = Header()
            color_msg.header.stamp.sec = t_sec
            color_msg.header.stamp.nanosec = t_nsec
            color_msg.height = self.img_height
            color_msg.width = self.img_width
            color_msg.encoding = 'bgr8'
            color_msg.is_bigendian = 0
            color_msg.step = self.img_width * 3
            color_msg.data = self.arr_bytes
            self.color_pubs[i].publish(color_msg)
        # 4路深度
        for i in range(4):
            depth_msg = Image()
            depth_msg.header = Header()
            depth_msg.header.stamp.sec = t_sec
            depth_msg.header.stamp.nanosec = t_nsec
            depth_msg.height = self.img_height
            depth_msg.width = self.img_width
            depth_msg.encoding = 'bgr8'
            depth_msg.is_bigendian = 0
            depth_msg.step = self.img_width * 3
            depth_msg.data = self.depth_bytes
            self.depth_pubs[i].publish(depth_msg)
        # joint
        joint_msg = JointState()
        joint_msg.header = Header()
        joint_msg.header.stamp.sec = t_sec
        joint_msg.header.stamp.nanosec = t_nsec
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
