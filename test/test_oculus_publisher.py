#!/usr/bin/env python3
"""
Test script: publish OculusControllers messages to test IK solver
Manual mode: press keys to move to next test point
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose
from common.msg import OculusControllers
import time
import math
import threading
import sys

def euler_to_quaternion(roll, pitch, yaw):
    """Convert Euler angles (roll, pitch, yaw) to quaternion (qx, qy, qz, qw)"""
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    
    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    
    return (qx, qy, qz, qw)

class OculusTestPublisher(Node):
    def __init__(self):
        super().__init__('oculus_test_publisher')
        
        self.declare_parameter('topic', '/oculus_controllers')
        
        topic = self.get_parameter('topic').value
        
        self.publisher_ = self.create_publisher(OculusControllers, topic, 10)
        
        # Test points from calibration data
        # Format: xyz = [x, y, z], kps = [roll, pitch, yaw] (Euler angles)
        calibration_data = {
            "R0_D00": {"xyz": [-0.6147998051129465, -2.351929862432055e-16, 0.0004895233220600577], 
                      "kps": [-4.804530767879343e-13, -1.5700000948921882, 4.808357812363919e-13]},
            "R0_D01": {"xyz": [-0.45125230637126557, -8.477459303201747e-8, 0.08515826183737765], 
                      "kps": [-3.141591785522368, -0.4497864164042696, 3.1415923713531213]},
            "R0_D02": {"xyz": [-0.27282362352783984, 1.6697353407972146e-8, 0.41234866355200417], 
                      "kps": [-3.1415906524340436, -1.4015066864125545, 3.141590782118139]},
            "R0_D03": {"xyz": [-0.39524097330177455, 0.05409589053460992, 0.2540315705346861], 
                      "kps": [-3.141518514061995, -1.556819012666717, 3.1415146215671346]},
            "R0_D04": {"xyz": [-0.44578962615977097, 0.054095433294251885, 0.25403153838622755], 
                      "kps": [-3.141548230715841, -1.55682043972031, 3.1415460360285947]},
            "R0_D05": {"xyz": [-0.36730348604103935, 0.3307191081391618, -0.009154202166839142], 
                      "kps": [-2.730700654672292, -1.4994322170295515, 2.658265627578435]},
            "R0_D06": {"xyz": [-0.42891638431415424, 0.3307191356440027, -0.009153206139059783], 
                      "kps": [-2.73041643192978, -1.4994253790823693, 2.657980282503626]},
        }
        
        # Convert to test points with quaternions
        self.test_points = []
        for name, data in calibration_data.items():
            xyz = data["xyz"]
            roll, pitch, yaw = data["kps"]
            q = euler_to_quaternion(roll, pitch, yaw)
            self.test_points.append({
                "name": name,
                "x": xyz[0],
                "y": xyz[1],
                "z": xyz[2],
                "q": q
            })
        
        self.current_point_idx = 0
        self.publish_count = 0
        self.current_arm = "left"  # Current arm: "left" or "right"
        
        self.get_logger().info(
            f"\n=== OculusTestPublisher Manual Mode ===\n"
            f"  Topic: {topic}\n"
            f"  Total Points: {len(self.test_points)}\n"
            f"  Available points:\n"
        )
        for i, pt in enumerate(self.test_points):
            self.get_logger().info(f"    [{i}] {pt['name']}: pos({pt['x']:.4f}, {pt['y']:.4f}, {pt['z']:.4f})")
        
        self.print_usage()
        
        # Start keyboard input thread
        self.input_thread = threading.Thread(target=self.keyboard_input_loop, daemon=True)
        self.input_thread.start()
        
        # Timer for periodic republishing of current point
        timer_period = 0.1  # 10 Hz publishing
        self.timer = self.create_timer(timer_period, self.timer_callback)
    
    def print_usage(self):
        """Print usage instructions"""
        self.get_logger().info(
            "\n=== Instructions ===\n"
            "  Press SPACE or ENTER: Publish current point and move to next\n"
            "  Press 0-6: Jump to point index\n"
            "  Press 'a': Switch to LEFT arm\n"
            "  Press 's': Switch to RIGHT arm\n"
            "  Press 'd': Show current arm\n"
            "  Press 'p': Print current point\n"
            "  Press 'l': List all points\n"
            "  Press 'q': Quit\n"
            f"  Current point: [{self.current_point_idx}] {self.test_points[self.current_point_idx]['name']}\n"
            f"  Current arm: {self.current_arm.upper()}\n"
        )
    
    def keyboard_input_loop(self):
        """Handle keyboard input in separate thread"""
        try:
            import tty
            import termios
            
            # Set terminal to raw mode for immediate input
            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)
            
            try:
                tty.setraw(fd)
                while True:
                    ch = sys.stdin.read(1)
                    
                    if ch == ' ' or ch == '\n' or ch == '\r':
                        # Publish current point
                        self.publish_current_point()
                        # Move to next point
                        self.current_point_idx = (self.current_point_idx + 1) % len(self.test_points)
                        self.get_logger().info(
                            f"Next point: [{self.current_point_idx}] "
                            f"{self.test_points[self.current_point_idx]['name']}"
                        )
                    elif ch.isdigit():
                        idx = int(ch)
                        if 0 <= idx < len(self.test_points):
                            self.current_point_idx = idx
                            self.get_logger().info(
                                f"Jumped to point: [{self.current_point_idx}] "
                                f"{self.test_points[self.current_point_idx]['name']}"
                            )
                        else:
                            self.get_logger().warn(f"Invalid point index: {idx}")
                    elif ch == 'a' or ch == 'A':
                        self.current_arm = "left"
                        self.get_logger().info(f"Switched to LEFT arm")
                    elif ch == 's' or ch == 'S':
                        self.current_arm = "right"
                        self.get_logger().info(f"Switched to RIGHT arm")
                    elif ch == 'd' or ch == 'D':
                        self.get_logger().info(f"Current arm: {self.current_arm.upper()}")
                    elif ch == 'p' or ch == 'P':
                        pt = self.test_points[self.current_point_idx]
                        self.get_logger().info(
                            f"Current point: [{self.current_point_idx}] {pt['name']}\n"
                            f"  pos: ({pt['x']:.4f}, {pt['y']:.4f}, {pt['z']:.4f})\n"
                            f"  quaternion: {pt['q']}\n"
                            f"  arm: {self.current_arm.upper()}"
                        )
                    elif ch == 'l' or ch == 'L':
                        self.get_logger().info(f"Test points:")
                        for i, pt in enumerate(self.test_points):
                            prefix = " > " if i == self.current_point_idx else "   "
                            self.get_logger().info(
                                f"{prefix}[{i}] {pt['name']}: pos({pt['x']:.4f}, {pt['y']:.4f}, {pt['z']:.4f})"
                            )
                    elif ch == 'q' or ch == 'Q':
                        self.get_logger().info("Quitting...")
                        raise KeyboardInterrupt
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        except ImportError:
            # Fallback for systems without tty support (e.g., Windows)
            self.get_logger().warn("tty not available, using standard input (line-buffered)")
            while True:
                try:
                    ch = input().strip()
                    if ch == '' or ch == ' ':
                        self.publish_current_point()
                        self.current_point_idx = (self.current_point_idx + 1) % len(self.test_points)
                        self.get_logger().info(
                            f"Next point: [{self.current_point_idx}] "
                            f"{self.test_points[self.current_point_idx]['name']}"
                        )
                    elif ch.isdigit():
                        idx = int(ch)
                        if 0 <= idx < len(self.test_points):
                            self.current_point_idx = idx
                            self.get_logger().info(
                                f"Jumped to point: [{self.current_point_idx}] "
                                f"{self.test_points[self.current_point_idx]['name']}"
                            )
                    elif ch == 'a':
                        self.current_arm = "left"
                        self.get_logger().info(f"Switched to LEFT arm")
                    elif ch == 's':
                        self.current_arm = "right"
                        self.get_logger().info(f"Switched to RIGHT arm")
                    elif ch == 'd':
                        self.get_logger().info(f"Current arm: {self.current_arm.upper()}")
                    elif ch == 'p':
                        pt = self.test_points[self.current_point_idx]
                        self.get_logger().info(
                            f"Current point: [{self.current_point_idx}] {pt['name']}\n"
                            f"  pos: ({pt['x']:.3f}, {pt['y']:.3f}, {pt['z']:.3f})\n"
                            f"  arm: {self.current_arm.upper()}"
                        )
                    elif ch == 'q':
                        raise KeyboardInterrupt
                except EOFError:
                    break
    
    def publish_current_point(self):
        """Publish the current test point"""
        pt = self.test_points[self.current_point_idx]
        msg = OculusControllers()
        
        if self.current_arm == "left":
            msg.left_valid = True
            msg.left_pose = self.create_pose(pt['x'], pt['y'], pt['z'], pt['q'])
            msg.right_valid = False
        else:  # right
            msg.left_valid = False
            msg.right_valid = True
            msg.right_pose = self.create_pose(pt['x'], pt['y'], pt['z'], pt['q'])
        
        self.publisher_.publish(msg)
        self.publish_count += 1
        
        self.get_logger().info(
            f"Published [{self.current_point_idx}] {pt['name']}: "
            f"pos({pt['x']:.4f}, {pt['y']:.4f}, {pt['z']:.4f}) "
            f"[arm={self.current_arm.upper()}, count={self.publish_count}]"
        )
    
    def create_pose(self, x, y, z, q):
        """Create geometry_msgs.Pose from coordinates"""
        pose = Pose()
        pose.position.x = float(x)
        pose.position.y = float(y)
        pose.position.z = float(z)
        pose.orientation.x = float(q[0])
        pose.orientation.y = float(q[1])
        pose.orientation.z = float(q[2])
        pose.orientation.w = float(q[3])
        return pose
    
    def timer_callback(self):
        """Periodically republish current point (for continuous stream)"""
        pt = self.test_points[self.current_point_idx]
        msg = OculusControllers()
        
        if self.current_arm == "left":
            msg.left_valid = True
            msg.left_pose = self.create_pose(pt['x'], pt['y'], pt['z'], pt['q'])
            msg.right_valid = False
        else:  # right
            msg.left_valid = False
            msg.right_valid = True
            msg.right_pose = self.create_pose(pt['x'], pt['y'], pt['z'], pt['q'])
        
        self.publisher_.publish(msg)
    
    def destroy_node(self):
        self.get_logger().info(
            f"OculusTestPublisher stopped. "
            f"Total publishes: {self.publish_count}"
        )
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = OculusTestPublisher()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down...")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
