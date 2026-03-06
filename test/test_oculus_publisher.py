#!/usr/bin/env python3
"""
Test script: publish OculusControllers messages to test IK solver
Manual mode: press keys to move to next test point
Reads test points from JSON file and generates interpolated trajectories
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose
from common.msg import OculusControllers
import time
import math
import threading
import sys
import json
import os

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
        self.declare_parameter('json_path', '')  # Path to JSON file with test points
        self.declare_parameter('interpolate_count', 5)  # Number of points to interpolate between each keyframe
        self.declare_parameter('include_current', True)  # Whether to include current position as first point
        
        topic = self.get_parameter('topic').value
        json_path = self.get_parameter('json_path').value
        interpolate_count = self.get_parameter('interpolate_count').value
        include_current = self.get_parameter('include_current').value
        
        self.publisher_ = self.create_publisher(OculusControllers, topic, 10)
        
        # Load test points from JSON or use default calibration data
        if json_path and os.path.isfile(json_path):
            self.get_logger().info(f"Loading test points from: {json_path}")
            calibration_data = self.load_json_points(json_path)
        else:
            if json_path:
                self.get_logger().warn(f"JSON file not found: {json_path}, using default calibration data")
            calibration_data = self.get_default_calibration_data()
        
        # Build test points with optional current position
        keyframe_points = self.build_keyframe_points(calibration_data)
        
        if include_current:
            # Add current/initial position as first point
            current_point = {
                "name": "__CURRENT__",
                "x": 0.0,
                "y": 0.0,
                "z": 0.3,
                "q": euler_to_quaternion(0, 0, 0)
            }
            keyframe_points = [current_point] + keyframe_points
        
        # Keep permanent keyframe list; interpolation is generated on-demand via keyboard
        self.test_points = list(keyframe_points)
        self.interpolate_count = int(interpolate_count)
        self.temp_segment = None
        
        self.current_point_idx = 0
        self.publish_count = 0
        self.current_arm = "left"  # Current arm: "left" or "right"
        
        self.get_logger().info(
            f"\n=== OculusTestPublisher Manual Mode ===\n"
            f"  Topic: {topic}\n"
            f"  Keyframes: {len(keyframe_points)}\n"
            f"  Total Points: {len(self.test_points)}\n"
            f"  Default Interpolation Count: {interpolate_count}\n"
            f"  Available points:\n"
        )
        for i, pt in enumerate(self.test_points):
            marker = "   "
            self.get_logger().info(
                f"{marker}[{i:3d}] {pt['name']:20s}: pos({pt['x']:.4f}, {pt['y']:.4f}, {pt['z']:.4f})"
            )
        
        self.print_usage()
        
        # Start keyboard input thread
        self.input_thread = threading.Thread(target=self.keyboard_input_loop, daemon=True)
        self.input_thread.start()
        
        # Timer for periodic republishing of current point
        timer_period = 0.1  # 10 Hz publishing
        self.timer = self.create_timer(timer_period, self.timer_callback)
    
    @staticmethod
    def load_json_points(json_path):
        """Load calibration points from JSON file"""
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            return data
        except Exception as e:
            raise RuntimeError(f"Failed to load JSON: {e}")
    
    @staticmethod
    def get_default_calibration_data():
        """Default calibration data (fallback)"""
        return {
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
    
    @staticmethod
    def build_keyframe_points(calibration_data):
        """Convert calibration data to test point format"""
        test_points = []
        for name, data in calibration_data.items():
            xyz = data["xyz"]
            kps = data["kps"]  # kps can be "kps" or "rpy"
            roll, pitch, yaw = kps[0], kps[1], kps[2]
            q = euler_to_quaternion(roll, pitch, yaw)
            test_points.append({
                "name": name,
                "x": xyz[0],
                "y": xyz[1],
                "z": xyz[2],
                "q": q
            })
        return test_points
    
    @staticmethod
    def interpolate_pose(pose1, pose2, t):
        """
        Linear interpolation between two poses
        t: 0.0 = pose1, 1.0 = pose2
        """
        # Linear interpolation for position
        x = pose1['x'] * (1 - t) + pose2['x'] * t
        y = pose1['y'] * (1 - t) + pose2['y'] * t
        z = pose1['z'] * (1 - t) + pose2['z'] * t
        
        # Slerp for quaternion
        q = OculusTestPublisher.slerp_quaternion(pose1['q'], pose2['q'], t)
        
        return {"x": x, "y": y, "z": z, "q": q}
    
    @staticmethod
    def slerp_quaternion(q1, q2, t):
        """Spherical linear interpolation (SLERP) between two quaternions"""
        # q format: (qx, qy, qz, qw)
        q1_x, q1_y, q1_z, q1_w = q1
        q2_x, q2_y, q2_z, q2_w = q2
        
        # Compute dot product
        dot = q1_x * q2_x + q1_y * q2_y + q1_z * q2_z + q1_w * q2_w
        
        # If dot product is negative, negate one quaternion to take the shorter path
        if dot < 0.0:
            q2_x, q2_y, q2_z, q2_w = -q2_x, -q2_y, -q2_z, -q2_w
            dot = -dot
        
        # Clamp dot product to avoid numerical error
        dot = max(-1.0, min(1.0, dot))
        
        # If quaternions are very close, use linear interpolation
        if dot > 0.9995:
            # Linear interpolation
            qx = q1_x + t * (q2_x - q1_x)
            qy = q1_y + t * (q2_y - q1_y)
            qz = q1_z + t * (q2_z - q1_z)
            qw = q1_w + t * (q2_w - q1_w)
        else:
            # Standard SLERP
            theta = math.acos(dot)
            sin_theta = math.sin(theta)
            w1 = math.sin((1.0 - t) * theta) / sin_theta
            w2 = math.sin(t * theta) / sin_theta
            
            qx = w1 * q1_x + w2 * q2_x
            qy = w1 * q1_y + w2 * q2_y
            qz = w1 * q1_z + w2 * q2_z
            qw = w1 * q1_w + w2 * q2_w
        
        # Normalize
        norm = math.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
        if norm > 0:
            qx, qy, qz, qw = qx/norm, qy/norm, qz/norm, qw/norm
        
        return (qx, qy, qz, qw)
    
    @staticmethod
    def generate_interpolated_trajectory(keyframe_points, interpolate_count):
        """Generate interpolated trajectory between keyframes"""
        if not keyframe_points:
            return []
        
        trajectory = []
        
        for i, keyframe in enumerate(keyframe_points):
            # Add keyframe
            trajectory.append(keyframe)
            
            # Interpolate between this keyframe and next
            if i < len(keyframe_points) - 1:
                next_keyframe = keyframe_points[i + 1]
                
                # Generate interpolated points
                for j in range(1, interpolate_count + 1):
                    t = j / (interpolate_count + 1)  # t in (0, 1)
                    interp_pose = OculusTestPublisher.interpolate_pose(keyframe, next_keyframe, t)
                    interp_point = {
                        "name": f"_interp_{i}_{j}",
                        "x": interp_pose['x'],
                        "y": interp_pose['y'],
                        "z": interp_pose['z'],
                        "q": interp_pose['q']
                    }
                    trajectory.append(interp_point)
        
        return trajectory
    
    def generate_straight_line_trajectory(self, start_idx, end_idx, num_points=20):
        """
        Generate a straight-line trajectory between two points
        Inserts these points right after current point as a temporary segment.
        The segment is removed automatically after target is reached.
        """
        if self.temp_segment is not None:
            self.get_logger().warn("A temporary segment is already active. Reach target first.")
            return
        if start_idx < 0 or start_idx >= len(self.test_points):
            self.get_logger().warn(f"Invalid start index: {start_idx}")
            return
        if end_idx < 0 or end_idx >= len(self.test_points):
            self.get_logger().warn(f"Invalid end index: {end_idx}")
            return
        
        start_point = self.test_points[start_idx]
        end_point = self.test_points[end_idx]
        
        straight_line_points = []
        
        for i in range(num_points):
            t = i / (num_points - 1) if num_points > 1 else 0.0
            
            interp_pose = self.interpolate_pose(start_point, end_point, t)
            interp_point = {
                "name": f"_line_{start_idx}_to_{end_idx}_{i}",
                "x": interp_pose['x'],
                "y": interp_pose['y'],
                "z": interp_pose['z'],
                "q": interp_pose['q']
            }
            straight_line_points.append(interp_point)
        
        # Add an explicit target copy so we can remove all temporary points after reaching it
        straight_line_points.append({
            "name": f"_line_target_{start_idx}_to_{end_idx}",
            "x": end_point['x'],
            "y": end_point['y'],
            "z": end_point['z'],
            "q": end_point['q']
        })

        # Insert temporary segment right after current position
        insert_pos = self.current_point_idx + 1
        self.test_points[insert_pos:insert_pos] = straight_line_points
        seg_start = insert_pos
        seg_end = insert_pos + len(straight_line_points) - 1
        self.temp_segment = {
            "start": seg_start,
            "end": seg_end,
            "target_index": end_idx,
            "mode": "line"
        }
        
        self.get_logger().info(
            f"\n=== Generated Straight-Line Trajectory ===\n"
            f"  From: [{start_idx}] {start_point['name']}\n"
            f"  To:   [{end_idx}] {end_point['name']}\n"
            f"  Points: {num_points}\n"
            f"  Inserted after current index: {self.current_point_idx}\n"
            f"  Temp segment range: [{seg_start} ~ {seg_end}]\n"
        )

    def generate_normal_interpolation(self, start_idx, end_idx, num_points):
        """
        Generate normal interpolation segment between start and target.
        Behavior is same lifecycle as straight-line mode: insert after current, auto-discard at target.
        """
        if self.temp_segment is not None:
            self.get_logger().warn("A temporary segment is already active. Reach target first.")
            return
        if start_idx < 0 or start_idx >= len(self.test_points):
            self.get_logger().warn(f"Invalid start index: {start_idx}")
            return
        if end_idx < 0 or end_idx >= len(self.test_points):
            self.get_logger().warn(f"Invalid end index: {end_idx}")
            return

        start_point = self.test_points[start_idx]
        end_point = self.test_points[end_idx]
        interp_points = []

        for i in range(1, num_points + 1):
            t = i / (num_points + 1)
            interp_pose = self.interpolate_pose(start_point, end_point, t)
            interp_points.append({
                "name": f"_interp_tmp_{start_idx}_to_{end_idx}_{i}",
                "x": interp_pose['x'],
                "y": interp_pose['y'],
                "z": interp_pose['z'],
                "q": interp_pose['q']
            })

        interp_points.append({
            "name": f"_interp_target_{start_idx}_to_{end_idx}",
            "x": end_point['x'],
            "y": end_point['y'],
            "z": end_point['z'],
            "q": end_point['q']
        })

        insert_pos = self.current_point_idx + 1
        self.test_points[insert_pos:insert_pos] = interp_points
        seg_start = insert_pos
        seg_end = insert_pos + len(interp_points) - 1
        self.temp_segment = {
            "start": seg_start,
            "end": seg_end,
            "target_index": end_idx,
            "mode": "interp"
        }

        self.get_logger().info(
            f"\n=== Generated Normal Interpolation ===\n"
            f"  From: [{start_idx}] {start_point['name']}\n"
            f"  To:   [{end_idx}] {end_point['name']}\n"
            f"  Interpolation points: {num_points}\n"
            f"  Inserted after current index: {self.current_point_idx}\n"
            f"  Temp segment range: [{seg_start} ~ {seg_end}]\n"
        )

    def advance_point_index(self):
        """Advance point index and clear temporary segment after reaching its target."""
        if self.temp_segment and self.current_point_idx == self.temp_segment["end"]:
            seg_start = self.temp_segment["start"]
            seg_end = self.temp_segment["end"]
            target_index = self.temp_segment["target_index"]
            mode = self.temp_segment["mode"]
            del self.test_points[seg_start:seg_end + 1]
            self.temp_segment = None
            self.current_point_idx = max(0, min(target_index, len(self.test_points) - 1))
            self.get_logger().info(
                f"Temporary {mode} segment discarded after reaching target. "
                f"Current reset to [{self.current_point_idx}] {self.test_points[self.current_point_idx]['name']}"
            )

        self.current_point_idx = (self.current_point_idx + 1) % len(self.test_points)
    
    def print_usage(self):
        """Print usage instructions"""
        self.get_logger().info(
            "\n=== Instructions ===\n"
            "  Press SPACE or ENTER: Publish current point and move to next\n"
            "  Press 0-9: Jump to point index (or type multi-digit number)\n"
            "  Press 'a': Switch to LEFT arm\n"
            "  Press 's': Switch to RIGHT arm\n"
            "  Press 'd': Show current arm\n"
            "  Press 'p': Print current point\n"
            "  Press 'l': List all points\n"
            "  Press 'i': Create normal interpolation segment (temporary)\n"
            "  Press 'c': Create straight-line trajectory (between current and selected point)\n"
            "  Press 'q': Quit\n"
            f"  Current point: [{self.current_point_idx}] {self.test_points[self.current_point_idx]['name']}\n"
            f"  Current arm: {self.current_arm.upper()}\n"
        )

    def log_points(self):
        """Print all current points with markers."""
        self.get_logger().info(f"Test points (total: {len(self.test_points)}):")
        for i, pt in enumerate(self.test_points):
            prefix = " > " if i == self.current_point_idx else "   "
            marker = ""
            if pt['name'].startswith('_interp_tmp_') or pt['name'].startswith('_interp_target_'):
                marker = " *"
            if pt['name'].startswith('_line_'):
                marker = " -"
            self.get_logger().info(
                f"{prefix}[{i:3d}] {pt['name']:20s}: pos({pt['x']:.4f}, {pt['y']:.4f}, {pt['z']:.4f}){marker}"
            )

    def read_int_raw(self, prompt, default=None):
        """Read an integer in raw terminal mode. Enter accepts default, q cancels."""
        self.get_logger().info(prompt)
        buf = ""
        while True:
            ch = sys.stdin.read(1)
            if ch.isdigit():
                buf += ch
                self.get_logger().info(f"  Input: {buf}")
            elif ch == '\x08' or ch == '\x7f':
                buf = buf[:-1]
                self.get_logger().info(f"  Input: {buf}")
            elif ch == '\t' or ch == '\n' or ch == '\r':
                if buf:
                    return int(buf)
                if default is not None:
                    return int(default)
                self.get_logger().info("  Please input a number")
            elif ch == 'q' or ch == 'Q':
                self.get_logger().info("  Cancelled")
                return None

    def create_segment_from_current(self, mode):
        """Shared interactive creator for temporary segment from current point."""
        if mode == "interp":
            self.get_logger().info(
                f"\nCreate normal interpolation:\n"
                f"  From: [{self.current_point_idx}] {self.test_points[self.current_point_idx]['name']}"
            )
            default_points = self.interpolate_count
        else:
            self.get_logger().info(
                f"\nCreate straight-line trajectory:\n"
                f"  From: [{self.current_point_idx}] {self.test_points[self.current_point_idx]['name']}"
            )
            default_points = 20

        target_idx = self.read_int_raw("  Enter target point index (Enter to confirm, q to cancel):")
        if target_idx is None:
            return
        num_points = self.read_int_raw(
            f"  Enter points [{default_points}] (Enter to use default, q to cancel):",
            default=default_points
        )
        if num_points is None:
            return

        if mode == "interp":
            self.generate_normal_interpolation(self.current_point_idx, target_idx, num_points)
        else:
            self.generate_straight_line_trajectory(self.current_point_idx, target_idx, num_points)

    def create_segment_from_current_fallback(self, mode):
        """Shared line-buffered creator for temporary segment from current point."""
        if mode == "interp":
            title = "Create normal interpolation"
            default_points = self.interpolate_count
        else:
            title = "Create straight-line trajectory"
            default_points = 20

        target_str = input(
            f"{title} from [{self.current_point_idx}] {self.test_points[self.current_point_idx]['name']} to point index: "
        ).strip()
        if not target_str:
            self.get_logger().warn("Cancelled")
            return
        try:
            target_idx = int(target_str)
            num_points_str = input(f"Number of points [{default_points}]: ").strip()
            num_points = int(num_points_str) if num_points_str else default_points
            if mode == "interp":
                self.generate_normal_interpolation(self.current_point_idx, target_idx, num_points)
            else:
                self.generate_straight_line_trajectory(self.current_point_idx, target_idx, num_points)
        except ValueError:
            self.get_logger().warn("Invalid input")
    
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
                input_buffer = ""
                while True:
                    ch = sys.stdin.read(1)
                    
                    if ch == ' ' or ch == '\n' or ch == '\r':
                        # Publish current point
                        self.publish_current_point()
                        # Move to next point
                        self.advance_point_index()
                        self.get_logger().info(
                            f"Next point: [{self.current_point_idx}] "
                            f"{self.test_points[self.current_point_idx]['name']}"
                        )
                        input_buffer = ""
                    elif ch == '\x08' or ch == '\x7f':  # Backspace
                        input_buffer = input_buffer[:-1]
                        self.get_logger().info(f"Input: {input_buffer}")
                    elif ch.isdigit():
                        # Accumulate digits for multi-digit index
                        input_buffer += ch
                        self.get_logger().info(f"Input: {input_buffer}")
                        
                        # Try to jump to index if it matches a valid range
                        idx = int(input_buffer)
                        if idx >= len(self.test_points):
                            self.get_logger().warn(f"Invalid point index: {idx} (max: {len(self.test_points)-1})")
                            input_buffer = ""
                    elif ch == 'a' or ch == 'A':
                        self.current_arm = "left"
                        self.get_logger().info(f"Switched to LEFT arm")
                        input_buffer = ""
                    elif ch == 's' or ch == 'S':
                        self.current_arm = "right"
                        self.get_logger().info(f"Switched to RIGHT arm")
                        input_buffer = ""
                    elif ch == 'd' or ch == 'D':
                        self.get_logger().info(f"Current arm: {self.current_arm.upper()}")
                        input_buffer = ""
                    elif ch == 'p' or ch == 'P':
                        pt = self.test_points[self.current_point_idx]
                        self.get_logger().info(
                            f"Current point: [{self.current_point_idx}] {pt['name']}\n"
                            f"  pos: ({pt['x']:.4f}, {pt['y']:.4f}, {pt['z']:.4f})\n"
                            f"  quaternion: {pt['q']}\n"
                            f"  arm: {self.current_arm.upper()}"
                        )
                        input_buffer = ""
                    elif ch == 'l' or ch == 'L':
                        self.log_points()
                        input_buffer = ""
                    elif ch == 'i' or ch == 'I':
                        self.create_segment_from_current("interp")
                        input_buffer = ""
                    elif ch == 'c' or ch == 'C':
                        self.create_segment_from_current("line")
                        input_buffer = ""
                    elif ch == 'q' or ch == 'Q':
                        self.get_logger().info("Quitting...")
                        raise KeyboardInterrupt
                    elif ch == '\t':  # Tab: confirm multi-digit input
                        if input_buffer:
                            idx = int(input_buffer)
                            if 0 <= idx < len(self.test_points):
                                self.current_point_idx = idx
                                self.get_logger().info(
                                    f"Jumped to point: [{self.current_point_idx}] "
                                    f"{self.test_points[self.current_point_idx]['name']}"
                                )
                            else:
                                self.get_logger().warn(f"Invalid point index: {idx} (max: {len(self.test_points)-1})")
                            input_buffer = ""
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        except ImportError:
            # Fallback for systems without tty support (e.g., Windows)
            self.get_logger().warn("tty not available, using standard input (line-buffered)")
            while True:
                try:
                    cmd = input("Command (SPACE=next, 0-999=jump, a/s=arm, p=print, l=list, q=quit): ").strip()
                    if cmd == '' or cmd == ' ':
                        self.publish_current_point()
                        self.advance_point_index()
                        self.get_logger().info(
                            f"Next point: [{self.current_point_idx}] "
                            f"{self.test_points[self.current_point_idx]['name']}"
                        )
                    elif cmd.isdigit():
                        idx = int(cmd)
                        if 0 <= idx < len(self.test_points):
                            self.current_point_idx = idx
                            self.get_logger().info(
                                f"Jumped to point: [{self.current_point_idx}] "
                                f"{self.test_points[self.current_point_idx]['name']}"
                            )
                        else:
                            self.get_logger().warn(f"Invalid point index: {idx} (max: {len(self.test_points)-1})")
                    elif cmd == 'a':
                        self.current_arm = "left"
                        self.get_logger().info(f"Switched to LEFT arm")
                    elif cmd == 's':
                        self.current_arm = "right"
                        self.get_logger().info(f"Switched to RIGHT arm")
                    elif cmd == 'd':
                        self.get_logger().info(f"Current arm: {self.current_arm.upper()}")
                    elif cmd == 'p':
                        pt = self.test_points[self.current_point_idx]
                        self.get_logger().info(
                            f"Current point: [{self.current_point_idx}] {pt['name']}\n"
                            f"  pos: ({pt['x']:.3f}, {pt['y']:.3f}, {pt['z']:.3f})\n"
                            f"  arm: {self.current_arm.upper()}"
                        )
                    elif cmd == 'l':
                        self.log_points()
                    elif cmd == 'i':
                        self.create_segment_from_current_fallback("interp")
                    elif cmd == 'c':
                        self.create_segment_from_current_fallback("line")
                    elif cmd == 'q':
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
