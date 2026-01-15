#!/usr/bin/env python3
"""
./replay_joint_states.py /path/to/your_file.jsonl --rate 20.0 --no-loop
"""
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import JointState
import json
import argparse
import copy
import os
from common_utils.ros2_qos import reliable_qos


class JointStatesReplayer(Node):
    def __init__(self, jsonl_path: str, rate_hz: float = 30.0, loop: bool = True):
        super().__init__('joint_states_replayer')
        self.pub_right = self.create_publisher(JointState, '/joint_cmd_right', reliable_qos)
        self.pub_left = self.create_publisher(JointState, '/joint_cmd_left', reliable_qos)

        self.rate_hz = rate_hz
        self.loop = loop

        self.right_list = []  # list of dicts: {stamp_ns, name, position, velocity, effort}
        self.left_list = []

        self._load_jsonl(jsonl_path)
        # self._load_csv(jsonl_path)

        if len(self.right_list) == 0 and len(self.left_list) == 0:
            self.get_logger().error("No joint states found in file.")
            raise RuntimeError("No joint states found in file.")

        self.index = 0
        timer_period = 1.0 / rate_hz
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.get_logger().info(f"Started replayer: file={jsonl_path}, rate={rate_hz} Hz, loop={loop}")

    def _load_jsonl(self, path: str):
        if not os.path.exists(path):
            self.get_logger().error(f"File does not exist: {path}")
            raise FileNotFoundError(path)
        with open(path, 'r', encoding='utf-8') as f:
            for line_no, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    js = json.loads(line)
                except Exception as e:
                    self.get_logger().warning(f"Skipping invalid JSON at line {line_no+1}: {e}")
                    continue
                # js expected to have 'joints' list
                joints = js.get('joints', [])
                for j in joints:
                    topic = j.get('topic', '')
                    entry = {
                        'stamp_ns': j.get('stamp_ns', 0),
                        'name': j.get('name', []),
                        'position': j.get('position', []),
                        'velocity': j.get('velocity', []),
                        'effort': j.get('effort', [])
                    }
                    if topic.endswith('right') or topic == '/joint_states_right':
                        self.right_list.append(entry)
                    elif topic.endswith('left') or topic == '/joint_states_left':
                        self.left_list.append(entry)
        self.get_logger().info(f"Loaded {len(self.right_list)} right entries and {len(self.left_list)} left entries")

    def _load_csv(self, path: str):
        if not os.path.exists(path):
            self.get_logger().error(f"File does not exist: {path}")
            raise FileNotFoundError(path)
        
        with open(path, 'r', encoding='utf-8') as f:
            for line_no, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    # 按逗号分割字符串并转换为浮点数
                    values = [float(x) for x in line.split(',')]
                except Exception as e:
                    self.get_logger().warning(f"Skipping invalid data at line {line_no+1}: {e}")
                    continue
                
                # 假设CSV每行包含7个关节数据
                if len(values) != 7:
                    self.get_logger().warning(f"Skipping line {line_no+1}: expected 7 values, got {len(values)}")
                    continue
                
                # 创建关节数据条目
                entry = {
                    'stamp_ns': 0,  # CSV中没有时间戳，设为0或行号
                    'name': [f'joint{i+1}' for i in range(7)],  # 生成关节名称
                    'position': values,  # 使用CSV中的位置数据
                    'velocity': [],  # CSV中没有速度数据
                    'effort': []     # CSV中没有力数据
                }
                
                self.right_list.append(entry)
        
        self.get_logger().info(f"Loaded {len(self.right_list)} right entries and {len(self.left_list)} left entries from CSV")

    def _make_jointstate_msg(self, entry: dict) -> JointState:
        msg = JointState()
        # set stamp if available
        stamp_ns = int(entry.get('stamp_ns', 0))
        try:
            # Time requires non-negative integer nanoseconds
            if stamp_ns > 0:
                msg.header.stamp = Time(nanoseconds=stamp_ns).to_msg()
        except Exception:
            # ignore stamp if invalid
            pass
        msg.name = entry.get('name', [])
        msg.position = [float(x) for x in entry.get('position', [])]
        msg.velocity = [float(x) for x in entry.get('velocity', [])]
        msg.effort = [float(x) for x in entry.get('effort', [])]
        return msg

    def timer_callback(self):
        # If we have mismatched lengths, publish available ones from their own lists by index modulo length
        if len(self.right_list) > 0:
            idx_r = self.index % len(self.right_list)
            msg_r = self._make_jointstate_msg(self.right_list[idx_r])
            self.pub_right.publish(msg_r)
        if len(self.left_list) > 0:
            idx_l = self.index % len(self.left_list)
            msg_l = self._make_jointstate_msg(self.left_list[idx_l])
            self.pub_left.publish(msg_l)

        self.index += 1

        # If not looping and we've reached the longest list end, stop timer and optionally shutdown node
        if not self.loop:
            # Determine whether we've published the last frame for both lists
            done_r = (len(self.right_list) == 0) or (self.index >= len(self.right_list))
            done_l = (len(self.left_list) == 0) or (self.index >= len(self.left_list))
            if done_r and done_l:
                self.get_logger().info("Finished replay (non-loop). Shutting down node.")
                self.timer.cancel()
                # optionally call destroy_node / rclpy.shutdown handled in main

def main():
    parser = argparse.ArgumentParser(description='Replay joint_states from a .jsonl file')
    parser.add_argument('jsonl', help='Path to the .jsonl file')
    parser.add_argument('--rate', type=float, default=30.0, help='Publish frequency in Hz (default 30.0)')
    parser.add_argument('--no-loop', dest='loop', action='store_false', help='Do not loop; stop after end of file')
    args = parser.parse_args()

    rclpy.init()
    try:
        node = JointStatesReplayer(args.jsonl, rate_hz=args.rate, loop=args.loop)
    except Exception as e:
        print(f"Failed to create node: {e}")
        rclpy.shutdown()
        return

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
