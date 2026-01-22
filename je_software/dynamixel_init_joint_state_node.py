#!/usr/bin/env python3
import ast
import json
import os
import sys
import threading
import time
from typing import List, Optional, Tuple

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

from common.msg import OculusInitJointState
from je_software.motors import Motor, MotorNormMode
from je_software.motors.dynamixel import DynamixelMotorsBus


class DynamixelInitJointStateNode(Node):
    def __init__(self):
        super().__init__("dynamixel_init_joint_state")

        self.declare_parameter("left_port", "/dev/ttyUSB0")
        self.declare_parameter("right_port", "/dev/ttyUSB1")
        self.declare_parameter("left_enabled", True)
        self.declare_parameter("right_enabled", True)
        self.declare_parameter("left_baudrate", 1000000)
        self.declare_parameter("right_baudrate", 1000000)
        self.declare_parameter("left_ids", "[1,2,3,4,5,6,7,8]")
        self.declare_parameter("right_ids", "[1,2,3,4,5,6,7,8]")
        self.declare_parameter("left_signs", "[1]")
        self.declare_parameter("right_signs", "[1]")
        self.declare_parameter(
            "joint_names",
            '["joint1","joint2","joint3","joint4","joint5","joint6","joint7"]',
        )
        self.declare_parameter("motor_model", "xl330-m077")
        self.declare_parameter("left_models", '["xl330-m077"]')
        self.declare_parameter("right_models", '["xl330-m077"]')
        self.declare_parameter("position_scale", 0.087890625)
        self.declare_parameter("left_gripper_min", 0.0)
        self.declare_parameter("left_gripper_max", 1.0)
        self.declare_parameter("right_gripper_min", 0.0)
        self.declare_parameter("right_gripper_max", 1.0)
        self.declare_parameter("zero_on_start", False)
        self.declare_parameter(
            "zero_file",
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "config",
                "dynamixel_zero_offsets.json",
            ),
        )
        self.declare_parameter("print_positions", False)
        self.declare_parameter("print_period", 0.1)
        self.declare_parameter("fps", 30.0)
        self.declare_parameter("publish_topic", "/oculus_init_joint_state")
        self.declare_parameter("frame_id", "")

        self._left_port = str(self.get_parameter("left_port").value)
        self._right_port = str(self.get_parameter("right_port").value)
        self._left_enabled = bool(self.get_parameter("left_enabled").value)
        self._right_enabled = bool(self.get_parameter("right_enabled").value)
        self._left_baudrate = int(self.get_parameter("left_baudrate").value)
        self._right_baudrate = int(self.get_parameter("right_baudrate").value)
        self._left_ids = self._parse_int_list(self.get_parameter("left_ids").value, "left_ids")
        self._right_ids = self._parse_int_list(self.get_parameter("right_ids").value, "right_ids")
        self._left_signs = self._parse_float_list(
            self.get_parameter("left_signs").value, "left_signs"
        )
        self._right_signs = self._parse_float_list(
            self.get_parameter("right_signs").value, "right_signs"
        )
        self._joint_names = self._parse_str_list(
            self.get_parameter("joint_names").value, "joint_names"
        )
        self._motor_model = str(self.get_parameter("motor_model").value)
        self._left_models = self._parse_str_list(
            self.get_parameter("left_models").value, "left_models"
        )
        self._right_models = self._parse_str_list(
            self.get_parameter("right_models").value, "right_models"
        )
        self._position_scale = float(self.get_parameter("position_scale").value)
        self._left_gripper_min = float(self.get_parameter("left_gripper_min").value)
        self._left_gripper_max = float(self.get_parameter("left_gripper_max").value)
        self._right_gripper_min = float(self.get_parameter("right_gripper_min").value)
        self._right_gripper_max = float(self.get_parameter("right_gripper_max").value)
        self._zero_on_start = bool(self.get_parameter("zero_on_start").value)
        self._zero_file = os.path.expanduser(
            str(self.get_parameter("zero_file").value)
        )
        self._print_positions = bool(self.get_parameter("print_positions").value)
        self._print_period = max(0.0, float(self.get_parameter("print_period").value))
        self._period_s = 1.0 / max(1.0, float(self.get_parameter("fps").value))
        self._topic = str(self.get_parameter("publish_topic").value)
        self._frame_id = str(self.get_parameter("frame_id").value)
        self._last_print_time = 0.0
        self._zero_offsets: dict[str, dict[str, float]] = {"left": {}, "right": {}}
        self._left_offsets: dict[str, float] = {}
        self._right_offsets: dict[str, float] = {}

        self._left_signs = self._resolve_signs(self._left_signs, "left_signs")
        self._right_signs = self._resolve_signs(self._right_signs, "right_signs")
        self._load_zero_file(allow_missing=self._zero_on_start)

        self._left_bus = self._init_bus(
            self._left_port,
            self._left_ids,
            self._left_models,
            self._left_baudrate,
            self._left_enabled,
            "left",
        )
        self._right_bus = self._init_bus(
            self._right_port,
            self._right_ids,
            self._right_models,
            self._right_baudrate,
            self._right_enabled,
            "right",
        )

        if self._zero_on_start:
            self._ensure_zero_file_exists()

        self._publisher = self.create_publisher(OculusInitJointState, self._topic, 10)
        self.get_logger().info(
            f"Publishing to {self._topic} at {1.0 / self._period_s:.2f} Hz"
        )

        self._stop_event = threading.Event()
        self._first_publish = True
        self._thread = threading.Thread(target=self._action_loop, daemon=True)
        self._thread.start()

    def _parse_list_param(self, value, label: str) -> List:
        if isinstance(value, list):
            return value
        if isinstance(value, tuple):
            return list(value)
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return []
            try:
                parsed = ast.literal_eval(text)
            except (SyntaxError, ValueError):
                parsed = None
            if isinstance(parsed, list):
                return parsed
            if "," in text:
                return [item.strip() for item in text.split(",") if item.strip()]
        self.get_logger().warn(f"Parameter {label} expected list, got {type(value).__name__}")
        return []

    def _parse_int_list(self, value, label: str) -> List[int]:
        items = self._parse_list_param(value, label)
        out = []
        for item in items:
            try:
                out.append(int(item))
            except (TypeError, ValueError):
                self.get_logger().warn(f"Parameter {label} has non-int value: {item}")
        return out

    def _parse_str_list(self, value, label: str) -> List[str]:
        items = self._parse_list_param(value, label)
        return [str(item) for item in items]

    def _parse_float_list(self, value, label: str) -> List[float]:
        items = self._parse_list_param(value, label)
        out = []
        for item in items:
            try:
                out.append(float(item))
            except (TypeError, ValueError):
                self.get_logger().warn(f"Parameter {label} has non-float value: {item}")
        return out

    def _resolve_models(self, models: List[str], label: str) -> Optional[List[str]]:
        if not models:
            return [self._motor_model] * 8
        if len(models) == 1:
            self.get_logger().info(f"{label} models has 1 entry; applying to all motors.")
            return models * 8
        if len(models) != 8:
            self.get_logger().error(
                f"{label} models length is {len(models)}; expected 8 (7 joints + gripper)."
            )
            return None
        return models

    def _resolve_signs(self, signs: List[float], label: str) -> List[float]:
        if not signs:
            return [1.0] * 8
        if len(signs) == 1:
            return [signs[0]] * 8
        if len(signs) != 8:
            self.get_logger().error(f"{label} length is {len(signs)}; expected 8.")
            return [1.0] * 8
        return signs

    def _load_zero_file(self, allow_missing: bool = False) -> None:
        self._zero_offsets = {"left": {}, "right": {}}
        if not self._zero_file:
            msg = "zero_file parameter is empty; cannot load dynamixel_zero_offsets.json"
            if allow_missing:
                self.get_logger().warn(f"{msg}. Will check after zero_on_start.")
                return
            self.get_logger().error(msg)
            raise RuntimeError(msg)
        try:
            with open(self._zero_file, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except FileNotFoundError:
            if allow_missing:
                self.get_logger().warn(
                    f"Zero offset file not found: '{self._zero_file}'. Will check after zero_on_start."
                )
                return
            msg = f"Zero offset file not found: '{self._zero_file}'"
            self.get_logger().error(msg)
            raise RuntimeError(msg)
        except (OSError, json.JSONDecodeError) as exc:
            msg = f"Failed to read zero file '{self._zero_file}': {exc}"
            self.get_logger().error(msg)
            raise RuntimeError(msg)

        for label in ("left", "right"):
            value = data.get(label)
            if isinstance(value, dict):
                self._zero_offsets[label] = {
                    str(name): float(val) for name, val in value.items()
                }

        self._left_offsets = self._zero_offsets["left"]
        self._right_offsets = self._zero_offsets["right"]

    def _ensure_zero_file_exists(self) -> None:
        if not self._zero_file or not os.path.isfile(self._zero_file):
            msg = f"Zero offset file not found after zero_on_start: '{self._zero_file}'"
            self.get_logger().error(msg)
            raise RuntimeError(msg)

    def _save_zero_file(self) -> None:
        if not self._zero_file:
            return
        folder = os.path.dirname(self._zero_file)
        if folder:
            os.makedirs(folder, exist_ok=True)
        with open(self._zero_file, "w", encoding="utf-8") as handle:
            json.dump(self._zero_offsets, handle, ensure_ascii=True, indent=2, sort_keys=True)

    def _capture_zero_offsets(self, bus: DynamixelMotorsBus, label: str) -> dict[str, float]:
        try:
            action = bus.sync_read("Present_Position", normalize=False)
        except Exception as exc:
            self.get_logger().warn(f"{label} read Present_Position failed: {exc}")
            return {}

        offsets: dict[str, float] = {}
        missing = []
        for name in self._joint_names + ["gripper"]:
            if name in action:
                offsets[name] = float(action[name])
            else:
                missing.append(name)

        if missing:
            self.get_logger().warn(f"{label} zero capture missing joints: {missing}")
        return offsets

    def _update_zero_offsets(self, label: str, offsets: dict[str, float]) -> None:
        if not offsets:
            return
        self._zero_offsets[label] = offsets
        if label == "left":
            self._left_offsets = offsets
        else:
            self._right_offsets = offsets
        self._save_zero_file()

    def _init_bus(
        self,
        port: str,
        ids: List[int],
        models: List[str],
        baudrate: int,
        enabled: bool,
        label: str,
    ) -> Optional[DynamixelMotorsBus]:
        if not enabled:
            self.get_logger().info(f"{label} arm disabled by parameter.")
            return None

        if not port:
            self.get_logger().warn(f"{label} port is empty; {label} arm disabled.")
            return None

        if len(ids) != 8:
            self.get_logger().error(
                f"{label} ids length is {len(ids)}; expected 8 (7 joints + gripper)."
            )
            return None

        if len(self._joint_names) != 7:
            self.get_logger().error(
                f"joint_names length is {len(self._joint_names)}; expected 7."
            )
            return None

        resolved_models = self._resolve_models(models, label)
        if not resolved_models:
            return None

        motors = {}
        for idx, (name, motor_id) in enumerate(zip(self._joint_names, ids[:7])):
            motors[name] = Motor(
                motor_id, resolved_models[idx], MotorNormMode.RANGE_M100_100
            )
        motors["gripper"] = Motor(
            ids[7], resolved_models[7], MotorNormMode.RANGE_0_100
        )
        bus = DynamixelMotorsBus(port=port, motors=motors)
        try:
            bus.connect(handshake=False)
            bus.set_baudrate(baudrate)
            bus._handshake()
            if self._zero_on_start:
                self._wait_for_enter(label)
                offsets = self._capture_zero_offsets(bus, label)
                self._update_zero_offsets(label, offsets)
            self.get_logger().info(f"{label} arm connected on {port}.")
        except Exception as exc:
            self.get_logger().error(f"Failed to connect {label} arm on {port}: {exc}")
            return None

        return bus

    def _wait_for_enter(self, label: str) -> None:
        self.get_logger().info(
            f"Move {label} arm to the zero pose, then press ENTER to record zero."
        )
        if sys.stdin.isatty():
            input()
            return
        try:
            with open("/dev/tty", "r", encoding="utf-8") as tty:
                tty.readline()
            return
        except OSError as exc:
            self.get_logger().error(
                f"No TTY available for ENTER input: {exc}. "
                "Run with a TTY or set zero_on_start:=false."
            )
            while rclpy.ok():
                time.sleep(0.5)

    def _action_loop(self) -> None:
        next_time = time.perf_counter()
        while not self._stop_event.is_set():
            self._publish_action()
            next_time += self._period_s
            sleep_s = next_time - time.perf_counter()
            if sleep_s > 0:
                time.sleep(sleep_s)
            else:
                next_time = time.perf_counter()

    def _read_positions(
        self, bus: DynamixelMotorsBus, label: str
    ) -> Tuple[List[float], float, bool]:
        positions = []
        valid = True
        try:
            action = bus.sync_read("Present_Position", normalize=False)
        except Exception as exc:
            self.get_logger().warn(f"{label} sync_read failed: {exc}")
            return [0.0] * len(self._joint_names), 0.0, False

        missing = []
        gripper = 0.0
        offsets = self._left_offsets if label == "left" else self._right_offsets
        for name in self._joint_names:
            if name in action:
                raw = float(action[name])
                positions.append(raw - float(offsets.get(name, 0.0)))
            else:
                positions.append(0.0)
                missing.append(name)
        if "gripper" in action:
            raw_gripper = float(action["gripper"])
            gripper = raw_gripper - float(offsets.get("gripper", 0.0))
        else:
            missing.append("gripper")

        if missing:
            self.get_logger().warn(f"{label} missing joints: {missing}")
            valid = False

        signs = self._left_signs if label == "left" else self._right_signs
        positions = [
            pos * self._position_scale * signs[idx]
            for idx, pos in enumerate(positions)
        ]
        gripper = gripper * self._position_scale * signs[7]
        gripper = self._normalize_gripper(label, gripper)
        return positions, gripper, valid

    def _normalize_gripper(self, label: str, value: float) -> float:
        if label == "left":
            min_val = self._left_gripper_min
            max_val = self._left_gripper_max
        else:
            min_val = self._right_gripper_min
            max_val = self._right_gripper_max

        if max_val <= min_val:
            self.get_logger().warn(
                f"{label} gripper min/max invalid: min={min_val} max={max_val}"
            )
            return value

        norm = (value - min_val) / (max_val - min_val)
        if norm < 0.0:
            return 0.0
        if norm > 1.0:
            return 1.0
        return norm

    def _format_positions_line(
        self,
        label: str,
        enabled: bool,
        valid: bool,
        positions: List[float],
        gripper: float,
    ) -> str:
        if not enabled:
            return f"{label}: disabled"
        if not positions:
            return f"{label}: no data"

        pairs = [f"{name}={pos:.3f}" for name, pos in zip(self._joint_names, positions)]
        pairs.append(f"gripper={gripper:.3f}")
        status = "" if valid else " (invalid)"
        return f"{label}: " + " ".join(pairs) + status

    def _maybe_print_positions(
        self,
        left_positions: List[float],
        right_positions: List[float],
        left_gripper: float,
        right_gripper: float,
        left_valid: bool,
        right_valid: bool,
    ) -> None:
        if not self._print_positions:
            return

        # now = time.perf_counter()
        # if now - self._last_print_time < self._print_period:
        #     return

        # self._last_print_time = now
        lines = [
            self._format_positions_line(
                "left", self._left_enabled, left_valid, left_positions, left_gripper
            ),
            # self._format_positions_line(
            #     "right", self._right_enabled, right_valid, right_positions, right_gripper
            # ),
        ]

        for line in lines:
            print(line, flush=True)

    def _publish_action(self) -> None:
        msg = OculusInitJointState()
        now = self.get_clock().now().to_msg()
        msg.header.stamp = now
        msg.header.frame_id = self._frame_id

        left = JointState()
        left.header.stamp = now
        left.header.frame_id = self._frame_id
        right = JointState()
        right.header.stamp = now
        right.header.frame_id = self._frame_id

        msg.left_valid = False
        msg.right_valid = False
        left_positions: List[float] = []
        right_positions: List[float] = []
        left_gripper = 0.0
        right_gripper = 0.0

        if self._left_bus:
            left_positions, left_gripper, valid = self._read_positions(
                self._left_bus, "left"
            )
            left.name = self._joint_names
            left.position = left_positions
            msg.left_valid = valid

        if self._right_bus:
            right_positions, right_gripper, valid = self._read_positions(
                self._right_bus, "right"
            )
            right.name = self._joint_names
            right.position = right_positions
            msg.right_valid = valid

        msg.left = left
        msg.left_gripper = left_gripper
        msg.right_gripper = right_gripper
        msg.right = right
        msg.init = self._first_publish
        self._maybe_print_positions(
            left_positions,
            right_positions,
            left_gripper,
            right_gripper,
            msg.left_valid,
            msg.right_valid,
        )

        if msg.left_valid or msg.right_valid:
            self._publisher.publish(msg)
            self._first_publish = False

    def destroy_node(self):
        self._stop_event.set()
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)

        for label, bus in (("left", self._left_bus), ("right", self._right_bus)):
            if bus and bus.is_connected:
                try:
                    bus.disconnect()
                    self.get_logger().info(f"{label} arm disconnected.")
                except Exception as exc:
                    self.get_logger().warn(f"{label} disconnect failed: {exc}")

        return super().destroy_node()


def main():
    rclpy.init()
    node = DynamixelInitJointStateNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
