#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import serial
import threading
from std_msgs.msg import Float32MultiArray

def _as_int(s: str, default: int) -> int:
    try:
        return int(str(s))
    except Exception:
        return default

def _as_float(s: str, default: float) -> float:
    try:
        return float(str(s))
    except Exception:
        return default

class TactileSensorNode(Node):
    def __init__(self):
        super().__init__('tactile_sensor_node')

        # --- 用字符串默认值声明参数（避免类型不匹配），再在代码里做类型转换 ---
        self.declare_parameter('port', '/dev/ttyUSB0')       # 串口
        self.declare_parameter('topic', '/tactile_data')     # 发布话题
        self.declare_parameter('baudrate', '115200')         # 波特率
        self.declare_parameter('timeout', '0.5')             # 读超时(s)
        self.declare_parameter('frame_size', '70')           # 帧总长度
        self.declare_parameter('header_hex', 'FF 84')        # 帧头，空格分隔的HEX

        port       = self.get_parameter('port').get_parameter_value().string_value
        topic      = self.get_parameter('topic').get_parameter_value().string_value
        baudrate   = _as_int(self.get_parameter('baudrate').get_parameter_value().string_value, 115200)
        timeout_s  = _as_float(self.get_parameter('timeout').get_parameter_value().string_value, 0.5)
        frame_size = _as_int(self.get_parameter('frame_size').get_parameter_value().string_value, 70)
        header_str = self.get_parameter('header_hex').get_parameter_value().string_value

        # 解析帧头
        try:
            self.header_bytes = bytes.fromhex(header_str)
            if len(self.header_bytes) < 1:
                raise ValueError("header_hex must contain at least 1 byte")
        except Exception as e:
            self.get_logger().warn(f'Invalid header_hex "{header_str}", fallback to FF 84: {e}')
            self.header_bytes = b'\xFF\x84'

        self.frame_size = frame_size

        # 发布器
        self.publisher = self.create_publisher(Float32MultiArray, topic, 10)
        self.get_logger().info(f"Init: port={port}, baud={baudrate}, timeout={timeout_s}s, "
                               f"frame_size={frame_size}, header={self.header_bytes.hex(' ')}; "
                               f"publishing on {topic}")

        # 打开串口
        try:
            self.ser = serial.Serial(port, baudrate=baudrate, timeout=timeout_s)
            self.get_logger().info(f"Serial port {port} opened")
        except Exception as e:
            self.get_logger().error(f"Open serial failed: {e}")
            raise

        # 线程停止标志 & 读线程
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def _read_loop(self):
        """后台线程：持续读取完整帧 -> 校验 -> 解析 -> 发布。"""
        saw_first = False
        try:
            while rclpy.ok() and not self._stop_event.is_set():
                b = self.ser.read(1)
                if not b:
                    continue
                if not saw_first:
                    if b == self.header_bytes[:1]:
                        saw_first = True
                    continue
                else:
                    # 检第二个字节（若存在）
                    if len(self.header_bytes) == 1 or b == self.header_bytes[1:2]:
                        remain = self.frame_size - len(self.header_bytes)
                        rest = self.ser.read(remain)
                        if len(rest) < remain:
                            saw_first = False
                            continue
                        frame = self.header_bytes + rest
                        if self._verify_frame_checksum(frame):
                            # 解析 32 路ADC（大端 u16），跳过 2B 计数器(位于 [2:4])
                            adc = [int.from_bytes(frame[4+2*i:6+2*i], 'big') for i in range(32)]
                            msg = Float32MultiArray()
                            msg.data = [float(x) for x in adc]
                            self.publisher.publish(msg)
                        else:
                            self.get_logger().warn("Checksum failed, dropping frame")
                        saw_first = False
                    else:
                        # 第二字节未命中，若该字节也是 header[0]，可视为新的起点
                        saw_first = (b == self.header_bytes[:1])
        except serial.SerialException as e:
            self.get_logger().error(f"Serial error: {e}")
        finally:
            self.get_logger().info("Read loop exiting")

    def _verify_frame_checksum(self, frame: bytes) -> bool:
        if len(frame) != self.frame_size:
            return False
        # 对字节2..67求和低16位，与字节68..69组成的16位比较
        calc = sum(frame[2:68]) & 0xFFFF
        recv = (frame[68] << 8) | frame[69]
        return calc == recv

    def destroy_node(self):
        """优雅退出：停线程、关串口。"""
        self._stop_event.set()
        try:
            if self.ser:
                # 可能在阻塞读，尝试打断
                try:
                    self.ser.cancel_read()
                except Exception:
                    pass
        except Exception:
            pass
        if hasattr(self, "_thread") and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        try:
            if getattr(self, "ser", None) and self.ser.is_open:
                self.ser.close()
        except Exception:
            pass
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = TactileSensorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Keyboard interrupt, shutting down.")
    node.destroy_node()
    rclpy.shutdown()
