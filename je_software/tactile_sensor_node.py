#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import time
import threading
from collections import deque
from typing import List

import serial  # pyserial
import rclpy
from rclpy.node import Node
from rclpy.logging import LoggingSeverity

from std_msgs.msg import Float32MultiArray


def _hexdump(b: bytes, max_len: int = 64) -> str:
    """将字节流做简要十六进制转储，限制长度，便于日志排查"""
    if not isinstance(b, (bytes, bytearray)):
        return "<not-bytes>"
    s = b[:max_len].hex(" ")
    if len(b) > max_len:
        s += f" ...(+{len(b) - max_len} bytes)"
    return s


class TactileSensorNode(Node):
    """
    串口读取（70 字节）：
      header(2B: 0xFF 0x84) + cnt(2B BE) + 32*ADC(64B: BE u16) + CRC(2B BE)
    后台线程持续抓帧 -> 校验 -> 解析 -> 发布 Float32MultiArray（32通道）
    """

    def __init__(self):
        super().__init__('tactile_sensor_node')

        # ---------------- 参数 ----------------
        # 基本串口配置
        self.declare_parameter('port', '/dev/ttyUSB0')       # 串口设备
        self.declare_parameter('baudrate', 115200)           # 波特率
        self.declare_parameter('timeout', 0.5)               # 读超时（秒）
        self.declare_parameter('frame_size', 70)             # 帧长度
        self.declare_parameter('header_hex', 'FF 84')        # 帧头（十六进制字符串）

        # ROS 话题
        self.declare_parameter('topic', '/tactile_data')

        # 日志/调试
        self.declare_parameter('log_level', 'info')          # debug/info/warn/error
        self.declare_parameter('dump_bad_frames', False)     # CRC 错时是否转储十六进制
        self.declare_parameter('dump_bad_limit', 3)          # 最多转储多少次
        self.declare_parameter('stats_period_s', 5.0)        # 多久打印一次统计

        # 获取参数（rclpy 会做类型转换）
        p = self.get_parameter
        self.port: str = p('port').value
        self.baudrate: int = int(p('baudrate').value)
        self.timeout_s: float = float(p('timeout').value)
        self.frame_size: int = int(p('frame_size').value)
        header_hex: str = str(p('header_hex').value)
        self.topic: str = p('topic').value

        log_level_str: str = str(p('log_level').value).lower().strip()
        self.dump_bad_frames: bool = bool(p('dump_bad_frames').value)
        self.dump_bad_limit: int = int(p('dump_bad_limit').value)
        self.stats_period_s: float = float(p('stats_period_s').value)

        # 设置日志等级（若可用）
        lvl = {
            'debug': LoggingSeverity.DEBUG,
            'info': LoggingSeverity.INFO,
            'warn': LoggingSeverity.WARN,
            'warning': LoggingSeverity.WARN,
            'error': LoggingSeverity.ERROR,
        }.get(log_level_str, LoggingSeverity.INFO)
        try:
            self.get_logger().set_level(lvl)
        except Exception:
            # 某些发行版不支持 set_level，忽略即可
            pass

        # 解析帧头
        try:
            self.header_bytes = bytes.fromhex(header_hex)
            if len(self.header_bytes) < 1:
                raise ValueError("header_hex must contain at least 1 byte")
        except Exception as e:
            self.get_logger().warn(f'Invalid header_hex "{header_hex}", fallback to FF 84: {e}')
            self.header_bytes = b'\xFF\x84'

        # ---------------- 发布器 ----------------
        self.publisher = self.create_publisher(Float32MultiArray, self.topic, 10)

        # ---------------- 统计量 ----------------
        self._frames_ok = 0          # 成功帧
        self._frames_crc_bad = 0     # CRC 错
        self._frames_short = 0       # 未读满
        self._resync_cnt = 0         # 头部重同步次数
        self._bytes_read = 0         # 读到的总字节
        self._last_stats_t = time.monotonic()
        self._ok_times = deque(maxlen=2048)  # 用于估计平均 FPS（成功帧时间戳）

        # ---------------- 启动串口 ----------------
        try:
            self.ser = serial.Serial(self.port, baudrate=self.baudrate, timeout=self.timeout_s)
            self.get_logger().info(
                f"Serial opened: port={self.port}, baud={self.baudrate}, timeout={self.timeout_s:.3f}s, "
                f"frame_size={self.frame_size}, header={self.header_bytes.hex(' ')}, topic={self.topic}, "
                f"stats_period={self.stats_period_s:.1f}s, dump_bad_frames={self.dump_bad_frames}"
            )
        except Exception as e:
            self.get_logger().error(f"Open serial failed: {e!r}")
            raise

        # ---------------- 后台线程 ----------------
        self._stop_event = threading.Event()
        self._bad_dumped = 0  # 已转储坏帧次数
        self._thread = threading.Thread(target=self._read_loop, name="tactile-read", daemon=True)
        self._thread.start()

        # 周期统计
        self._stats_timer = self.create_timer(self.stats_period_s, self._emit_stats)

    # ---------------- 内部：CRC 校验 ----------------
    def _verify_frame_checksum(self, frame: bytes) -> bool:
        """对 frame[2:frame_size-2] 求和取低 16 位，与末尾两字节（BE）比对"""
        if len(frame) != self.frame_size:
            return False
        calc = sum(frame[2:self.frame_size - 2]) & 0xFFFF
        recv = (frame[self.frame_size - 2] << 8) | frame[self.frame_size - 1]
        return calc == recv

    # ---------------- 后台读线程 ----------------
    def _read_loop(self):
        saw_first = False
        header0 = self.header_bytes[:1]
        header1 = self.header_bytes[1:2] if len(self.header_bytes) >= 2 else b''

        try:
            while rclpy.ok() and not self._stop_event.is_set():
                b0 = self.ser.read(1)
                if not b0:
                    continue
                self._bytes_read += 1

                if not saw_first:
                    if b0 == header0:
                        saw_first = True
                    continue

                # 已有第一字节，检查第二字节（如果定义了）
                if (not header1) or (b0 == header1):
                    remain = self.frame_size - len(self.header_bytes)
                    rest = self.ser.read(remain)
                    self._bytes_read += len(rest)

                    if len(rest) < remain:
                        # 未读满，重置同步
                        self._frames_short += 1
                        self.get_logger().debug(
                            f"short frame: got {len(rest)} need {remain} (port={self.port})"
                        )
                        saw_first = False
                        self._resync_cnt += 1
                        continue

                    frame = self.header_bytes + rest
                    ok = self._verify_frame_checksum(frame)
                    if not ok:
                        self._frames_crc_bad += 1
                        if self.dump_bad_frames and self._bad_dumped < self.dump_bad_limit:
                            self._bad_dumped += 1
                            self.get_logger().warn(
                                f"CRC failed, dump[{self._bad_dumped}/{self.dump_bad_limit}]: "
                                f"{_hexdump(frame, 80)}"
                            )
                        # 失败也重置同步
                        self._resync_cnt += 1
                        saw_first = False
                        continue

                    # 解析 32 路 ADC（BE u16），跳过 [2:4] 的计数器
                    try:
                        adc: List[float] = []
                        for i in range(32):
                            v = int.from_bytes(frame[4 + 2 * i:6 + 2 * i], 'big', signed=False)
                            adc.append(float(v))
                        msg = Float32MultiArray(data=adc)
                        self.publisher.publish(msg)
                        self._frames_ok += 1
                        self._ok_times.append(time.monotonic())
                    except Exception as e:
                        self.get_logger().warn(f"parse ADC failed: {e!r}")
                        self._resync_cnt += 1

                    # 重置以寻找下一帧
                    saw_first = False
                else:
                    # 第二字节不匹配，若它又是 header0，视为新的起点（允许连续 0xFF）
                    saw_first = (b0 == header0)
                    if not saw_first:
                        self._resync_cnt += 1

        except serial.SerialException as e:
            self.get_logger().error(f"Serial error: {e!r}")
        except Exception as e:
            self.get_logger().error(f"Unexpected read loop exception: {e!r}")
        finally:
            self.get_logger().info("Read loop exiting")

    # ---------------- 周期统计日志 ----------------
    def _emit_stats(self):
        now = time.monotonic()
        dt = now - self._last_stats_t
        if dt <= 0.0:
            return

        # 平均 FPS：用成功帧的时间戳窗口计算
        fps = 0.0
        if len(self._ok_times) >= 2:
            dt_win = self._ok_times[-1] - self._ok_times[0]
            if dt_win > 0:
                fps = (len(self._ok_times) - 1) / dt_win

        self.get_logger().info(
            f"[stats {dt:.1f}s] ok={self._frames_ok} ({fps:.2f} FPS avg), "
            f"crc_bad={self._frames_crc_bad}, short={self._frames_short}, "
            f"resync={self._resync_cnt}, bytes={self._bytes_read}"
        )
        # 归零计数器只在本周期内累计（如需全局累计，去掉这几行）
        self._frames_ok = 0
        self._frames_crc_bad = 0
        self._frames_short = 0
        self._resync_cnt = 0
        self._bytes_read = 0
        self._last_stats_t = now

    # ---------------- 优雅关闭 ----------------
    def destroy_node(self):
        self.get_logger().info("Shutting down tactile sensor node ...")
        self._stop_event.set()
        try:
            if self.ser:
                try:
                    # 打断阻塞读（部分平台支持）
                    self.ser.cancel_read()
                except Exception:
                    pass
        except Exception:
            pass
        try:
            if hasattr(self, "_thread") and self._thread.is_alive():
                self._thread.join(timeout=1.0)
        except Exception:
            pass
        try:
            if getattr(self, "ser", None) and self.ser.is_open:
                self.ser.close()
                self.get_logger().info("Serial closed")
        except Exception:
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = TactileSensorNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        # 兜底日志，避免 launch 只看到退出码
        print(f"[FATAL] tactile_sensor_node crashed: {e!r}", file=sys.stderr)
        raise
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
