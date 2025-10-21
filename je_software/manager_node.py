#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, json, threading, time, re
from typing import List, Tuple, Optional, Any
from datetime import datetime
from queue import SimpleQueue, Empty

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Float32MultiArray

from cv_bridge import CvBridge
import cv2
import numpy as np


# ====================== 小工具 ======================

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def sanitize(name: str) -> str:
    return re.sub(r'[^A-Za-z0-9_.-]+', '_', name.strip('/'))


def stamp_to_ns(stamp) -> int:
    # 假设 stamp 有 sec / nanosec，且可能为 0
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


# ====================== 高效多流对齐器 ======================

class MultiStreamAligner:
    """
    无锁摄取 → 工作线程独占本地窗口：
      - 回调 put_nowait 到 ingest 队列
      - step():
        1) drain 各 ingest → 本地 times / msgs（保持单调；乱序帧丢弃）
        2) t_ref = min(latest_i)
        3) 对每路推进“只前进”游标到 <= target 的最大位置，并在它与后一位择近；误差<=tol_i
        4) 成功返回 (t_ref, picks)，并剪枝到选中位置
      - 支持 max_window_ns 以限制窗口大小
    """

    def __init__(
            self,
            num_streams: int,
            tolerances_ns: List[int],
            offsets_ns: Optional[List[int]] = None,
            max_window_ns: Optional[int] = None,
    ):
        assert num_streams == len(tolerances_ns), "tolerances_ns length must match num_streams"
        self.N = int(num_streams)
        self.tolerances_ns = list(map(int, tolerances_ns))
        self.offsets_ns = list(map(int, offsets_ns)) if offsets_ns else [0] * self.N
        self.max_window_ns = int(max_window_ns) if max_window_ns else None

        self.ingest: List[SimpleQueue] = [SimpleQueue() for _ in range(self.N)]
        self.times: List[List[int]] = [[] for _ in range(self.N)]
        self.msgs: List[List[Any]] = [[] for _ in range(self.N)]
        self.cursor: List[int] = [-1] * self.N

    def put_nowait(self, i: int, t_ns: int, msg: Any):
        try:
            self.ingest[i].put_nowait((t_ns, msg))
        except Exception:
            # 极端情况下直接丢帧
            pass

    def _drain_one(self, i: int):
        qi = self.ingest[i]
        ti = self.times[i]
        mi = self.msgs[i]
        last = ti[-1] if ti else -1
        while True:
            try:
                t_ns, msg = qi.get_nowait()
            except Empty:
                break
            if t_ns >= last:
                ti.append(t_ns)
                mi.append(msg)
                last = t_ns
            # 否则：偶发乱序，直接丢弃

        # 限定窗口长度以控内存
        if self.max_window_ns and len(ti) >= 2:
            cutoff = last - self.max_window_ns
            drop_k, n = 0, len(ti)
            while drop_k < n and ti[drop_k] < cutoff:
                drop_k += 1
            if drop_k > 0:
                del ti[:drop_k]
                del mi[:drop_k]
                self.cursor[i] -= drop_k
                if self.cursor[i] < -1:
                    self.cursor[i] = -1

    def _drain(self):
        for i in range(self.N):
            self._drain_one(i)

    @staticmethod
    def _advance_and_pick(times: List[int], cur: int, target: int) -> Tuple[int, int]:
        n = len(times)
        # 推进到最后一个 <= target
        while cur + 1 < n and times[cur + 1] <= target:
            cur += 1
        # 在 cur 与 cur+1 中择近
        pick = 0 if cur < 0 else cur
        if cur + 1 < n:
            left_ts, right_ts = times[pick], times[cur + 1]
            if abs(right_ts - target) < abs(left_ts - target):
                pick = cur + 1
        return pick, cur

    def step(self) -> Optional[Tuple[int, List[Tuple[int, Any]]]]:
        self._drain()

        latest = []
        for i in range(self.N):
            if not self.times[i]:
                return None
            latest.append(self.times[i][-1])

        t_ref = min(latest)

        picks_idx: List[int] = [-1] * self.N
        picks: List[Tuple[int, Any]] = [None] * self.N  # type: ignore
        for i in range(self.N):
            target = t_ref + self.offsets_ns[i]
            pick_i, cur_after = self._advance_and_pick(self.times[i], self.cursor[i], target)
            if abs(self.times[i][pick_i] - target) > self.tolerances_ns[i]:
                return None
            picks_idx[i] = pick_i
            picks[i] = (self.times[i][pick_i], self.msgs[i][pick_i])
            self.cursor[i] = cur_after

        for i in range(self.N):
            k = picks_idx[i]
            del self.times[i][:k + 1]
            del self.msgs[i][:k + 1]
            self.cursor[i] -= (k + 1)
            if self.cursor[i] < -1:
                self.cursor[i] = -1

        return t_ref, picks


# ====================== ROS2 节点 ======================

class Manager(Node):
    """
    订阅多路彩色/深度/关节/触觉；保存线程使用 min-latest + 单调游标 对齐；锁外转换/写盘。
    修复点：
      - 0 时间戳/无时间戳 → 回退本地时钟
      - 过滤空话题
      - 图像/深度 BEST_EFFORT；关节/触觉 RELIABLE
    """

    def __init__(self):
        super().__init__('manager')

        # ---------- 参数 ----------
        # 话题
        self.declare_parameter('color_topics', [])
        self.declare_parameter('depth_topics', [])
        self.declare_parameter('color_topics_csv', '')
        self.declare_parameter('depth_topics_csv', '')
        self.declare_parameter('joint_state_topic', '/robot/joint_states')
        self.declare_parameter('tactile_topic', '/tactile_data')

        # 频率与容差（ms）
        self.declare_parameter('rate_hz', 30.0)
        self.declare_parameter('image_tolerance_ms', 15.0)  # ≤ 半帧
        self.declare_parameter('joint_tolerance_ms', 15.0)
        self.declare_parameter('tactile_tolerance_ms', 60.0)

        # 窗口与目录
        self.declare_parameter('queue_seconds', 2.0)
        self.declare_parameter('save_dir', os.path.expanduser('~/ros2_logs/sensor_logger'))
        self.declare_parameter('session_name', '')
        self.declare_parameter('save_depth', True)

        # 其他
        self.declare_parameter('use_ros_time', True)
        self.declare_parameter('color_jpeg_quality', 95)
        self.declare_parameter('do_calculate_hz', True)

        # 偏置（ms）
        self.declare_parameter('color_offsets_ms', [])
        self.declare_parameter('depth_offsets_ms', [])
        self.declare_parameter('joint_offset_ms', 0.0)
        self.declare_parameter('tactile_offset_ms', 0.0)

        p = self.get_parameter

        # 读取话题参数
        color_topics = list(p('color_topics').value or [])
        depth_topics = list(p('depth_topics').value or [])
        if not color_topics:
            csv = p('color_topics_csv').value or ''
            if csv.strip():
                color_topics = [s.strip() for s in csv.split(',') if s.strip()]
        if not depth_topics:
            csv = p('depth_topics_csv').value or ''
            if csv.strip():
                depth_topics = [s.strip() for s in csv.split(',') if s.strip()]

        # 过滤空字符串话题
        color_topics = [t.strip() for t in color_topics if t and t.strip()]
        depth_topics = [t.strip() for t in depth_topics if t and t.strip()]

        self.color_topics: List[str] = color_topics
        self.depth_topics: List[str] = depth_topics
        self.joint_topic: str = p('joint_state_topic').value
        self.tactile_topic: str = p('tactile_topic').value

        # 频率/容差
        self.rate_hz: float = float(p('rate_hz').value)
        image_tol_ns = int(float(p('image_tolerance_ms').value) * 1e6)
        joint_tol_ns = int(float(p('joint_tolerance_ms').value) * 1e6)
        tactile_tol_ns = int(float(p('tactile_tolerance_ms').value) * 1e6)

        # 窗口/目录
        self.queue_seconds: float = float(p('queue_seconds').value)
        self.save_dir: str = p('save_dir').value
        session_name: str = p('session_name').value
        self.save_depth: bool = bool(p('save_depth').value)
        self.use_ros_time: bool = bool(p('use_ros_time').value)
        self.jpeg_quality: int = int(p('color_jpeg_quality').value)

        # 偏置
        color_offsets_ms = list(p('color_offsets_ms').value or [])
        depth_offsets_ms = list(p('depth_offsets_ms').value or [])
        joint_offset_ms = float(p('joint_offset_ms').value or 0.0)
        tactile_offset_ms = float(p('tactile_offset_ms').value or 0.0)

        # 统计频率
        self.do_calculate_hz: bool = bool(p('do_calculate_hz').value)
        self._attempt_win = 0
        self._success_win = 0

        if not session_name:
            session_name = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.session_dir = os.path.join(self.save_dir, sanitize(session_name))
        ensure_dir(self.session_dir)

        # 相机目录
        self.cam_count = len(self.color_topics)
        self.cam_dirs = []
        for i in range(self.cam_count):
            cam_name = sanitize(self.color_topics[i])
            base = os.path.join(self.session_dir, f'cam_{i:02d}_{cam_name}')
            ensure_dir(base)
            ensure_dir(os.path.join(base, 'color'))
            if self.save_depth and (i < len(self.depth_topics) and self.depth_topics[i]):
                ensure_dir(os.path.join(base, 'depth'))
            self.cam_dirs.append(base)

        self.meta_dir = os.path.join(self.session_dir, 'meta')
        ensure_dir(self.meta_dir)

        self.get_logger().info(f"Session: {self.session_dir}")
        self.get_logger().info(f"Color topics: {self.color_topics}")
        self.get_logger().info(f"Depth  topics: {self.depth_topics}")
        self.get_logger().info(f"Joint: {self.joint_topic}, Tactile: {self.tactile_topic}")
        self.get_logger().info(
            f"rate={self.rate_hz}Hz, tol(img/joint/tactile)=[{image_tol_ns / 1e6:.1f},{joint_tol_ns / 1e6:.1f},{tactile_tol_ns / 1e6:.1f}]ms"
        )

        # ---------- 对齐器 ----------
        C = len(self.color_topics)
        D = len(self.depth_topics) if self.save_depth else 0
        self._idx_color = list(range(C))
        self._idx_depth = list(range(C, C + D))
        self._idx_joint = C + D
        self._idx_tact = C + D + 1
        num_streams = C + D + 2

        tolerances_ns: List[int] = []
        tolerances_ns += [image_tol_ns] * C
        tolerances_ns += [image_tol_ns] * D
        tolerances_ns += [joint_tol_ns, tactile_tol_ns]

        offsets_ns: List[int] = []
        if color_offsets_ms and len(color_offsets_ms) == C:
            offsets_ns += [int(ms * 1e6) for ms in color_offsets_ms]
        else:
            offsets_ns += [0] * C
        if D > 0:
            if depth_offsets_ms and len(depth_offsets_ms) == D:
                offsets_ns += [int(ms * 1e6) for ms in depth_offsets_ms]
            else:
                offsets_ns += [0] * D
        offsets_ns += [int(joint_offset_ms * 1e6), int(tactile_offset_ms * 1e6)]

        max_window_ns = int(self.queue_seconds * 1e9)

        self.aligner = MultiStreamAligner(
            num_streams=num_streams,
            tolerances_ns=tolerances_ns,
            offsets_ns=offsets_ns,
            max_window_ns=max_window_ns,
        )

        # ---------- 订阅（区分 QoS） ----------
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        reliable_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        for i, topic in enumerate(self.color_topics):
            self.create_subscription(Image, topic, self._mk_color_cb(i), sensor_qos)

        for j in range(D):
            topic = self.depth_topics[j]
            if topic and self.save_depth:
                self.create_subscription(Image, topic, self._mk_depth_cb(j), sensor_qos)

        # 关节/触觉用 RELIABLE（若上游是 BEST_EFFORT，可改成 sensor_qos）
        self.create_subscription(JointState, self.joint_topic, self._joint_cb, reliable_qos)
        self.create_subscription(Float32MultiArray, self.tactile_topic, self._tactile_cb, reliable_qos)

        # 工具
        self.bridge = CvBridge()

        # 保存线程
        self.sample_idx = 0
        self._stop_evt = threading.Event()
        self.worker = threading.Thread(target=self._save_loop, name='logger-aligner', daemon=False)
        self.worker.start()

    # ---------- 时间戳回退：0/无时间戳 → 本地时钟 ----------
    def _ns_from_header_or_clock(self, header) -> int:
        try:
            s = int(header.stamp.sec)
            ns = int(header.stamp.nanosec)
            if s != 0 or ns != 0:
                return s * 1_000_000_000 + ns
        except Exception:
            pass
        return self.get_clock().now().nanoseconds

    # ---------- 回调：无锁摄取 ----------
    def _mk_color_cb(self, i: int):
        def _cb(msg: Image):
            if self.use_ros_time:
                t_ns = self._ns_from_header_or_clock(msg.header)
            else:
                t_ns = self.get_clock().now().nanoseconds
            self.aligner.put_nowait(self._idx_color[i], t_ns, msg)
            # self.get_logger().info(f"receive image:{i} at {time.perf_counter()}")

        return _cb

    def _mk_depth_cb(self, j: int):
        def _cb(msg: Image):
            if self.use_ros_time:
                t_ns = self._ns_from_header_or_clock(msg.header)
            else:
                t_ns = self.get_clock().now().nanoseconds
            self.aligner.put_nowait(self._idx_depth[j], t_ns, msg)
            # self.get_logger().info(f"receive depth:{j} at {time.perf_counter()}")

        return _cb

    def _joint_cb(self, msg: JointState):
        # Joint 用 header（若 0 则回退）
        t_ns = self._ns_from_header_or_clock(msg.header)
        self.aligner.put_nowait(self._idx_joint, t_ns, msg)

    def _tactile_cb(self, msg: Float32MultiArray):
        # 触觉通常无 header，用接收时钟
        t_ns = self.get_clock().now().nanoseconds
        self.aligner.put_nowait(self._idx_tact, t_ns, msg)

    def _save_loop(self):
        period = 1.0 / max(1e-6, self.rate_hz)
        next_t = time.perf_counter()
        while rclpy.ok() and not self._stop_evt.is_set():
            now = time.perf_counter()
            if now < next_t:
                time.sleep(next_t - now)
            next_t += period
            try:
                # === 新增：记录一次尝试 ===
                if self.do_calculate_hz:
                    self._attempt_win += 1

                out = self.aligner.step()
                if out is None:
                    continue

                # === 新增：记录一次成功 ===
                if self.do_calculate_hz:
                    self._success_win += 1

                t_ref, picks = out
                self._save_once(t_ref, picks)

                # === 新增：保存成功后更新统计 ===
                if self.do_calculate_hz:
                    self._on_saved_stats()
            except Exception as e:
                self.get_logger().error(f"save loop error: {e}")

    # === 新增：统计函数 ===
    def _on_saved_stats(self):
        now = time.perf_counter()
        self._save_times.append(now)

        # 滑动窗口：只保留最近 stats_window_s 秒的时间戳
        cutoff = now - self.stats_window_s
        while self._save_times and self._save_times[0] < cutoff:
            self._save_times.popleft()

        # 计算窗口 FPS 和瞬时 FPS
        win_fps = 0.0
        inst_fps = 0.0
        if len(self._save_times) >= 2:
            dt_win = self._save_times[-1] - self._save_times[0]
            if dt_win > 0:
                win_fps = (len(self._save_times) - 1) / dt_win
            dt_inst = self._save_times[-1] - self._save_times[-2]
            if dt_inst > 0:
                inst_fps = 1.0 / dt_inst

        # 到时间就打印一次统计，并清空尝试/成功的窗口计数
        if (now - self._last_rate_log_t) >= self.stats_log_period_s:
            hit_ratio = (self._success_win / self._attempt_win) if self._attempt_win > 0 else 0.0
            self.get_logger().info(
                f"[save stats] inst_fps={inst_fps:.2f}, "
                f"win({self.stats_window_s:.1f}s)_fps={win_fps:.2f}, "
                f"align_hit_ratio={hit_ratio:.1%} "
                f"(attempts={self._attempt_win}, success={self._success_win})"
            )
            self._attempt_win = 0
            self._success_win = 0
            self._last_rate_log_t = now

    def _save_once(self, t_ref: int, picks: List[Tuple[int, Any]]):
        idx = self.sample_idx
        self.sample_idx += 1

        C = len(self._idx_color)
        D = len(self._idx_depth)
        color_picks = picks[0:C]
        depth_picks = picks[C:C + D] if D > 0 else []
        joint_ts, js_msg = picks[C + D]
        tact_ts, tact_msg = picks[C + D + 1]

        meta = {
            'index': idx,
            't_ref_ns': int(t_ref),
            'color': [],
            'depth': [],
            'joint': {'stamp_ns': int(joint_ts)},
            'tactile': {'stamp_ns': int(tact_ts)},
        }

        # --- 保存彩色 ---
        for cam_i, (t_ns, msg) in enumerate(color_picks):
            # cv_bridge 转换（锁外）
            if getattr(msg, "encoding", "") != 'bgr8':
                img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            else:
                img = self.bridge.imgmsg_to_cv2(msg)
            cam_dir = self.cam_dirs[cam_i]
            fn = f'color_{idx:06d}.jpg'
            fp = os.path.join(cam_dir, 'color', fn)
            cv2.imwrite(fp, img, [int(cv2.IMWRITE_JPEG_QUALITY), int(self.jpeg_quality)])
            meta['color'].append({
                'cam_index': cam_i,
                'topic': self.color_topics[cam_i],
                'stamp_ns': int(t_ns),
                'file': os.path.relpath(fp, self.session_dir),
            })

        # --- 保存深度 ---
        if self.save_depth and depth_picks:
            for dep_i, (t_ns, msg) in enumerate(depth_picks):
                cv_img = self.bridge.imgmsg_to_cv2(msg)
                scale_info = None
                if cv_img.dtype in (np.float32, np.float64):
                    cv_mm = np.clip(cv_img * 1000.0, 0, 65535).astype(np.uint16)
                    out = cv_mm
                    scale_info = 'float_meter_to_uint16_mm'
                elif cv_img.dtype == np.uint16:
                    out = cv_img
                else:
                    out = cv_img.astype(np.uint16)

                cam_dir = self.cam_dirs[dep_i] if dep_i < len(self.cam_dirs) else os.path.join(self.session_dir,
                                                                                               f'cam_dep_{dep_i:02d}')
                ensure_dir(os.path.join(cam_dir, 'depth'))
                fn = f'depth_{idx:06d}.png'
                fp = os.path.join(cam_dir, 'depth', fn)
                cv2.imwrite(fp, out)
                meta['depth'].append({
                    'depth_index': dep_i,
                    'topic': self.depth_topics[dep_i],
                    'stamp_ns': int(t_ns),
                    'file': os.path.relpath(fp, self.session_dir),
                    'note': scale_info,
                })

        # --- JointState ---
        js: JointState = js_msg
        meta['joint'].update({
            'name': list(js.name),
            'position': [float(x) for x in js.position],
            'velocity': [float(x) for x in js.velocity],
            'effort': [float(x) for x in js.effort],
        })

        # --- 触觉 ---
        tm: Float32MultiArray = tact_msg
        meta['tactile'].update({
            'data': [float(x) for x in tm.data]
        })

        # --- 写 meta ---
        meta_fp = os.path.join(self.meta_dir, f'meta_{idx:06d}.json')
        with open(meta_fp, 'w') as f:
            json.dump(meta, f, indent=2)

        # 周期日志
        if idx % int(max(1, self.rate_hz)) == 0:
            self.get_logger().info(f"saved idx={idx} @t_ref={t_ref} ns")

    # ---------- 关闭 ----------
    def destroy_node(self):
        self._stop_evt.set()
        try:
            if self.worker.is_alive():
                self.worker.join(timeout=3.0)
        except Exception:
            pass
        return super().destroy_node()


# ====================== main：多线程执行器 ======================

def main(args=None):
    rclpy.init(args=args)
    node = Manager()
    try:
        from rclpy.executors import MultiThreadedExecutor
        executor = MultiThreadedExecutor(num_threads=os.cpu_count() or 4)
        executor.add_node(node)
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
